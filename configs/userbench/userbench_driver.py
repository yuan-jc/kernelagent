"""user-bench worker driver (userbench-correctness-v1 + userbench-timing-v1).

Runs INSIDE the worker boundary (container or process executor); the trusted
parent stages a single read-only mount at /task with exactly five files:

  /task/problem.yaml          the loader-validated problem definition
  /task/reference.py          the user reference callable (sha256 verified)
  /task/candidate.py          the candidate source (same entry symbol)
  /task/case.json             workload params + tolerance + timing protocol
  /task/userbench_driver.py   this driver

The candidate never judges itself: correctness is decided here (in trusted
driver code, not candidate code) by comparing against the reference output
under the problem's frozen tolerance policy, and timing uses CUDA events
with the suite's frozen warmup/batch parameters. The parent re-verifies the
payload: raw batch samples, tensor hashes and the protocol identity are all
written to /out/result.json.

Status values: passed | incorrect | build_failed | timeout |
resource_exceeded | inconclusive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import traceback

TASK_ROOT = "/task"
OUT_PATH = "/out/result.json"

DTYPE_TABLE = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}


def _torch_dtype(dtype: str):
    import torch

    return getattr(torch, DTYPE_TABLE[dtype])


def _tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materialize_inputs(case: dict):
    """Deterministic input materialization from the frozen generation spec."""
    import torch

    workload = case["workload"]
    generation = case["input_generation"]
    scale = float(generation.get("scale", 1.0))
    offset = float(generation.get("offset", 0.0))
    distribution = generation.get("distribution", "randn")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(workload["seed"]))
    tensors = []
    for shape, dtype in zip(workload["shapes"], workload["dtypes"]):
        torch_dtype = _torch_dtype(dtype)
        numel = 1
        for dim in shape:
            numel *= dim
        if distribution == "randn":
            tensor = torch.randn(*shape, device="cuda", dtype=torch_dtype, generator=generator)
            tensor = tensor * scale + offset
        elif distribution == "rand":
            tensor = torch.rand(*shape, device="cuda", dtype=torch_dtype, generator=generator)
            tensor = tensor * scale + offset
        elif distribution == "uniform":
            # deterministic uniform on [offset, offset + scale]
            tensor = torch.rand(*shape, device="cuda", dtype=torch_dtype, generator=generator)
            tensor = tensor * scale + offset
        elif distribution == "zeros":
            tensor = torch.zeros(*shape, device="cuda", dtype=torch_dtype)
        elif distribution == "ones":
            tensor = torch.ones(*shape, device="cuda", dtype=torch_dtype)
        elif distribution == "arange":
            tensor = (
                torch.arange(numel, device="cuda", dtype=torch_dtype).reshape(shape).contiguous()
            )
        else:  # loader whitelist prevents this; defensive only
            raise ValueError(f"unsupported distribution {distribution!r}")
        tensors.append(tensor)
    return tensors


def _compare(ref_output, new_output, tolerance: dict) -> tuple[bool, str]:
    import torch

    policy = tolerance.get("policy", "allclose")
    if isinstance(ref_output, torch.Tensor) and isinstance(new_output, torch.Tensor):
        if ref_output.shape != new_output.shape:
            expected = tuple(ref_output.shape)
            got = tuple(new_output.shape)
            return False, f"shape mismatch: expected {expected}, got {got}"
        if policy == "exact":
            return bool(torch.equal(ref_output, new_output)), ""
        ok = torch.allclose(
            ref_output,
            new_output,
            rtol=float(tolerance.get("rtol", 0.0)),
            atol=float(tolerance.get("atol", 0.0)),
            equal_nan=bool(tolerance.get("equal_nan", False)),
        )
        if not ok:
            diff = (ref_output - new_output).abs().max().item()
            return False, f"allclose failed (max abs diff {diff!r})"
        return True, ""
    if isinstance(ref_output, (tuple, list)) and isinstance(new_output, (tuple, list)):
        if len(ref_output) != len(new_output):
            return False, "output arity mismatch"
        for index, (ref_item, new_item) in enumerate(zip(ref_output, new_output)):
            ok, note = _compare(ref_item, new_item, tolerance)
            if not ok:
                return False, f"output[{index}]: {note}"
        return True, ""
    if ref_output != new_output:
        return False, f"scalar mismatch: expected {ref_output!r}, got {new_output!r}"
    return True, ""


def _flatten_outputs(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        flat = []
        for item in value:
            flat.extend(_flatten_outputs(item))
        return flat
    return []


def _time_candidate(entry, inputs, timing: dict) -> list[float]:
    """userbench-timing-v1: CUDA events, warmup then num_batches batches of
    iters_per_batch calls; each batch sample is one event-pair elapsed ms.
    Raw batch samples are reported; the parent computes statistics."""
    import torch

    warmup = int(timing.get("warmup_iters", 10))
    num_batches = int(timing.get("num_batches", 4))
    iters_per_batch = int(timing.get("iters_per_batch", 500))
    with torch.no_grad():
        for _ in range(warmup):
            entry(*inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(num_batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters_per_batch):
                entry(*inputs)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return samples


def main() -> int:
    import torch

    case = json.load(open(f"{TASK_ROOT}/case.json"))
    payload = {
        "protocol": case.get("protocol", {}),
        "task_id": case.get("task_id"),
        "workload_id": case.get("workload", {}).get("workload_id"),
        "status": "inconclusive",
        "correct": False,
        "notes": [],
        "timing_batch_samples_ms": [],
    }
    try:
        torch.cuda.init()
        payload["device"] = torch.cuda.get_device_name(0)

        inputs = _materialize_inputs(case)
        payload["input_sha256"] = [_tensor_sha256(t) for t in inputs]

        tolerance = case["tolerance"]
        entry_name = case["reference_entry"]

        reference_module = _import_module("userbench_reference", f"{TASK_ROOT}/reference.py")
        reference_entry = getattr(reference_module, entry_name)
        with torch.no_grad():
            reference_outputs = reference_entry(*inputs)
        torch.cuda.synchronize()
        payload["reference_output_sha256"] = [
            _tensor_sha256(t) for t in _flatten_outputs(reference_outputs)
        ]

        candidate_module = _import_module("userbench_candidate", f"{TASK_ROOT}/candidate.py")
        if not hasattr(candidate_module, entry_name):
            payload["status"] = "build_failed"
            payload["notes"].append(f"candidate entry {entry_name!r} missing")
            raise _PayloadExit(payload)
        candidate_entry = getattr(candidate_module, entry_name)
        candidate_inputs = [t.clone() for t in inputs]
        with torch.no_grad():
            candidate_outputs = candidate_entry(*candidate_inputs)
        torch.cuda.synchronize()
        payload["candidate_output_sha256"] = [
            _tensor_sha256(t) for t in _flatten_outputs(candidate_outputs)
        ]

        ok, note = _compare(reference_outputs, candidate_outputs, tolerance)
        payload["correct"] = ok
        if not ok:
            payload["status"] = "incorrect"
            payload["notes"].append(note)
            raise _PayloadExit(payload)

        payload["timing_batch_samples_ms"] = _time_candidate(
            candidate_entry, [t.clone() for t in inputs], case["timing_params"]
        )
        payload["status"] = "passed"
    except _PayloadExit:
        pass
    except torch.cuda.OutOfMemoryError as exc:  # noqa: F821 - torch imported above
        payload["status"] = "resource_exceeded"
        payload["notes"].append(f"cuda oom: {exc}")
    except Exception as exc:  # noqa: BLE001 - honest terminal states, never a pass
        status = payload.get("status", "inconclusive")
        if status != "build_failed":
            status = "build_failed" if _is_import_or_attribute_error(exc) else "inconclusive"
        payload["status"] = status
        payload["notes"].append(f"{type(exc).__name__}: {exc}")
        payload["traceback_tail"] = traceback.format_exc()[-2000:]
    finally:
        os.makedirs("/out", exist_ok=True)
        with open(OUT_PATH, "w") as handle:
            json.dump(payload, handle, indent=2)
    return 0


class _PayloadExit(Exception):
    """Raised to short-circuit after a terminal payload is already set."""

    def __init__(self, payload: dict):
        super().__init__(payload.get("status", "inconclusive"))
        self.payload = payload


def _is_import_or_attribute_error(exc: Exception) -> bool:
    return isinstance(exc, (ImportError, AttributeError, SyntaxError))


if __name__ == "__main__":
    sys.exit(main())
