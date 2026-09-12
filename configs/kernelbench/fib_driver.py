"""FlashInfer-Bench trace driver (T19). Runs INSIDE the ADR-0001 container
boundary; the parent stages the pinned trace (definition + workload +
solution JSON), a case.json, and reads /out/fib_result.json.

Semantics (design §8.3): inputs are generated from the workload axis
bindings with a fixed seed; the definition's ``reference`` is the
baseline; the solution entry point is invoked return-style (len(params)
== len(inputs)) or DPS-style (pre-allocated output buffers appended in
definition output order); the result is verified against the reference
and timed with CUDA event batches; input/output tensor hashes and the
trace revision are recorded."""

import json
import os
import sys

TASK_ROOT = "/task"


def _dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }[name]


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["HOME"] = "/tmp"
    sys.path.insert(0, f"{TASK_ROOT}/src")

    import torch

    with open(f"{TASK_ROOT}/definition.json") as handle:
        definition = json.load(handle)
    solution = json.load(open(f"{TASK_ROOT}/solution.json"))
    workload = case["workload"]

    # Axis resolution: const axes from the definition, var axes from the
    # workload entry.
    axes = {}
    for name, spec in definition["axes"].items():
        axes[name] = spec["value"] if spec["type"] == "const" else workload["axes"][name]

    def resolve(shape):
        return [axes[dim] if isinstance(dim, str) else dim for dim in shape]

    import hashlib

    def tensor_hash(t) -> str:
        import numpy as np

        return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()

    set_seed_value = case["seed"]

    def make_inputs():
        import random

        import numpy as np

        random.seed(set_seed_value)
        np.random.seed(set_seed_value)
        torch.manual_seed(set_seed_value)
        torch.cuda.manual_seed_all(set_seed_value)
        out = []
        for name, spec in definition["inputs"].items():
            tensor = torch.randn(*resolve(spec["shape"]), device=device).to(_dtype(spec["dtype"]))
            out.append((name, tensor))
        return out

    reference_src = definition["reference"]
    ref_namespace: dict = {}
    exec(compile(reference_src, "reference.py", "exec"), ref_namespace)
    reference_run = ref_namespace["run"]

    solution_module_src = next(
        s["content"] for s in solution["sources"] if s["path"].endswith(".py")
    )
    sol_namespace: dict = {}
    exec(compile(solution_module_src, "solution.py", "exec"), sol_namespace)
    run = sol_namespace[case["entry_function"]]

    device = torch.device(f"cuda:{case['device']}")
    named_inputs = make_inputs()
    tensors = [t for _, t in named_inputs]
    input_hashes = {name: tensor_hash(t) for name, t in named_inputs}

    with torch.no_grad():
        ref_out = reference_run(*[t.clone() for t in tensors])
        if isinstance(ref_out, torch.Tensor):
            ref_out = [ref_out]
        torch.cuda.synchronize()

    output_specs = list(definition["outputs"].items())
    protocol = case["protocol"]

    def run_solution(mode: str) -> dict:
        buffers = [
            torch.empty(*resolve(spec["shape"]), device=device, dtype=_dtype(spec["dtype"]))
            for _, spec in output_specs
        ]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        if mode == "dps":
            run(*[t.clone() for t in tensors], *buffers)
        else:
            produced = run(*[t.clone() for t in tensors])
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        final = (
            buffers
            if mode == "dps"
            else [produced if isinstance(produced, torch.Tensor) else produced[0]]
        )
        hashes = {f"out{i}": tensor_hash(t) for i, t in enumerate(final)}
        return {"elapsed_ms": elapsed, "output_hashes": hashes, "outputs": final}

    verification = {}
    timings = []
    with torch.no_grad():
        for mode in case["modes"]:
            attempt = run_solution(mode)
            ok = all(
                torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
                for actual, expected in zip(attempt["outputs"], ref_out)
            )
            verification[mode] = {
                "correct": ok,
                "output_hashes": attempt["output_hashes"],
            }
            samples = []
            for _ in range(protocol["warmup_iters"]):
                run_solution(mode)
            torch.cuda.synchronize()
            for _ in range(protocol["num_batches"]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(protocol["iters_per_batch"]):
                    run_solution(mode)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / protocol["iters_per_batch"])
            timings.append({"mode": mode, "batch_samples_ms": samples})

    payload = {
        "protocol": "t19-flashinfer-bench-v1",
        "workload_uuid": workload["uuid"],
        "trace_revision": case["trace_revision"],
        "definition_sha256": case["definition_sha256"],
        "solution_sha256": case["solution_sha256"],
        "input_hashes": input_hashes,
        "verification": verification,
        "timings": timings,
        "axes": axes,
        "source": "cuda_event_timing",
    }
    with open("/out/fib_result.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
