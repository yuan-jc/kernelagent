"""Formal timing driver (T07, design §9.3). Runs INSIDE the ADR-0001
container boundary.

Staged by the trusted parent (single read-only mount at /task):
  /task/src/kernelbench/*.py   pinned upstream evaluator (loader helpers)
  /task/KernelBench/levelN/..  the pinned problem file
  /task/candidate.py           the model to time (role-dependent)
  /task/case.json              TimingProtocol v1 + role configuration
and expects the raw batch samples at /out/timing.json.

Measured quantity: warmed operator-entry GPU latency via CUDA event pairs
on the current stream, one (start, end) pair per batch around
``iters_per_batch`` calls, device-wide synchronize after every batch end
event. JIT compilation / autotune / IO stay outside the window (warmup
absorbs first-call compile) but the first batch is labeled as the
cold-cache observation. Raw per-batch samples are reported unmodified -
the parent owns all statistics. The single-shot sync-bracketed wall-clock
measurement exists solely as the async-leak integrity check: a payload
that returns before its side-stream work completes shows a protocol
median far below the sync-bracketed time and is flagged ``async_leak``.
The result never claims another source: ``source`` is fixed to
``cuda_event`` (profiling time belongs to T15 and is rejected by the
control plane)."""

import json
import os
import sys
import time

TASK_ROOT = "/task"
SOURCE_TAG = "cuda_event"


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    protocol = case["protocol"]
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_cache"
    os.environ["HOME"] = "/tmp"
    sys.path.insert(0, f"{TASK_ROOT}/src")

    import torch
    from kernelbench.eval import (
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
    )

    with open(f"{TASK_ROOT}/{case['problem_path']}") as handle:
        problem_src = handle.read()
    with open(f"{TASK_ROOT}/candidate.py") as handle:
        candidate_src = handle.read()

    def set_seed(seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})
    device = torch.device(f"cuda:{case['device']}")
    set_seed(case["seed"])
    init_inputs = get_init_inputs()
    reference = Model(*init_inputs).to(device)

    role = case["role"]
    tempfile_obj = None
    if role == "candidate":
        if case.get("backend", "cuda").lower() in ("triton", "tilelang", "cute"):
            ModelNew, tempfile_obj = load_custom_model_with_tempfile(candidate_src)
            target = ModelNew(*init_inputs).to(device)
        else:
            target = load_custom_model(candidate_src, {}, "/tmp/torch_ext")
            target = target(*init_inputs).to(device)
    elif role == "compile":
        target = torch.compile(Model(*init_inputs).to(device))
    else:
        target = reference

    inputs = [t.to(device) for t in get_inputs()]

    # Correctness precondition (honest timing needs matching outputs): one
    # eager reference call vs one target call, device-synced comparison.
    with torch.no_grad():
        ref_out = reference(*inputs)
        torch.cuda.synchronize()
        tgt_out = target(*inputs)
        torch.cuda.synchronize()
        precondition = bool(torch.allclose(ref_out, tgt_out, rtol=1e-2, atol=1e-2))

    batches = []
    cold_batch_note = None
    sync_shot_ms = None
    if precondition:
        with torch.no_grad():
            for _ in range(protocol["warmup_iters"]):
                target(*inputs)
            torch.cuda.synchronize()
            for b in range(protocol["num_batches"]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if b == 0:
                    cold_start = time.perf_counter()
                start.record()
                for _ in range(protocol["iters_per_batch"]):
                    target(*inputs)
                end.record()
                torch.cuda.synchronize()
                if b == 0:
                    cold_batch_note = {
                        "wall_seconds_including_post_batch_sync": time.perf_counter() - cold_start
                    }
                batches.append(start.elapsed_time(end) / protocol["iters_per_batch"])

        # Async-leak integrity check: one call bracketed by device-wide
        # synchronize with wall clock. Side-stream work is invisible to
        # default-stream event pairs but not to the device sync.
        sync_shot_ms = None
        if case.get("integrity_check", True):
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                target(*inputs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            sync_shot_ms = (t1 - t0) * 1000.0

    sorted_batches = sorted(batches)
    median = sorted_batches[len(sorted_batches) // 2] if sorted_batches else None
    async_leak = bool(
        sync_shot_ms is not None
        and median is not None
        and median > 0
        and sync_shot_ms > max(4.0 * median, median + 0.25)
    )

    payload = {
        "source": SOURCE_TAG,
        "role": role,
        "protocol": protocol,
        "correctness_precondition": precondition,
        "batch_samples_ms": batches,
        "cold_batch_note": cold_batch_note,
        "single_shot_sync_ms": sync_shot_ms,
        "async_leak": async_leak,
        "device": str(device),
    }
    with open("/out/timing.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
