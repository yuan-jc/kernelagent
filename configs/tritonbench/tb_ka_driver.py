# noqa: I001 - runtime hook keeps torch before triton deliberately
"""T18 driver: registers a project-authored backend into the pinned
tritonbench vector_add operator via the upstream registration API
(REGISTERED_BENCHMARKS - no upstream file is modified), then
1. runs the upstream runner in-process (its latency table goes to
   stdout and must contain the registered backend), and
2. times baseline / upstream-config / plugin-config callables with the
   T07 CUDA-event protocol (raw batches to /out/tb_ka.json)."""

import json
import os
import sys

TASK_ROOT = "/task"


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["HOME"] = "/tmp"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    sys.path.insert(0, f"{TASK_ROOT}/tritonbench")
    sys.path.insert(0, TASK_ROOT)

    import torch
    import triton
    from tritonbench.operators.vector_add.kernels import triton_add_kernel
    from tritonbench.operators.vector_add.operator import Operator as VectorAddOperator
    from tritonbench.utils.triton_op import (
        REGISTERED_BENCHMARKS,
        register_benchmark_mannually,
    )

    def ka_vector_add_sm89(self, x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        def _inner():
            triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4096)
            return output

        return _inner

    # Runtime plugin registration: attaches our backend to the pinned
    # upstream operator class without modifying any upstream file.
    VectorAddOperator.ka_vector_add_sm89 = ka_vector_add_sm89
    register_benchmark_mannually(operator_name="vector_add", func_name="ka_vector_add_sm89")

    registered_before_run = "ka_vector_add_sm89" in REGISTERED_BENCHMARKS.get("vector_add", {})

    # Same-protocol comparison (T07 events) on the fixed workload.
    device = torch.device(f"cuda:{case['device']}")
    size = case["size"]
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    output = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    def make(fn):
        def _inner():
            fn()

        return _inner

    candidates = {
        "baseline_torch_add": make(lambda: torch.add(x, y)),
        "triton_add_block1024_upstream": make(
            lambda: triton_add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
        ),
        "triton_add_block4096_ka_variant": make(
            lambda: triton_add_kernel[grid](x, y, output, size, BLOCK_SIZE=4096)
        ),
    }
    protocol = case["protocol"]
    entries = []
    for name, fn in candidates.items():
        fn()
        torch.cuda.synchronize()
        for _ in range(protocol["warmup_iters"]):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(protocol["num_batches"]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(protocol["iters_per_batch"]):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / protocol["iters_per_batch"])
        entries.append(
            {
                "backend": name,
                "batch_samples_ms": samples,
                "median_ms": sorted(samples)[len(samples) // 2],
            }
        )

    # Run the upstream runner in-process AFTER registration: its table is
    # the direct upstream comparison and must include our backend.
    import io
    import runpy
    from contextlib import redirect_stdout

    sys.argv = ["run.py", "--op", "vector_add", "--device", "cuda"]
    captured = io.StringIO()
    with redirect_stdout(captured):
        runpy.run_path("/tritonbench/run.py", run_name="__main__")
    upstream_table = captured.getvalue()

    payload = {
        "protocol": "t18-same-protocol-v1",
        "registered_before_run": registered_before_run,
        "upstream_table_contains_ka_backend": "ka_vector_add_sm89" in upstream_table,
        "upstream_table_tail": upstream_table[-1200:],
        "entries": entries,
    }
    with open("/out/tb_ka.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
