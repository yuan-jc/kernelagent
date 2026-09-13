"""T20 driver (CUDA tuning via Kernel Tuner + CUTLASS template path).
Runs INSIDE the diagnostic container (T20 image layer of ADR-0002's
evaluation stack; see docs/work-packages/T20.md).

Part A (CUDA tuning really runs): Kernel Tuner sweeps the vector-add CUDA
kernel's BLOCK_SIZE on the real GPU; every configuration's measured time
is accounted, and the winning configuration is selected by Kernel Tuner.
Part B (wrapper retest): the winner configuration is re-run through a
wrapper that re-verifies the output on fresh inputs - the tuned parameter
must reproduce correctness outside the tuning loop.
Part C (CUTLASS template path): the pinned cutlass_gemm.cu template is
compiled with nvcc (CUDA 12.4 toolkit in the image) and executed; it
self-verifies and prints cutlass-ok.
Writes /out/t20_result.json with all three parts."""
import json

import os
import subprocess
import sys

TASK_ROOT = "/task"

KERNEL_STRING = """
__global__ void vector_add(float *y, const float *x, const float *w, int n, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + w[i];
}
"""


def part_a_kernel_tuner(device: int) -> dict:
    import numpy as np
    from kernel_tuner import tune_kernel

    size = 1 << 20
    a = 2.0
    x = np.ones(size, dtype=np.float32)
    w = np.ones(size, dtype=np.float32) * 0.5
    y = np.zeros(size, dtype=np.float32)
    args = [y, x, w, np.int32(size), a]
    tune = tune_kernel(
        "vector_add",
        KERNEL_STRING,
        size,
        args,
        {"BLOCK_SIZE": [64, 128, 256, 512, 1024]},
        grid_div_x=["BLOCK_SIZE"],
        lang="CUDA",
        quiet=True,
    )
    results, winner_config = tune
    # Wrapper retest: fresh buffers, winner config, exact verification.
    y2 = np.zeros(size, dtype=np.float32)
    args2 = [y2, x, w, np.int32(size), a]
    import kernel_tuner

    kernel_tuner.run_kernel(
        "vector_add",
        KERNEL_STRING,
        size,
        args2,
        {"BLOCK_SIZE": winner_config["BLOCK_SIZE"]},
        lang="CUDA",
        grid_div_x=["BLOCK_SIZE"],
    )
    expected = a * x + w
    wrapper_ok = bool(np.array_equal(y2, expected))
    measurements = {str(result["BLOCK_SIZE"]): result.get("time") for result in results}
    return {
        "part": "cuda_tuning",
        "tuner": "kernel-tuner",
        "license": "GPLv3 (container-only; not redistributed)",
        "configs_measured": len(results),
        "measurements_us": measurements,
        "winner_config": winner_config,
        "wrapper_retest_ok": wrapper_ok,
    }


def part_c_cutlass(toolkit: str) -> dict:
    nvcc = os.path.join(toolkit, "bin", "nvcc")
    src = os.path.join(TASK_ROOT, "cutlass_gemm.cu")
    include = os.path.join(TASK_ROOT, "cutlass", "include")
    binary = "/tmp/cutlass_gemm"
    compile_cmd = [nvcc, "-arch=sm_89", "-O2", "-I" + include, src, "-o", binary]
    compiled = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
    if compiled.returncode != 0:
        return {"part": "cutlass", "compiled": False, "error": compiled.stderr[-500:]}
    run = subprocess.run([binary], capture_output=True, text=True, timeout=120)
    return {
        "part": "cutlass",
        "compiled": True,
        "ran": run.returncode == 0,
        "stdout_tail": run.stdout.strip(),
        "license": "BSD-3-Clause (CUTLASS C++ headers; NVIDIA CuTeDSL EULA not used)",
    }


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    toolkit = case.get("toolkit_root", "/usr/local/cuda")
    payload = {"protocol": "t20-cuda-cutlass-v1"}
    try:
        payload["cuda_tuning"] = part_a_kernel_tuner(case["device"])
    except Exception as exc:  # noqa: BLE001 - record the failure mode
        payload["cuda_tuning"] = {"part": "cuda_tuning", "error": f"{type(exc).__name__}: {exc}"[:400]}
    payload["cutlass"] = part_c_cutlass(toolkit)
    with open("/out/t20_result.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
