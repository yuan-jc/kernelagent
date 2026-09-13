"""T15 acceptance smoke: real .ncu-rep generation (ADR-0003 diagnostic
lease) and interpretation on the host.

Steps:
1. Build the fixed parent-authored two-kernel binary (nvcc, host).
2. Profile it inside the privileged diagnostic container -> real report.
3. Import the report on the host (no GPU / no elevation needed) and
   derive the evidence view: per-launch metrics, launch association,
   missing-metric accounting, source tags, CC identity.
Report: artifacts/t15/ncu-evidence.json; exit 0 requires >= 3 launches,
both kernel names associated, and at least one catalogued metric present."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.profiling import (
    MISSING,
    MetricCatalog,
    associate_launches,
    evidence_view,
    import_report,
    profile_in_diagnostic_container,
)

CUDA_BIN = Path("/home/y/toolchains/cuda")
SOURCE_CU = """#include <cstdio>
__global__ void saxpy(int n, float a, float *x, float *y){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n) y[i] = a*x[i]+y[i];
}
__global__ void scale(int n, float *y){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n) y[i] *= 2.0f;
}
int main(){
  int n = 1<<20; size_t bytes = n*sizeof(float);
  float *x, *y, *hx = (float*)malloc(bytes), *hy = (float*)malloc(bytes);
  cudaMalloc(&x, bytes); cudaMalloc(&y, bytes);
  for (int i=0;i<n;i++){hx[i]=1.0f; hy[i]=2.0f;}
  cudaMemcpy(x,hx,bytes,cudaMemcpyHostToDevice); cudaMemcpy(y,hy,bytes,cudaMemcpyHostToDevice);
  saxpy<<<(n+255)/256,256>>>(n, 2.0f, x, y);
  scale<<<(n+255)/256,256>>>(n, y);
  scale<<<(n+255)/256,256>>>(n, y);
  cudaMemcpy(hy,y,bytes,cudaMemcpyDeviceToHost);
  printf("y[0]=%f\\n", hy[0]);
  return 0;
}
"""


def _nvcc() -> str:
    for candidate in (str(CUDA_BIN / "bin" / "nvcc"), shutil.which("nvcc")):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("nvcc not found on host; T15 binary build requires the host toolchain")


def _default_gpu_device() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return "nvidia.com/gpu=0"
    first = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    return f"nvidia.com/gpu={first}" if first else "nvidia.com/gpu=0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/t15"))
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    build_dir = args.output / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "two_kernels.cu"
    source.write_text(SOURCE_CU, encoding="utf-8")
    binary = build_dir / "two_kernels"
    if not binary.exists():
        completed = subprocess.run(
            [_nvcc(), "-arch=sm_89", "-o", str(binary), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(f"nvcc failed: {completed.stderr[-400:]}")
            return 2

    report_path = profile_in_diagnostic_container(
        binary_path=binary,
        output_path=args.output / "t15_two_kernels.ncu-rep",
        gpu_device=gpu_device,
    )

    ncu_bin = str(CUDA_BIN / "bin" / "ncu")
    launches = import_report(ncu_bin, report_path)
    catalog = MetricCatalog.default()
    view = evidence_view(launches, catalog)

    saxpy = associate_launches(launches, "saxpy")
    scale = associate_launches(launches, "scale")
    duration_ok = all(launch.value("gpu__time_duration.sum") is not MISSING for launch in launches)
    identity_ok = all(launch.compute_capability == "8.9" for launch in launches)
    accepted = (
        len(launches) == 3
        and len(saxpy) == 1
        and len(scale) == 2
        and duration_ok
        and identity_ok
        and view["source"] == "ncu_profile"
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "protocol": "t15-ncu-evidence-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "ncu_binary": ncu_bin,
        "launch_count": len(launches),
        "associated": {"saxpy": len(saxpy), "scale": len(scale)},
        "duration_metrics_present": duration_ok,
        "compute_capability": "8.9",
        "identity_ok": identity_ok,
        "evidence": view,
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_json = args.output / "ncu-evidence.json"
    report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n"
    )
    print(
        f"launches={len(launches)} saxpy={len(saxpy)} scale={len(scale)} duration_ok={duration_ok}"
    )
    print(f"report={report_json} accepted={accepted}")
    print(f"report_sha256={hashlib.sha256(report_json.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
