"""T20 acceptance smoke: CUDA tuning via Kernel Tuner plus the CUTLASS
template path, both real GPU runs inside the diagnostic container.
Report: artifacts/t20/t20-report.json."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    execute_container,
)

DRIVER = Path("configs/t20/t20_driver.py")
CUTLASS_CU = Path("configs/t20/cutlass_gemm.cu")
EVAL_IMAGE_ID = "sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b"
T20_IMAGE_BUILD = Path("configs/t20/Dockerfile")


def _t20_image_id() -> str:
    """The T20 layer image is built locally from configs/t20/Dockerfile on
    the pinned base; its content-addressed ID is recorded in
    configs/t20/image_id at build time."""
    id_file = Path("configs/t20/image_id")
    if not id_file.exists():
        raise SystemExit(
            "configs/t20/image_id missing: build the T20 image first:\n"
            "  docker build -t kernelagent-eval:t20 configs/t20\n"
            "  docker inspect kernelagent-eval:t20 --format '{{.Id}}' > configs/t20/image_id"
        )
    return id_file.read_text(encoding="utf-8").strip()


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t20"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("research/sources/cutlass"))
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    cutlass_root = args.snapshot_root
    include_ok = (cutlass_root / "include/cutlass/cutlass.h").is_file() or (
        cutlass_root / "include" / "cutlass" / "cutlass.h"
    ).exists()

    inputs = args.output / "inputs"
    cutlass_dir = inputs / "cutlass"
    (cutlass_dir / "include").mkdir(parents=True, exist_ok=True)
    # Stage the pinned CUTLASS headers (include tree only) plus the template.
    src_include = cutlass_root / "include"
    if src_include.is_dir():
        import shutil

        shutil.copytree(src_include, cutlass_dir / "include", dirs_exist_ok=True)
    (inputs / "cutlass_gemm.cu").write_text(CUTLASS_CU.read_text(encoding="utf-8"))
    (inputs / "t20_driver.py").write_text(DRIVER.read_text(encoding="utf-8"))
    (inputs / "case.json").write_text(
        json.dumps({"device": 0, "toolkit_root": "/usr/local/cuda"}, indent=2),
        encoding="utf-8",
    )

    request = WorkerRequest(
        request_id="t20-cuda-cutlass",
        argv=("python3", "/task/t20_driver.py"),
        timeout_seconds=1500.0,
        workspace_root=args.output,
        env_extra=(("PATH", "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"),),
    )
    spec = ContainerSpec(
        image="kernelagent-eval",
        image_id=_t20_image_id(),
        memory_bytes=8 * 1024 * 1024 * 1024,
        pids_limit=256,
        read_only_mounts=(("/task", inputs),),
        gpu_devices=(gpu_device,),
    )
    outcome = execute_container(request, spec)
    result_file = Path(outcome.workdir) / "out" / "t20_result.json"
    if not result_file.is_file():
        print(f"t20 driver produced no result: {outcome.stderr_tail[-800:]}")
        return 1
    result = json.loads(result_file.read_text(encoding="utf-8"))

    tuning = result.get("cuda_tuning", {})
    cutlass = result.get("cutlass", {})
    cuda_ok = (
        tuning.get("configs_measured", 0) >= 5
        and bool(tuning.get("winner_config"))
        and tuning.get("wrapper_retest_ok") is True
    )
    cutlass_ok = (
        cutlass.get("compiled") is True
        and cutlass.get("ran") is True
        and "cutlass-ok" in cutlass.get("stdout_tail", "")
    )
    accepted = cuda_ok and cutlass_ok

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "protocol": "t20-cuda-cutlass-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "image_id": EVAL_IMAGE_ID,
        "dockerfile": str(T20_IMAGE_BUILD),
        "cutlass_include_staged": include_ok,
        "licenses": {
            "kernel-tuner": "GPLv3 (container-only, not redistributed)",
            "cutlass": "BSD-3-Clause (C++ headers)",
        },
        "cuda_tuning": tuning,
        "cutlass": cutlass,
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "t20-report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    print(f"cuda_ok={cuda_ok} cutlass_ok={cutlass_ok} accepted={accepted}")
    print(f"report={report_path}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
