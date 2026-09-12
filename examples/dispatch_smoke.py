"""T21 acceptance smoke: multi-workload dispatch through the container
boundary on LayerNorm.

Workloads: base shape (16x64x256x256) and a 2x element-count variant with
a guard cap that the 2x workload VIOLATES (guard -> fallback), plus a
degradation-threshold retreat case. The dispatch wrapper (guard decision
+ call) itself is timed. Report: artifacts/t21/dispatch-report.json."""

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

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
DRIVER = Path("configs/kernelbench/dispatch_driver.py")
CANDIDATE = Path("configs/kernelbench/eval-fixtures/correct_l1_p40_layernorm.py")
EVAL_IMAGE_ID = "sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b"


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t21"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    inputs = args.output / "inputs"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / "KernelBench" / "level1"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (evaluator_dir / harness).write_bytes(
            (args.snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        )
    (problem_dir / "40_LayerNorm.py").write_bytes(
        (args.snapshot_root / "KernelBench/level1/40_LayerNorm.py").read_bytes()
    )
    candidate_source = CANDIDATE.read_text(encoding="utf-8")
    (inputs / "candidate.py").write_text(candidate_source, encoding="utf-8")
    case_config = {
        "problem_path": "KernelBench/level1/40_LayerNorm.py",
        "device": 0,
        "seed": 42,
        "degradation_threshold": 0.02,
        "workloads": [
            # base is 16*64*256*256 = 67108864 elements; the cap equals it
            # exactly, so the 2x workload violates the guard.
            {"workload_id": "base", "size_multiplier": 1, "max_elements": 1 << 26},
            {"workload_id": "over-cap", "size_multiplier": 2, "max_elements": 1 << 26},
        ],
    }
    (inputs / "case.json").write_text(json.dumps(case_config, indent=2), encoding="utf-8")
    (inputs / "dispatch_driver.py").write_text(DRIVER.read_text(encoding="utf-8"))

    request = WorkerRequest(
        request_id="dispatch-smoke",
        argv=("python3", "/task/dispatch_driver.py"),
        timeout_seconds=900.0,
        workspace_root=args.output,
        env_extra=(("PYTHONPATH", "/task/src"),),
    )
    spec = ContainerSpec(
        image="kernelagent-eval",
        image_id=EVAL_IMAGE_ID,
        memory_bytes=8 * 1024 * 1024 * 1024,
        pids_limit=256,
        read_only_mounts=(("/task", inputs),),
        gpu_devices=(gpu_device,),
    )
    outcome = execute_container(request, spec)
    result_file = Path(outcome.workdir) / "out" / "dispatch_result.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    workloads = {w["workload_id"]: w for w in payload["workloads"]}

    base = workloads["base"]
    over = workloads["over-cap"]
    checks = {
        "base_correct": base["correct"] and not base["degraded"],
        "base_wrapper_timed": len(base["wrapper_batch_samples_ms"]) == 5,
        "over_guard_violated": not over["guard_ok"],
        "over_fallback_used": over["fallback_used"],
        "over_fallback_correct": over["fallback_correct"] is True,
    }
    accepted = outcome.status == "completed" and all(checks.values())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload_out = {
        "protocol": "t21-dispatch-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "checks": checks,
        "workloads": payload["workloads"],
        "candidate_sha256": hashlib.sha256(candidate_source.encode("utf-8")).hexdigest(),
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "dispatch-report.json"
    report_path.write_text(
        json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(json.dumps(checks, indent=2))
    print(f"report={report_path} accepted={accepted}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
