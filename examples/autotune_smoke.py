"""T14 acceptance smoke: a real Triton autotune sweep (per-config
measurement, prune, in-place restore, frozen winner re-verification)
through the container boundary on LayerNorm. Report:
artifacts/t14/autotune-report.json."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.tuning import audit_sweep, select_winner
from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    execute_container,
)

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
EVAL_IMAGE_ID = "sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b"
DRIVER = Path("configs/kernelbench/autotune_driver.py")
CANDIDATE = Path("configs/kernelbench/eval-fixtures/autotune_l1_p40_layernorm.py")

CONFIGS = [
    {"block_size": 1024},
    {"block_size": 2048},
    {"block_size": 512},
    {"block_size": 3000},  # non-power-of-two -> structural prune
    {"block_size": 1048576},  # compiles? no -> resource prune
]


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t14"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    problem_rel = "KernelBench/level1/40_LayerNorm.py"
    inputs = args.output / "inputs"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / "KernelBench/level1"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (evaluator_dir / harness).write_bytes(
            (args.snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        )
    (problem_dir / "40_LayerNorm.py").write_bytes((args.snapshot_root / problem_rel).read_bytes())
    (inputs / "candidate.py").write_text(CANDIDATE.read_text(encoding="utf-8"))
    case_config = {
        "problem_path": problem_rel,
        "device": 0,
        "seed": 42,
        "configs": CONFIGS,
        "protocol": {
            "warmup_iters": 3,
            "num_batches": 5,
            "iters_per_batch": 10,
        },
    }
    (inputs / "case.json").write_text(json.dumps(case_config, indent=2), encoding="utf-8")
    (inputs / "autotune_driver.py").write_text(DRIVER.read_text(encoding="utf-8"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    request = WorkerRequest(
        request_id="autotune-sweep",
        argv=("python3", "/task/autotune_driver.py"),
        timeout_seconds=600.0,
        workspace_root=args.output,
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
    assert outcome.workdir is not None
    payload = json.loads((Path(outcome.workdir) / "out" / "autotune.json").read_text())

    statuses = {c["config"]["block_size"]: c["status"] for c in payload["configs"]}
    winner = select_winner(payload["configs"])
    audit_ok, audit_reason = audit_sweep(payload)
    expected_statuses = {
        3000: "pruned",
        1048576: "pruned",
    }
    prunes_ok = all(statuses.get(k) == v for k, v in expected_statuses.items())
    winner_is_correct_config = winner in (
        {"block_size": 1024},
        {"block_size": 2048},
        {"block_size": 512},
    )
    accepted = (
        outcome.status == "completed"
        and audit_ok
        and prunes_ok
        and winner_is_correct_config
        and payload.get("frozen_verified") is True
        and payload.get("input_restore_events", 0) > 0
    )
    payload_out = {
        "protocol": "t14-autotune-smoke-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "statuses": {str(k): v for k, v in statuses.items()},
        "winner": winner,
        "frozen_verified": payload.get("frozen_verified"),
        "input_restore_events": payload.get("input_restore_events"),
        "audit": {"ok": audit_ok, "reason": audit_reason},
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "autotune-report.json"
    report_path.write_text(
        json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {k: payload_out[k] for k in ("statuses", "winner", "frozen_verified", "accepted")},
            indent=2,
        )
    )
    print(f"report={report_path}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
