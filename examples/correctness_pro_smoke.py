"""T06 GPU acceptance smoke: the extended correctness track executed
through the container boundary on LayerNorm (l1 p040).

Cases: the legal Triton candidate must pass everything (sanitizer pass);
each counterexample must be rejected by exactly its matching check; one
legal case runs with a missing sanitizer binary to prove ``not_run`` is a
coverage gap rather than a pass. Report:
artifacts/t06/correctness-pro-report.json; exit 0 requires every case to
match its expectation."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import run_pro_case

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")

# (case_id, fixture, expected_verdict, sanitizer_binary)
CASES = [
    ("legal", "correct_l1_p40_layernorm", True, None),
    ("constant", "pro_constant_l1_p40_layernorm", False, None),
    ("mutating", "pro_mutating_l1_p40_layernorm", False, None),
    ("wrong-tail", "pro_wrong_tail_l1_p40_layernorm", False, None),
    ("stale-state", "pro_stale_l1_p40_layernorm", False, None),
    ("legal-no-sanitizer", "correct_l1_p40_layernorm", False, "compute-sanitizer-missing"),
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t06"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    report_cases = []
    all_ok = True
    for case_id, fixture, expected, sanitizer_binary in CASES:
        result = run_pro_case(
            case_id=case_id,
            level=1,
            problem_id=40,
            problem_name="40_LayerNorm.py",
            candidate_name=fixture,
            candidate_source=(FIXTURES_DEFAULT / f"{fixture}.py").read_text(encoding="utf-8"),
            snapshot_root=args.snapshot_root,
            workspace_root=args.output / "workspace",
            gpu_devices=(gpu_device,),
            sanitizer_binary=sanitizer_binary,
        )
        matched = result["verdict"] == expected
        all_ok = all_ok and matched
        report_cases.append({**result, "expected_verdict": expected, "matched": matched})
        print(
            f"{case_id}: verdict={result['verdict']} expected={expected} matched={matched} "
            f"sanitizer={result['checks'].get('sanitizer', {}).get('status')}"
        )

    report = {
        "protocol": "t06-correctness-pro-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "image_id": "sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b",
        "cases": report_cases,
        "accepted": all_ok,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "correctness-pro-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"report={report_path} accepted={all_ok}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
