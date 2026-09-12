"""T05 GPU acceptance smoke: the pinned KernelBench upstream evaluator run
through the T04 container boundary (ADR-0001/0002) on three problem
categories with known-correct and known-wrong candidates plus a
torch.compile baseline.

Every case must match its expectation AND agree with the upstream verdict
(adapter consistency). The combined report is written to
<output>/kernelbench-eval-report.json; exit 0 requires all cases accepted."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import EvalCase, evaluate_case

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")


def _fixture(name: str) -> str:
    return (FIXTURES_DEFAULT / f"{name}.py").read_text(encoding="utf-8")


def _cases() -> list[EvalCase]:
    matrix = [
        # (case_id, level, problem_id, problem_name, fixture, expect_pass)
        (
            "correct-matmul",
            1,
            1,
            "1_Square_matrix_multiplication_.py",
            "correct_l1_p1_matmul",
            True,
        ),
        (
            "wrong-matmul",
            1,
            1,
            "1_Square_matrix_multiplication_.py",
            "wrong_l1_p1_matmul",
            False,
        ),
        (
            "correct-layernorm",
            1,
            40,
            "40_LayerNorm.py",
            "correct_l1_p40_layernorm",
            True,
        ),
        (
            "wrong-layernorm",
            1,
            40,
            "40_LayerNorm.py",
            "wrong_l1_p40_layernorm",
            False,
        ),
        (
            "correct-conv2d-relu-biasadd",
            2,
            1,
            "1_Conv2D_ReLU_BiasAdd.py",
            "correct_l2_p1_conv2d_relu_biasadd",
            True,
        ),
        (
            "wrong-conv2d-relu-biasadd",
            2,
            1,
            "1_Conv2D_ReLU_BiasAdd.py",
            "wrong_l2_p1_conv2d_relu_biasadd",
            False,
        ),
        (
            "compile-baseline-layernorm",
            1,
            40,
            "40_LayerNorm.py",
            "compile_l1_p40_layernorm",
            True,
        ),
    ]
    return [
        EvalCase(
            case_id=case_id,
            level=level,
            problem_id=problem_id,
            problem_name=problem_name,
            candidate_name=fixture,
            candidate_source=_fixture(fixture),
            expect_pass=expect_pass,
        )
        for case_id, level, problem_id, problem_name, fixture, expect_pass in matrix
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t05"))
    parser.add_argument(
        "--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT, help="Verified snapshot directory"
    )
    parser.add_argument(
        "--gpu-device", default=None, help="CDI device name (default: UUID via nvidia-smi)"
    )
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    workspace = args.output / "workspace"
    cases = []
    for case in _cases():
        result = evaluate_case(
            case,
            snapshot_root=args.snapshot_root,
            workspace_root=workspace,
            gpu_devices=(gpu_device,),
        )
        entry = {
            "case_id": result.case_id,
            "task_id": result.task_id,
            "outcome_status": result.outcome_status,
            "exit_code": result.exit_code,
            "adapter_pass": result.adapter_pass,
            "upstream_compiled": result.upstream_compiled,
            "upstream_correctness": result.upstream_correctness,
            "consistent_with_upstream": result.consistent_with_upstream,
            "matches_expectation": result.matches_expectation,
            "problem_sha256": result.problem_sha256,
            "candidate_sha256": result.candidate_sha256,
            "detail": result.detail,
        }
        cases.append(entry)
        print(
            f"{result.case_id}: outcome={result.outcome_status} pass={result.adapter_pass} "
            f"expect={result.detail['expect_pass']} "
            f"accepted={result.matches_expectation and result.consistent_with_upstream}"
        )

    all_accepted = all(c["matches_expectation"] and c["consistent_with_upstream"] for c in cases)
    report = {
        "protocol": "t05-kernelbench-eval-v1",
        "commit": commit,
        "snapshot_root": str(args.snapshot_root),
        "gpu_device": gpu_device,
        "image_id": "sha256:bb4ddb1e04d2662ef5c684d8ce12bda1ad8b87c87d9979dedb0642c749c16874",
        "note": "correctness only (measure_performance=False); formal timing is T07",
        "cases": cases,
        "accepted": all_accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "kernelbench-eval-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"report={report_path} accepted={all_accepted}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if all_accepted else 1


if __name__ == "__main__":
    sys.exit(main())
