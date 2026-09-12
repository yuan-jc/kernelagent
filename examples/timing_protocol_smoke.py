"""T07 GPU acceptance smoke: the formal timing protocol (design §9.3)
executed through the T04 container boundary on LayerNorm.

Cases:
- aa-eager-1 / aa-eager-2: eager vs eager under two protocol seeds (A/A).
  The bootstrap ratio CI over independent batches must cover 1.0.
- compile: torch.compile role under the same protocol.
- triton-candidate: Triton LayerNorm candidate under the same protocol
  (upstream tempfile load path).
- async-leak: side-stream candidate that returns before its work
  completes; the protocol must flag async_leak.

Report: artifacts/t07/timing-protocol-report.json; exit 0 requires every
case to satisfy its expectation and the protocol identity checks."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import (
    EVAL_IMAGE_ID,
    TimingCase,
    TimingProtocol,
    bootstrap_ratio_ci,
    run_timing_case,
)

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")


def _fixture(name: str) -> str:
    return (FIXTURES_DEFAULT / f"{name}.py").read_text(encoding="utf-8")


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t07"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    results = []
    protocol_aa1 = TimingProtocol(seed=42)
    protocol_aa2 = TimingProtocol(seed=43)

    aa_specs = [
        ("aa-eager-1", "eager", "correct_l1_p40_layernorm", protocol_aa1),
        ("aa-eager-2", "eager", "correct_l1_p40_layernorm", protocol_aa2),
        (
            "compile",
            "compile",
            "correct_l1_p40_layernorm",
            protocol_aa1,
        ),
        (
            "triton-candidate",
            "candidate",
            "triton_l1_p40_layernorm",
            protocol_aa1,
        ),
        (
            "async-leak",
            "candidate",
            "async_leak_l1_p40_layernorm",
            protocol_aa1,
        ),
    ]
    cases = {
        case_id: TimingCase(
            case_id=case_id,
            level=1,
            problem_id=40,
            problem_name="40_LayerNorm.py",
            candidate_name=fixture,
            candidate_source=_fixture(fixture),
            role=role,
            backend="triton" if case_id == "triton-candidate" else "cuda",
            expect_async_leak=(case_id == "async-leak"),
        )
        for case_id, role, fixture, _ in aa_specs
    }
    for case_id, _role, _fixture_name, protocol in aa_specs:
        result = run_timing_case(
            cases[case_id],
            protocol,
            snapshot_root=args.snapshot_root,
            workspace_root=args.output / "workspace",
            gpu_devices=(gpu_device,),
        )
        results.append((case_id, protocol, result))
        print(
            f"{case_id}: outcome={result.outcome_status} source_ok={result.source_ok} "
            f"precondition={result.correctness_precondition} "
            f"batches={len(result.batch_samples_ms)} async_leak={result.async_leak}"
        )

    by_id = {case_id: result for case_id, _, result in results}
    aa1, aa2 = by_id["aa-eager-1"], by_id["aa-eager-2"]
    aa_ci = None
    if aa1.batch_samples_ms and aa2.batch_samples_ms:
        aa_ci = bootstrap_ratio_ci(list(aa1.batch_samples_ms), list(aa2.batch_samples_ms))
        print(f"A/A ratio 95% CI: {aa_ci}")

    report_cases = {}
    all_ok = True
    for case_id, protocol, result in results:
        ok = (
            result.outcome_status == "completed"
            and result.source_ok
            and result.protocol_hash_ok
            and result.correctness_precondition
            and len(result.batch_samples_ms) == protocol.num_batches
            and result.async_leak_matches_expectation
        )
        report_cases[case_id] = {
            "outcome_status": result.outcome_status,
            "source_ok": result.source_ok,
            "protocol_hash_ok": result.protocol_hash_ok,
            "correctness_precondition": result.correctness_precondition,
            "batch_samples_ms": list(result.batch_samples_ms),
            "single_shot_sync_ms": result.single_shot_sync_ms,
            "async_leak": result.async_leak,
            "async_leak_matches_expectation": result.async_leak_matches_expectation,
            "ok": ok,
            "detail": result.detail,
        }
        all_ok = all_ok and ok

    aa_ok = aa_ci is not None and aa_ci[0] <= 1.0 <= aa_ci[1]
    all_ok = all_ok and aa_ok
    report = {
        "protocol_name": "t07-timing-protocol-v1",
        "commit": commit,
        "snapshot_root": str(args.snapshot_root),
        "gpu_device": gpu_device,
        "image_id": EVAL_IMAGE_ID,
        "timing_protocol": protocol_aa1.to_dict(),
        "timing_protocol_sha256": protocol_aa1.identity_sha256(),
        "source": "cuda_event",
        "aa_ratio_ci_95": list(aa_ci) if aa_ci else None,
        "aa_ci_covers_1": aa_ok,
        "cases": report_cases,
        "accepted": all_ok,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "timing-protocol-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"report={report_path} accepted={all_ok}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
