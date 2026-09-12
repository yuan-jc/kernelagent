"""T10 acceptance smoke: the orchestrator drives the REAL evaluation /
timing / promotion chain (T05/T07/T09 adapters) over the Triton LayerNorm
candidate through the container boundary, with crash recovery.

Phase 1 (default, only when no journal exists): run with
``--crash-after evaluate`` - the orchestrator writes ``action_started``
for the timing action and the process dies abruptly (os._exit). Exit
code 9 is the expected crash signature.

Phase 2 (``--resume``): recovery marks the orphaned action interrupted,
re-runs it and the remaining actions for real, settles all budgets, and
writes artifacts/t10/orchestrator-report.json. Exit 0 requires the
interrupted marker, a completed experiment, and honest billing of every
action including the one that was re-run."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import (
    EvalCase,
    TimingCase,
    TimingProtocol,
    evaluate_case,
    run_timing_case,
)
from kernelagent.orchestrator import Action, Budget, Orchestrator
from kernelagent.promotion import TERMINAL_PROMOTE, CandidateFacts, confirm_promotion

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")
TRITON_FIXTURE = "triton_l1_p40_layernorm"


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t10"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    parser.add_argument(
        "--crash-after", default=None, help="Die abruptly after this action (phase 1)"
    )
    parser.add_argument("--resume", action="store_true", help="Phase 2: recover from the journal")
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()
    args.output.mkdir(parents=True, exist_ok=True)
    journal_path = args.output / "journal.jsonl"
    workspace = args.output / "workspace"

    candidate_source = _fixture(TRITON_FIXTURE)
    protocol = TimingProtocol(warmup_iters=5, num_batches=6, iters_per_batch=10)

    results: dict[str, dict] = {}

    def make_action(name: str, work) -> Action:
        def run() -> dict:
            if args.crash_after == name:
                # Simulated hard crash AFTER the started event is durable.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(9)
            result = work()
            results[name] = result
            return result

        return Action(
            name=name,
            input_hash=hashlib.sha256(candidate_source.encode()).hexdigest()[:16] + f":{name}",
            estimated_gpu_seconds=300.0,
            estimated_tokens=0,
            run=run,
        )

    def do_evaluate() -> dict:
        outcome = evaluate_case(
            EvalCase(
                case_id="orchestrator-eval",
                level=1,
                problem_id=40,
                problem_name="40_LayerNorm.py",
                candidate_name=TRITON_FIXTURE,
                candidate_source=candidate_source,
                backend="triton",
            ),
            snapshot_root=args.snapshot_root,
            workspace_root=workspace,
            gpu_devices=(gpu_device,),
        )
        ok = outcome.outcome_status == "completed" and outcome.adapter_pass is True
        if not ok:
            print(f"evaluation failed: {outcome}")
            raise SystemExit(3)
        return {"result_ref": "eval-verified"}

    def do_time(role: str, backend: str):
        def work() -> dict:
            timing_case = TimingCase(
                case_id=f"orchestrator-time-{role}",
                level=1,
                problem_id=40,
                problem_name="40_LayerNorm.py",
                candidate_name=TRITON_FIXTURE if role == "candidate" else "eager-role",
                candidate_source=candidate_source
                if role == "candidate"
                else _fixture("correct_l1_p40_layernorm"),
                role="candidate" if role == "candidate" else "eager",
                backend=backend,
            )
            result = run_timing_case(
                timing_case,
                protocol,
                snapshot_root=args.snapshot_root,
                workspace_root=workspace,
                gpu_devices=(gpu_device,),
            )
            if result.outcome_status != "completed" or not result.batch_samples_ms:
                print(f"timing failed for {role}: {result.detail}")
                raise SystemExit(4)
            results[f"time-{role}"] = {
                "batches": list(result.batch_samples_ms),
                "async_leak": result.async_leak,
            }
            return {"result_ref": f"time-{role}", "gpu_seconds": 60.0}

        return work

    def do_confirm() -> dict:
        decision = confirm_promotion(
            facts=CandidateFacts(compiled=True, correct=True),
            incumbent_batches_ms=results["time-eager"]["batches"],
            candidate_batches_ms=results["time-candidate"]["batches"],
            protocol_sha256=protocol.identity_sha256(),
        )
        results["confirm"] = {"outcome": decision.outcome, "reason": decision.reason}
        return {"result_ref": f"confirm:{decision.outcome}"}

    actions = [
        make_action("evaluate", do_evaluate),
        make_action("time_candidate", do_time("candidate", "triton")),
        make_action("time_incumbent", do_time("eager", "cuda")),
        make_action("confirm", do_confirm),
    ]

    budget = Budget(gpu_seconds_limit=1800.0, tokens_limit=0)
    orchestrator = Orchestrator(journal_path, budget)
    report = orchestrator.run(actions, resume=args.resume)
    if not args.resume:
        # Phase 1 only reaches here when --crash-after named a missing action.
        print(json.dumps(report))
        return 5

    kinds = [entry["kind"] for entry in orchestrator.journal.entries]
    interrupted = "action_interrupted" in kinds
    completed = report["state"] == "completed"
    # Reconstruct the outcome from the durable journal, not from this
    # process's memory: a resumed run may have skipped finished actions.
    confirm_refs = [
        entry.get("result_ref")
        for entry in orchestrator.journal.entries
        if entry["kind"] == "action_finished" and entry.get("action_id") == "confirm"
    ]
    confirm_outcome = confirm_refs[-1].split(":")[-1] if confirm_refs else None
    accepted = (
        interrupted
        and completed
        and confirm_outcome in (TERMINAL_PROMOTE, "retain_incumbent")
        and report["settled_gpu_seconds"] > 0
    )
    payload = {
        "protocol": "t10-orchestrator-v1",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "gpu_device": gpu_device,
        "interrupted_recovered": interrupted,
        "state": report["state"],
        "finished": report["finished"],
        "settled_gpu_seconds": report["settled_gpu_seconds"],
        "journal_entries": len(orchestrator.journal.entries),
        "confirm_outcome": confirm_outcome,
        "accepted": accepted,
    }
    report_path = args.output / "orchestrator-report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"report={report_path} accepted={accepted}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
