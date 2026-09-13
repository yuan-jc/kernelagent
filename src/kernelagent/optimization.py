"""Minimal recoverable optimization loop (launch plan Task 5, T16).

One problem in, an evidence-backed champion (or an honest terminal state)
out. The loop is the product entry point behind ``kernelagent optimize``
and runs the frozen alpha order per candidate:

    generate -> policy gate -> evaluate -> correctness_pro -> timing
             -> confirm -> persist

Trust rules (ADR-0004 + design §12): candidates are untrusted (the AST
policy gate runs before any GPU work; alpha remains
candidate_trust=cooperative); verdicts come only from the injected
evaluation/timing ports backed by the pinned container evaluators; the
durable journal is the only billing truth; malformed timing evidence is a
typed error, never a crash; no-improvement, budget exhaustion, and infra
errors are recorded terminal states - never fake success.

The stage ports are injectable so the loop is testable offline; the CLI
wires the real T05/T06/T07/T09 adapters over the container boundary."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kernelagent.adapters.evals.timing import validate_batch_samples
from kernelagent.adapters.models.generation import (
    CandidateGenerator,
    GenerationFailure,
    GenerationSuccess,
    inspect_candidate_policy,
)
from kernelagent.orchestrator import (
    Action,
    Budget,
    Journal,
    Orchestrator,
    reconstruct_budget,
)
from kernelagent.promotion import (
    TERMINAL_PROMOTE,
    CandidateFacts,
    confirm_promotion,
)

OPTIMIZATION_PROTOCOL = "optimize-v1"
DEFAULT_SNAPSHOT_ROOT = Path("research/sources/ScalingIntelligence__KernelBench")
CHAMPION_TRUNC = 800

EXIT_SUCCESS = 0
EXIT_NO_IMPROVEMENT = 1
EXIT_BUDGET_EXHAUSTED = 2
EXIT_CONFIG_ERROR = 3
EXIT_INFRA_ERROR = 4

_EST_GPU_SECONDS_PER_STAGE = 300.0
_EST_TOKENS_PER_CANDIDATE = 2048


class OptimizationConfigError(ValueError):
    """Raised for user-fixable configuration problems (exit code 3)."""


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    problem: str
    backend: str
    model_id: str
    base_url: str
    max_candidates: int
    max_repair_rounds: int
    gpu_budget_seconds: float
    token_budget: int
    output: Path
    resume: bool = False
    snapshot_root: Path | None = None
    gpu_device: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    state: str
    champion_sha256: str | None
    champion_path: Path | None
    report_path: Path


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    level: int
    problem_id: int


def parse_problem_spec(spec: str) -> ProblemSpec:
    match = re.fullmatch(r"kernelbench:l(\d+):(\d+)", spec.strip())
    if not match:
        raise OptimizationConfigError(f"problem must look like 'kernelbench:l1:40'; got {spec!r}")
    return ProblemSpec(level=int(match.group(1)), problem_id=int(match.group(2)))


def resolve_problem(spec: ProblemSpec, snapshot_root: Path):
    """Pin the problem file by id prefix inside the frozen snapshot."""
    level_dir = Path(snapshot_root) / "KernelBench" / f"level{spec.level}"
    if not level_dir.is_dir():
        raise OptimizationConfigError(
            f"snapshot level directory missing: {level_dir} "
            "(restore the KernelBench snapshot first)"
        )
    matches = sorted(
        path
        for path in level_dir.iterdir()
        if path.is_file() and re.match(rf"^{spec.problem_id}_", path.name) and path.suffix == ".py"
    )
    if len(matches) != 1:
        raise OptimizationConfigError(
            f"expected exactly one level{spec.level} problem matching id {spec.problem_id}; "
            f"found {[p.name for p in matches]}"
        )
    problem_path = matches[0]
    return problem_path, problem_path.read_text(encoding="utf-8")


def default_gpu_device() -> str:
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


def exit_code_for(state: str) -> int:
    codes = {
        "completed": EXIT_SUCCESS,
        "no_improvement": EXIT_NO_IMPROVEMENT,
        "budget_exhausted": EXIT_BUDGET_EXHAUSTED,
        "infra_error": EXIT_INFRA_ERROR,
    }
    if state not in codes:
        raise OptimizationConfigError(f"unknown terminal state {state!r}")
    return codes[state]


def parse_exit_code(result: OptimizationResult) -> int:
    return exit_code_for(result.state)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _feedback_note(stage: str, detail: str) -> str:
    bounded = detail[:CHAMPION_TRUNC]
    return (
        f"\n\nYour previous candidate failed at stage '{stage}': {bounded}\n"
        "Fix the problem and return a new complete candidate as before."
    )


class GenerationPort:
    """Calls the CandidateGenerator and reports the per-call token delta
    for budget settlement - cumulative ledger totals must never be
    settled twice across attempts."""

    def __init__(self, generator: CandidateGenerator):
        self._generator = generator
        self._billed_tokens = 0

    def __call__(self, problem_source: str, model_id: str, seed_note: str) -> dict:
        outcome, _request = self._generator.generate(problem_source, model_id, seed_note)
        total = self._generator.ledger.total_tokens()
        tokens = total - self._billed_tokens
        self._billed_tokens = total
        if isinstance(outcome, GenerationFailure):
            return {
                "ok": False,
                "stage": outcome.stage,
                "reason": outcome.reason,
                "candidate_source": None,
                "candidate_sha256": None,
                "tokens": tokens,
            }
        assert isinstance(outcome, GenerationSuccess)
        return {
            "ok": True,
            "stage": "generate",
            "reason": "",
            "candidate_source": outcome.candidate_source,
            "candidate_sha256": outcome.candidate_sha256,
            "tokens": tokens,
        }


def optimize(
    config: OptimizationConfig,
    *,
    generator: CandidateGenerator | None = None,
    evaluate=None,
    correctness_pro=None,
    timing=None,
    confirm=None,
) -> OptimizationResult:
    """Run the alpha optimization loop for one pinned problem.

    The stage ports default to the real adapters; tests inject offline
    fakes. Every candidate is an orchestrator action, so resume semantics
    and billing come from the durable journal, not from memory."""
    from kernelagent.config import build_stage_ports, default_generator

    output = Path(config.output)
    if config.resume and not Journal(output / "journal.jsonl").path.exists():
        raise OptimizationConfigError(
            f"no run to resume at {output} (journal.jsonl missing); start without --resume"
        )
    snapshot_root = config.snapshot_root or DEFAULT_SNAPSHOT_ROOT
    spec = parse_problem_spec(config.problem)
    problem_path, problem_source = resolve_problem(spec, snapshot_root)
    problem_sha256 = _sha256_text(problem_source)
    gpu_device = config.gpu_device or default_gpu_device()

    if generator is None:
        generator = default_generator(config)
    generate = GenerationPort(generator)
    if evaluate is None or timing is None or confirm is None:
        real = build_stage_ports(config, spec, problem_path, snapshot_root, gpu_device)
        evaluate = evaluate or real.evaluate
        timing = timing or real.timing
        confirm = confirm or real.confirm
        # correctness_pro stays whatever the caller passed: an explicit
        # None disables the pro stage (recorded as not_run), the real
        # wiring enables it.
        if correctness_pro is None and (evaluate is real.evaluate):
            correctness_pro = real.correctness_pro
    if confirm is None:

        def confirm(facts: CandidateFacts, incumbent: list[float], candidate: list[float]):
            return confirm_promotion(
                facts=facts,
                incumbent_batches_ms=incumbent,
                candidate_batches_ms=candidate,
            )

    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    champion_dir = output / "champion"
    journal_path = output / "journal.jsonl"

    budget = Budget(
        gpu_seconds_limit=config.gpu_budget_seconds,
        tokens_limit=config.token_budget,
    )
    orchestrator = Orchestrator(journal_path, budget)

    state = "completed"
    champion_sha: str | None = None
    champion_path: Path | None = None
    feedback = ""
    stop = False

    def _persist_record(name: str, record: dict) -> None:
        (records_dir / f"{name}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _load_record(name: str) -> dict | None:
        path = records_dir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _baseline_action():
        def run() -> dict:
            result = timing(problem_source, "eager")
            batches = list(result.get("batches") or ())
            ok, note = validate_batch_samples(batches, expected_count=len(batches))
            if not (result.get("source_ok") and ok):
                return {
                    "status": "infra_error",
                    "stage": "timing",
                    "reason": f"baseline timing invalid: {note or 'source rejected'}",
                    "gpu_seconds": _EST_GPU_SECONDS_PER_STAGE,
                    "tokens": 0,
                }
            record = {
                "candidate": "baseline-eager",
                "stage": "timing",
                "status": "measured",
                "batches_ms": batches,
                "async_leak": bool(result.get("async_leak")),
            }
            _persist_record("baseline-eager", record)
            return {
                "result_ref": "baseline-eager",
                "gpu_seconds": _EST_GPU_SECONDS_PER_STAGE,
                "tokens": 0,
            }

        return Action(
            name="baseline-eager",
            input_hash=f"baseline:{problem_sha256[:16]}",
            estimated_gpu_seconds=_EST_GPU_SECONDS_PER_STAGE,
            estimated_tokens=0,
            run=run,
        )

    def _attempt_action(index: int):
        name = f"candidate-{index:03d}"

        def run() -> dict:
            nonlocal feedback
            gen = generate(problem_source, config.model_id, feedback)
            if not gen["ok"]:
                record = {
                    "candidate": name,
                    "stage": "generation",
                    "status": "failed",
                    "detail": gen["reason"][:CHAMPION_TRUNC],
                }
                _persist_record(name, record)
                feedback = _feedback_note("generation", gen["reason"])
                return {"result_ref": f"{name}:generation-failed", "tokens": gen["tokens"]}

            source = gen["candidate_source"]
            policy = inspect_candidate_policy(source)
            if not policy.allowed:
                record = {
                    "candidate": name,
                    "stage": "policy",
                    "status": "rejected",
                    "detail": list(policy.violations)[:8],
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                feedback = _feedback_note(
                    "policy", "Policy violations: " + "; ".join(policy.violations)
                )
                return {"result_ref": f"{name}:policy-rejected", "tokens": gen["tokens"]}

            evaluation = evaluate(source, name)
            if evaluation.get("adapter_pass") is None:
                # The evaluator itself failed to produce a verdict: this is
                # infrastructure, not a candidate failure - record it and
                # keep the incumbent (a missing verdict is never a pass).
                record = {
                    "candidate": name,
                    "stage": "evaluate",
                    "status": "infra_error",
                    "detail": str(evaluation.get("stderr_tail", ""))[:CHAMPION_TRUNC],
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                return {"result_ref": f"{name}:eval-infra", "tokens": gen["tokens"]}
            if evaluation.get("adapter_pass") is not True:
                reason = (
                    f"upstream evaluator verdict: compiled={evaluation.get('compiled')} "
                    f"correct={evaluation.get('correct')} "
                    f"metadata={str(evaluation.get('upstream_metadata'))[:200]} "
                    f"stderr_tail={str(evaluation.get('stderr_tail'))[:400]}"
                )
                record = {
                    "candidate": name,
                    "stage": "evaluate",
                    "status": "failed",
                    "detail": reason[:CHAMPION_TRUNC],
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                feedback = _feedback_note("evaluate", reason)
                return {"result_ref": f"{name}:eval-failed", "tokens": gen["tokens"]}

            pro_note = "not_run"
            pro_ok = True
            if correctness_pro is not None:
                pro = correctness_pro(source, name)
                pro_ok, pro_note = pro
            if not pro_ok:
                record = {
                    "candidate": name,
                    "stage": "correctness_pro",
                    "status": "failed",
                    "detail": str(pro_note)[:CHAMPION_TRUNC],
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                feedback = _feedback_note("correctness_pro", str(pro_note))
                return {"result_ref": f"{name}:pro-failed", "tokens": gen["tokens"]}

            timing_result = timing(source, "candidate")
            batches = list(timing_result.get("batches") or ())
            ok, note = validate_batch_samples(batches, expected_count=len(batches))
            if not timing_result.get("source_ok") or not ok:
                reason = f"timing evidence invalid: {note or 'source rejected'}"
                record = {
                    "candidate": name,
                    "stage": "timing",
                    "status": "rejected",
                    "detail": reason,
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                feedback = _feedback_note("timing", reason)
                return {"result_ref": f"{name}:timing-invalid", "tokens": gen["tokens"]}
            if timing_result.get("async_leak"):
                record = {
                    "candidate": name,
                    "stage": "timing",
                    "status": "rejected",
                    "detail": "async leak integrity check failed",
                    "candidate_sha256": gen["candidate_sha256"],
                }
                _persist_record(name, record)
                feedback = _feedback_note(
                    "timing", "Async leak: work continued past the timed region."
                )
                return {"result_ref": f"{name}:async-leak", "tokens": gen["tokens"]}

            baseline_record = _load_record("baseline-eager") or {}
            incumbent_batches = list(baseline_record.get("batches_ms") or [])
            facts = CandidateFacts(
                compiled=bool(evaluation.get("compiled")),
                correct=bool(evaluation.get("correct")),
            )
            decision = confirm(facts, incumbent_batches, batches)
            if decision.outcome == TERMINAL_PROMOTE:
                champion_dir.mkdir(exist_ok=True)
                champion_file = champion_dir / f"{name}_{gen['candidate_sha256'][:12]}.py"
                champion_file.write_text(source, encoding="utf-8")
                (champion_dir / "champion.json").write_text(
                    json.dumps(
                        {
                            "candidate": name,
                            "candidate_sha256": gen["candidate_sha256"],
                            "path": str(champion_file),
                            "outcome": decision.outcome,
                            "reason": decision.reason,
                            "ratio_ci_95": list(decision.ratio_ci_95 or ()),
                            "delta": decision.delta,
                            "confidence": decision.confidence,
                            "problem_sha256": problem_sha256,
                            "backend": config.backend,
                            "gpu_device": gpu_device,
                            "protocol_sha256": None,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                record = {
                    "candidate": name,
                    "stage": "confirm",
                    "status": "promoted",
                    "detail": decision.reason,
                    "candidate_sha256": gen["candidate_sha256"],
                    "champion_path": str(champion_file),
                    "ratio_ci_95": list(decision.ratio_ci_95 or ()),
                }
                _persist_record(name, record)
                feedback = ""
                return {
                    "result_ref": f"{name}:promoted",
                    "tokens": gen["tokens"],
                    "gpu_seconds": _EST_GPU_SECONDS_PER_STAGE,
                }
            record = {
                "candidate": name,
                "stage": "confirm",
                "status": "retained",
                "detail": decision.reason,
                "candidate_sha256": gen["candidate_sha256"],
            }
            _persist_record(name, record)
            feedback = _feedback_note("confirm", decision.reason)
            return {"result_ref": f"{name}:retained", "tokens": gen["tokens"]}

        return Action(
            name=name,
            input_hash=f"{problem_sha256[:16]}:{config.backend}:{index}",
            estimated_gpu_seconds=_EST_GPU_SECONDS_PER_STAGE,
            estimated_tokens=_EST_TOKENS_PER_CANDIDATE,
            run=run,
        )

    baseline = orchestrator.run([_baseline_action()], resume=config.resume)
    if baseline["state"] == "budget_exhausted":
        state = "budget_exhausted"
        stop = True

    attempts = 0
    repairs = 0
    if not stop:
        for index in range(config.max_candidates):
            attempts += 1
            report = orchestrator.run([_attempt_action(index)], resume=True)
            if report["state"] == "budget_exhausted":
                state = "budget_exhausted"
                break
            record = _load_record(f"candidate-{index:03d}")
            if record is None:
                state = "infra_error"
                break
            if record.get("status") == "infra_error":
                state = "infra_error"
                break
            if record.get("status") == "promoted":
                champion_sha = record.get("candidate_sha256")
                champion_path = Path(record["champion_path"])
                state = "completed"
                break
            if record.get("status") == "failed" or record.get("status") == "rejected":
                repairs += 1
                if repairs > config.max_repair_rounds:
                    state = "no_improvement"
                    break
            if attempts >= config.max_candidates:
                state = "no_improvement"

    durable = reconstruct_budget(orchestrator.journal.entries)
    candidate_records = [
        _load_record(path.stem) for path in sorted(records_dir.glob("candidate-*.json"))
    ]
    report_payload = {
        "protocol": OPTIMIZATION_PROTOCOL,
        "config": {
            "problem": config.problem,
            "backend": config.backend,
            "model_id": config.model_id,
            "base_url": config.base_url,
            "max_candidates": config.max_candidates,
            "max_repair_rounds": config.max_repair_rounds,
            "output": str(output),
        },
        "problem_path": str(problem_path),
        "problem_sha256": problem_sha256,
        "gpu_device": gpu_device,
        "commit": _git_commit(),
        "state": state,
        "champion": {
            "candidate_sha256": champion_sha,
            "path": str(champion_path) if champion_path else None,
        },
        "candidates": [record for record in candidate_records if record],
        "budget": {
            "gpu_seconds_limit": config.gpu_budget_seconds,
            "tokens_limit": config.token_budget,
            "settled_gpu_seconds": durable.settled_gpu_seconds,
            "settled_tokens": durable.settled_tokens,
            "reserved_gpu_seconds": durable.reserved_gpu_seconds,
            "reserved_tokens": durable.reserved_tokens,
        },
        "journal_entries": len(orchestrator.journal.entries),
        "candidate_trust": "cooperative",
        "adversarially_secure": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return OptimizationResult(
        state=state,
        champion_sha256=champion_sha,
        champion_path=champion_path,
        report_path=report_path,
    )


def run_status(output: Path) -> dict:
    """Read a run's durable state for ``kernelagent status``."""
    output = Path(output)
    report_path = output / "report.json"
    journal_path = output / "journal.jsonl"
    if not journal_path.exists():
        raise OptimizationConfigError(f"no run found at {output}")
    journal = Journal(journal_path)
    durable = reconstruct_budget(journal.entries)
    summary = {
        "output": str(output),
        "journal_entries": len(journal.entries),
        "settled_gpu_seconds": durable.settled_gpu_seconds,
        "settled_tokens": durable.settled_tokens,
        "reserved_gpu_seconds": durable.reserved_gpu_seconds,
        "reserved_tokens": durable.reserved_tokens,
        "has_report": report_path.is_file(),
    }
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        summary["state"] = payload.get("state")
        summary["champion"] = payload.get("champion")
        summary["candidates"] = len(payload.get("candidates") or ())
    return summary
