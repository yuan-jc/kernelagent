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

Run identity (RV01 / review R1): every run persists a versioned, validated
RunManifest (see kernelagent.domain.run_manifest) before its first
billable action. Resume verifies the stored identity - GPU device, image,
evaluation/timing protocol, problem, backend, model provider - and refuses
to reuse previous results on any mismatch; revisible budget/allowance
fields may change only with explicit ``manifest_revised`` journal events.
Journal ``stage_started`` events plus the finished/terminal records let an
external reader reconstruct each candidate's pipeline stage sequence.

The stage ports are injectable so the loop is testable offline; the CLI
wires the real T05/T06/T07/T09 adapters over the container boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kernelagent.adapters.evals.timing import validate_batch_samples
from kernelagent.adapters.models.generation import (
    CandidateGenerator,
    GenerationFailure,
    GenerationSuccess,
    inspect_candidate_policy,
)
from kernelagent.adapters.profiling.pipeline import (
    PROFILE_DIR_NAME,
    build_baseline_profiler,
    load_collected_evidence,
)
from kernelagent.domain.errors import DomainError
from kernelagent.domain.run_manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    PipelineStage,
    RunManifest,
    RunState,
)
from kernelagent.domain.serialization import dumps as domain_dumps
from kernelagent.domain.serialization import loads as domain_loads
from kernelagent.generators.catalog import load_default as load_method_catalog
from kernelagent.generators.features import (
    extract_code_features,
    task_profile_from_problem,
)
from kernelagent.generators.input import (
    AttemptRecord,
    GeneratorInput,
    HardwareFacts,
    attempt_from_record,
)
from kernelagent.generators.plan import MethodPlan, MethodPlanner
from kernelagent.generators.prompt_fragments import compose_seed_note
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

# Journal action id of the baseline NCU profiling step (see
# _run_baseline_profile). Profiling is a separate, source-tagged evidence
# path (T15, ADR-0003): it never produces formal timing results and never
# decides promotion - it only feeds the method planner's classification.
PROFILE_ACTION_ID = "profile-baseline"

# Sentinel for optimize(planner=...): unset means "plan with the default
# frozen-catalog planner"; an explicit None disables planning entirely and
# reproduces the legacy behavior exactly (no method_plan journal events, no
# prompt injection).
_PLANNER_UNSET = object()

# Sentinel for optimize(profiler=...): unset means "use the real baseline
# NCU profiler when the run uses the real timing port and profiling is not
# disabled"; an explicit None disables profiling (offline/test runs with
# injected timing fakes never profile). An explicit callable is used as-is
# and must return the pipeline's outcome dict.
_PROFILER_UNSET = object()

EXIT_SUCCESS = 0
EXIT_NO_IMPROVEMENT = 1
EXIT_BUDGET_EXHAUSTED = 2
EXIT_CONFIG_ERROR = 3
EXIT_INFRA_ERROR = 4

_EST_GPU_SECONDS_PER_STAGE = 300.0
_EST_TOKENS_PER_CANDIDATE = 2048

# Name of the env var holding the provider credential. The credential value
# itself is only ever read from the environment at call time and never
# enters the manifest, journal, or reports.
API_KEY_ENV = "MODEL_PROVIDER_API_KEY"

# Single-writer lock file inside a run directory (RV02).
RUN_LOCK_FILENAME = "run.lock"

try:  # POSIX file locking; the msvcrt fallback keeps Windows dev boxes working.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


class OptimizationConfigError(ValueError):
    """Raised for user-fixable configuration problems (exit code 3)."""


class RunLockHeldError(OptimizationConfigError):
    """Another process holds the run directory's single-writer lock, so a
    concurrent resume of the same run was refused (RV02). Recovery is
    user-fixable: wait for the other process, or take over after it died
    (the OS releases the lock when its process exits)."""


@dataclass
class RunLock:
    """An OS-level exclusive lock over one run directory.

    The lock lives in ``<output>/run.lock`` and is held with
    ``fcntl.flock`` (POSIX) or ``msvcrt.locking`` (Windows), so the kernel
    releases it when the owning process dies - a crashed run never leaves
    a stale lock that blocks recovery forever."""

    path: Path
    _fd: int | None = None

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def acquire_run_lock(output: Path) -> RunLock:
    """Acquire the run directory's single-writer lock (RV02).

    Exactly one process may append to a run's journal at a time: journal
    appends are crash-consistent for one writer, not for two processes
    interleaving writes. A second concurrent resume fails with
    :class:`RunLockHeldError` before touching any durable state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / RUN_LOCK_FILENAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:  # pragma: no cover - Windows
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - no locking primitive
            raise OSError("no file-locking primitive available on this platform")
    except OSError as exc:
        os.close(fd)
        raise RunLockHeldError(
            f"run at {output} is already owned by another process "
            f"(single-writer lock {lock_path} is held); concurrent resume of one "
            "run is not allowed - wait for the other process to finish"
        ) from exc
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
    except OSError:  # pragma: no cover - diagnostics only, never fatal
        pass
    return RunLock(path=lock_path, _fd=fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` (RV02): the payload is
    fully written and fsynced to a temp file in the same directory, then
    moved into place with ``os.replace``, which is atomic within one
    filesystem. Readers and a crash can therefore never observe a
    half-written result record, champion, manifest, or report; the
    previous content stays complete until the instant of replacement."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


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
    # Baseline NCU profiling (default on): after the baseline timing, the
    # reference implementation is profiled once under the ADR-0003
    # diagnostic lease so the method planner classifies from measured
    # evidence instead of static priors. Any NCU/GPU failure degrades to an
    # explicit ``profile_not_run`` journal event; the loop never breaks.
    # Profiling is NOT part of the run identity: enabling/disabling it does
    # not change the frozen evaluation/timing protocol.
    profile_baseline: bool = True

    def __post_init__(self) -> None:
        # Review R6 / RV04: resources and paid calls must never be created
        # from an invalid configuration. Validation lives on the config
        # object itself so both the CLI and direct callers fail before any
        # model or GPU work is scheduled.
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates < 1
        ):
            raise OptimizationConfigError(
                f"max_candidates must be an integer >= 1; got {self.max_candidates!r}"
            )
        if (
            isinstance(self.max_repair_rounds, bool)
            or not isinstance(self.max_repair_rounds, int)
            or self.max_repair_rounds < 0
        ):
            raise OptimizationConfigError(
                f"max_repair_rounds must be an integer >= 0; got {self.max_repair_rounds!r}"
            )
        if (
            isinstance(self.gpu_budget_seconds, bool)
            or not isinstance(self.gpu_budget_seconds, (int, float))
            or not math.isfinite(self.gpu_budget_seconds)
            or self.gpu_budget_seconds <= 0
        ):
            raise OptimizationConfigError(
                f"gpu_budget_seconds must be a finite number > 0; got {self.gpu_budget_seconds!r}"
            )
        if (
            isinstance(self.token_budget, bool)
            or not isinstance(self.token_budget, int)
            or self.token_budget <= 0
        ):
            raise OptimizationConfigError(
                f"token_budget must be an integer > 0; got {self.token_budget!r}"
            )


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


def exit_code_for(state: str | RunState) -> int:
    name = state.value if isinstance(state, RunState) else str(state)
    codes = {
        RunState.COMPLETED.value: EXIT_SUCCESS,
        RunState.NO_IMPROVEMENT.value: EXIT_NO_IMPROVEMENT,
        RunState.BUDGET_EXHAUSTED.value: EXIT_BUDGET_EXHAUSTED,
        RunState.INFRA_ERROR.value: EXIT_INFRA_ERROR,
    }
    if name not in codes:
        raise OptimizationConfigError(f"unknown terminal state {name!r}")
    return codes[name]


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol_identity() -> dict[str, str]:
    """Frozen evaluation-protocol identity of the real stack.

    Read lazily through the adapter modules so tests can monkeypatch the
    pinned image/protocol attributes; the values describe what WOULD run,
    which is exactly what the manifest must pin. Module attributes are read
    at call time (not import time) for that reason."""
    from kernelagent.adapters.evals import kernelbench_eval
    from kernelagent.adapters.evals import timing as timing_adapter

    protocol = timing_adapter.TimingProtocol()
    return {
        "timing_protocol_sha256": protocol.identity_sha256(),
        "eval_driver_sha256": _sha256_file(kernelbench_eval._DRIVER_PATH),
        "timing_driver_sha256": _sha256_file(timing_adapter._TIMING_DRIVER_PATH),
        "image_repo": kernelbench_eval.EVAL_IMAGE_REPO,
        "image_id": kernelbench_eval.EVAL_IMAGE_ID,
    }


def build_run_manifest(
    config: OptimizationConfig,
    *,
    problem_path: Path,
    problem_source: str,
    problem_sha256: str,
    snapshot_root: Path,
    gpu_device: str,
    created_at: str | None = None,
) -> RunManifest:
    """Assemble the validated manifest for the run ``config`` describes.

    Pure assembly: no secrets are read (the credential env var is recorded
    by name only) and no model/GPU work is performed."""
    protocol = _protocol_identity()
    snapshot_root = Path(snapshot_root)
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_protocol=OPTIMIZATION_PROTOCOL,
        problem=config.problem,
        problem_name=problem_path.name,
        problem_sha256=problem_sha256,
        problem_source=problem_source,
        benchmark="kernelbench",
        snapshot=snapshot_root.name,
        snapshot_root=str(snapshot_root),
        backend=config.backend,
        eval_driver_sha256=protocol["eval_driver_sha256"],
        timing_driver_sha256=protocol["timing_driver_sha256"],
        timing_protocol_sha256=protocol["timing_protocol_sha256"],
        image_repo=protocol["image_repo"],
        image_id=protocol["image_id"],
        gpu_device=gpu_device,
        model_id=config.model_id,
        base_url=config.base_url,
        api_key_env=API_KEY_ENV,
        max_candidates=config.max_candidates,
        max_repair_rounds=config.max_repair_rounds,
        gpu_budget_seconds=float(config.gpu_budget_seconds),
        token_budget=config.token_budget,
    )


def load_run_manifest(path: Path) -> RunManifest:
    """Load and validate the persisted run manifest for a resume.

    Missing manifests and manifests written by other schema versions are
    hard configuration errors - a resume never defaults to "identity
    matches" when the identity cannot be verified (RV01)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise OptimizationConfigError(
            f"run manifest missing at {path}: runs without a persisted manifest "
            "cannot be resumed (start a new experiment)"
        ) from exc
    try:
        manifest = domain_loads(text)
    except DomainError as exc:
        raise OptimizationConfigError(
            f"run manifest at {path} is not a valid RunManifest: {exc}"
        ) from exc
    if not isinstance(manifest, RunManifest):
        raise OptimizationConfigError(
            f"run manifest at {path} holds {type(manifest).__name__}, not RunManifest"
        )
    return manifest


def validate_terminal(state: str | RunState, champion_sha256: str | None) -> None:
    """Enforce the legal-terminal invariant: ``completed`` must carry a
    verifiable champion (review R6). A champion-less completion is an
    illegal ledger state and is refused instead of reported as success."""
    name = state.value if isinstance(state, RunState) else str(state)
    if name == RunState.COMPLETED.value and not champion_sha256:
        raise OptimizationConfigError(
            "internal invariant violated: terminal state 'completed' requires a "
            "champion with a content hash; refusing to report an empty completion"
        )


def _feedback_note(stage: str, detail: str) -> str:
    bounded = detail[:CHAMPION_TRUNC]
    return (
        f"\n\nYour previous candidate failed at stage '{stage}': {bounded}\n"
        "Fix the problem and return a new complete candidate as before."
    )


def _run_baseline_profile(
    *,
    profiler: object,
    journal: Journal,
    budget: Budget | None,
    records_dir: Path,
    profile_dir: Path,
    problem_sha256: str,
) -> dict:
    """Baseline NCU profiling between baseline timing and the generation
    loop. Returns an outcome dict: ``status`` is ``collected`` (with the
    ``evidence_view`` payload for the method planner) or ``not_run`` with
    an explicit ``reason``.

    Honesty rules (B3 wiring):

    - every skip/failure is a durable ``profile_not_run`` journal event with
      an explicit reason; profiling must never break the main loop;
    - an already-collected evidence file is reused verbatim on resume
      (idempotent, no re-run, no duplicate billing);
    - measured GPU wall time is billed to the same durable budget via a
      ``budget_settled`` event under a fresh attempt id (profiling counts
      into gpu_budget; it is never free or hidden) and reported in the
      record's ``profile`` field;
    - the evidence stays ``source="ncu_profile"``: it never enters formal
      timing and never decides promotion.
    """
    record_path = records_dir / "baseline-eager.json"
    record: dict = {}
    if record_path.is_file():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
            record = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            record = {}
    if record.get("status") != "measured":
        journal.append(
            "profile_not_run",
            action_id=PROFILE_ACTION_ID,
            reason="baseline_not_measured",
            detail="baseline timing record missing or not measured; there is no "
            "baseline to profile",
            ts=time.time(),
        )
        return {"status": "not_run", "reason": "baseline_not_measured", "evidence_view": None}
    existing = record.get("profile")
    if isinstance(existing, dict) and existing.get("status") == "collected":
        view = load_collected_evidence(profile_dir)
        if view is not None:
            # Resume: evidence already collected; reuse verbatim, no re-run.
            return {
                "status": "collected",
                "reused": True,
                "reason": None,
                "evidence_view": view,
                "report_path": Path(str(existing.get("report") or "")).name,
                "evidence_path": Path(str(existing.get("evidence") or "")).name,
                "summary": existing.get("summary"),
            }
    if budget is not None and budget.exhausted:
        journal.append(
            "profile_not_run",
            action_id=PROFILE_ACTION_ID,
            reason="budget_exhausted",
            detail="gpu budget already exhausted; profiling skipped",
            ts=time.time(),
        )
        return {"status": "not_run", "reason": "budget_exhausted", "evidence_view": None}
    journal.append(
        "profile_started",
        action_id=PROFILE_ACTION_ID,
        problem_sha256=problem_sha256,
        ts=time.time(),
    )
    try:
        outcome = profiler()
    except Exception as exc:  # noqa: BLE001 - profiling must never break the loop
        outcome = {
            "status": "not_run",
            "reason": "profiler_error",
            "detail": f"{type(exc).__name__}: {exc}",
            "capture": {},
        }
    capture = outcome.get("capture") or {}
    if outcome.get("status") == "collected":
        wall = float(outcome.get("gpu_wall_seconds") or 0.0)
        attempt_id = uuid.uuid4().hex
        journal.append(
            "budget_settled",
            action_id=PROFILE_ACTION_ID,
            attempt_id=attempt_id,
            gpu_seconds=wall,
            tokens=0,
        )
        if budget is not None:
            budget.apply_settlement(PROFILE_ACTION_ID, attempt_id, wall, 0)
        journal.append(
            "profile_collected",
            action_id=PROFILE_ACTION_ID,
            problem_sha256=problem_sha256,
            capture=capture,
            driver_sha256=outcome.get("driver_sha256"),
            report=f"{PROFILE_DIR_NAME}/{outcome.get('report_path')}",
            report_sha256=outcome.get("report_sha256"),
            evidence=f"{PROFILE_DIR_NAME}/{outcome.get('evidence_path')}",
            summary=outcome.get("summary"),
            gpu_wall_seconds=wall,
            billed_gpu_seconds=wall,
            source=outcome.get("source"),
            ts=time.time(),
        )
        record["profile"] = {
            "status": "collected",
            "source": outcome.get("source"),
            "problem_sha256": problem_sha256,
            "capture": capture,
            "driver_sha256": outcome.get("driver_sha256"),
            "report": f"{PROFILE_DIR_NAME}/{outcome.get('report_path')}",
            "report_sha256": outcome.get("report_sha256"),
            "evidence": f"{PROFILE_DIR_NAME}/{outcome.get('evidence_path')}",
            "summary": outcome.get("summary"),
            "gpu_wall_seconds": wall,
            "billed_against_gpu_budget": True,
        }
        _atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True))
        return outcome
    journal.append(
        "profile_not_run",
        action_id=PROFILE_ACTION_ID,
        reason=outcome.get("reason"),
        detail=str(outcome.get("detail"))[:CHAMPION_TRUNC],
        capture=capture,
        ts=time.time(),
    )
    record["profile"] = {
        "status": "not_run",
        "reason": outcome.get("reason"),
        "detail": str(outcome.get("detail"))[:CHAMPION_TRUNC],
        "capture": capture,
    }
    _atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True))
    return outcome


def run_baseline_profile_for_run(
    output: Path,
    *,
    profiler: object,
    problem_sha256: str,
) -> dict:
    """`kernelagent profile` entry: collect (or reuse) the baseline NCU
    evidence of an existing run directory and write it back (evidence file,
    record ``profile`` field, journal events). Independent of the webapp;
    the caller must hold the run's single-writer lock.

    Uses the same idempotence and honesty rules as the in-loop path; the
    measured GPU wall time is journaled as a ``budget_settled`` event for
    the run's ledger even though this out-of-band call has no live Budget
    object in memory."""
    output = Path(output)
    return _run_baseline_profile(
        profiler=profiler,
        journal=Journal(output / "journal.jsonl"),
        budget=None,
        records_dir=output / "records",
        profile_dir=output / PROFILE_DIR_NAME,
        problem_sha256=problem_sha256,
    )


class GenerationPort:
    """Calls the CandidateGenerator and reports the per-call token delta
    for budget settlement - cumulative ledger totals must never be
    settled twice across attempts. The request content hash travels with
    every outcome (RV03): candidate records persist it as the durable
    request-identity evidence, so a resumed run's next request can be
    verified against the uninterrupted path from disk alone."""

    def __init__(self, generator: CandidateGenerator):
        self._generator = generator
        self._billed_tokens = 0

    def __call__(self, problem_source: str, model_id: str, seed_note: str) -> dict:
        outcome, _request = self._generator.generate(problem_source, model_id, seed_note)
        request_sha256 = outcome.request_sha256
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
                "request_sha256": request_sha256,
                "tokens": tokens,
            }
        assert isinstance(outcome, GenerationSuccess)
        return {
            "ok": True,
            "stage": "generate",
            "reason": "",
            "candidate_source": outcome.candidate_source,
            "candidate_sha256": outcome.candidate_sha256,
            "request_sha256": request_sha256,
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
    planner: MethodPlanner | None = _PLANNER_UNSET,
    profiler: object = _PROFILER_UNSET,
) -> OptimizationResult:
    """Run the alpha optimization loop for one pinned problem.

    The stage ports default to the real adapters; tests inject offline
    fakes. Every candidate is an orchestrator action, so resume semantics
    and billing come from the durable journal, not from memory.

    Method planning (generator-design §4): before each candidate's
    generate call the loop asks the planner for a MethodPlan from the
    problem's static features, task profile, structured attempt history
    and - when available - the baseline NCU evidence view; the top
    actionable item's single-factor prompt fragment is appended to
    the seed note and a ``method_plan`` journal event records the
    classification and selected method. Planning never decides outcomes -
    correctness, timing and promotion stay with the trusted ports. Pass
    ``planner=None`` to disable planning (legacy behavior,
    byte-identical requests); the default planner plans from the frozen
    method catalog even with no profile evidence (baseline plan, honest
    ``uncertain`` classification).

    Baseline profiling (B3): when ``config.profile_baseline`` is on (the
    default) and the run uses the real timing port, the reference
    implementation is profiled once with NCU under the ADR-0003 diagnostic
    lease after the baseline timing and before the first candidate. The
    resulting evidence view is passed to the planner, upgrading the
    classification coverage from ``static_only`` to ``ncu_full`` /
    ``ncu_partial``. Missing ncu, counter-permission blockers, timeouts
    and every other collection failure degrade to an explicit
    ``profile_not_run`` journal event with the loop continuing. Tests
    injecting a timing fake never profile; pass ``profiler=None`` to
    disable explicitly, or a callable to inject a fake profiler (it must
    return the pipeline's outcome dict).

    Run identity (RV01): a fresh run persists a validated RunManifest to
    ``<output>/run_manifest.json`` before any billable action. Resume loads
    it and refuses to reuse previous results when the GPU device, container
    image, evaluation/timing protocol, problem, backend, or model provider
    configuration changed - start a new experiment with a fresh --output in
    that case. Revisible fields (candidate allowance, repair rounds,
    budgets) may be overridden on resume; each override is recorded as an
    explicit ``manifest_revised`` journal event. A non-resume run into an
    existing non-empty output directory is a configuration error raised
    before any model or GPU call.

    Concurrency (RV02): the whole run executes under a single-writer lock
    on the run directory (``run.lock``, an OS-level lock the kernel
    releases when the owning process dies). A second concurrent resume of
    the same run fails with :class:`RunLockHeldError` before touching any
    durable state, instead of interleaving journal appends."""
    if planner is _PLANNER_UNSET:
        planner = MethodPlanner(load_method_catalog())

    output = Path(config.output)
    journal_path = output / "journal.jsonl"
    manifest_path = output / MANIFEST_FILENAME

    stored_manifest: RunManifest | None = None
    if config.resume:
        if not journal_path.exists():
            raise OptimizationConfigError(
                f"no run to resume at {output} (journal.jsonl missing); start without --resume"
            )
        stored_manifest = load_run_manifest(manifest_path)
    elif output.exists():
        # The web console writes its launcher bookkeeping (job.json, no key)
        # into the run directory before starting the optimize thread; that is
        # not prior experiment state. Any other pre-existing content means a
        # previous experiment may live here: refuse rather than mix.
        bookkeeping = {"job.json"}
        unexpected = {p.name for p in output.iterdir()} - bookkeeping
        if unexpected:
            raise OptimizationConfigError(
                f"output directory {output} already exists and is not empty; refusing to mix "
                "experiments - use the resume command to continue this run or pick a new --output"
            )

    # RV02 single-writer lock: at most one process may work on a run
    # directory. The lock is taken after the pure configuration checks and
    # before any durable write (journal, manifest, records); a failed
    # resume therefore never leaves the directory created mid-boot.
    run_lock = acquire_run_lock(output)
    try:
        return _optimize_under_lock(
            config,
            output=output,
            journal_path=journal_path,
            manifest_path=manifest_path,
            stored_manifest=stored_manifest,
            generator=generator,
            evaluate=evaluate,
            correctness_pro=correctness_pro,
            timing=timing,
            confirm=confirm,
            planner=planner,
            profiler=profiler,
        )
    finally:
        run_lock.release()


def _optimize_under_lock(
    config: OptimizationConfig,
    *,
    output: Path,
    journal_path: Path,
    manifest_path: Path,
    stored_manifest: RunManifest | None,
    generator: CandidateGenerator | None,
    evaluate,
    correctness_pro,
    timing,
    confirm,
    planner: MethodPlanner | None,
    profiler: object = _PROFILER_UNSET,
) -> OptimizationResult:
    """Run the optimization loop body while holding the run directory's
    single-writer lock. Called only from :func:`optimize`, which has
    already validated the configuration, restored the stored manifest on
    resume, and acquired the lock - every durable write below (journal,
    manifest, records, champion, report) happens under that lock."""
    from kernelagent.config import build_stage_ports, default_generator

    # Profiling default: only real runs (real timing port, feature not
    # disabled) profile; offline tests with injected timing fakes never do.
    timing_was_none = timing is None

    snapshot_root = config.snapshot_root
    if snapshot_root is None and stored_manifest is not None:
        snapshot_root = Path(stored_manifest.snapshot_root)
    snapshot_root = snapshot_root or DEFAULT_SNAPSHOT_ROOT
    spec = parse_problem_spec(config.problem)
    problem_path, problem_source = resolve_problem(spec, snapshot_root)
    problem_sha256 = _sha256_text(problem_source)
    gpu_device = config.gpu_device
    if gpu_device is None:
        gpu_device = (
            stored_manifest.gpu_device if stored_manifest is not None else default_gpu_device()
        )

    run_manifest = build_run_manifest(
        config,
        problem_path=problem_path,
        problem_source=problem_source,
        problem_sha256=problem_sha256,
        snapshot_root=snapshot_root,
        gpu_device=gpu_device,
    )
    if stored_manifest is not None:
        # Identity guard (RV01): the environment, problem, protocol and
        # model identity this run would use must equal the stored manifest
        # before any previous result may be reused.
        mismatches = stored_manifest.identity_mismatches(run_manifest)
        if mismatches:
            detail = "; ".join(
                f"{field}: manifest={stored!r} vs requested={requested!r}"
                for field, (stored, requested) in sorted(mismatches.items())
            )
            raise OptimizationConfigError(
                "resume identity mismatch - previous results must not be reused in a "
                f"changed environment ({detail}); start a new experiment with a fresh --output"
            )

    if generator is None:
        generator = default_generator(config.base_url)
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

    if profiler is _PROFILER_UNSET:
        profiler = None
        if config.profile_baseline and timing_was_none:
            # Real run (real timing port): build the real baseline NCU
            # profiler over the pinned problem + staged workspace. Any
            # environment failure inside it degrades to profile_not_run.
            profiler = build_baseline_profiler(
                profile_dir=output / PROFILE_DIR_NAME,
                workspace_root=output / "workspace",
                snapshot_root=snapshot_root,
                level=spec.level,
                problem_name=problem_path.name,
                problem_source=problem_source,
                gpu_device=gpu_device,
                problem_sha256=problem_sha256,
            )

    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    champion_dir = output / "champion"

    # Method-planning inputs (generator-design §1): static code features and
    # the name-inferred task profile come from the pinned problem; structured
    # attempt history is rebuilt from persisted candidate records so a
    # resumed run plans on the same evidence path as an uninterrupted one.
    # The last persisted failure also restores the seed-note feedback text
    # (REVIEW.md R5), keeping the resumed request content on the same path
    # as an uninterrupted run.
    task_profile = task_profile_from_problem(problem_path.name, spec.level, spec.problem_id)
    code_features = extract_code_features(problem_source)
    attempt_history: list[AttemptRecord] = []
    last_failure: tuple[str, str] | None = None
    for record_path in sorted(records_dir.glob("candidate-*.json")):
        if record_path.name.endswith(".progress.json"):
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        restored = attempt_from_record(record) if isinstance(record, dict) else None
        if restored is not None:
            attempt_history.append(restored)
            detail = record.get("detail")
            if (
                restored.outcome == "failed"
                and restored.failure_class != "infra"
                and isinstance(detail, str)
                and detail
            ):
                last_failure = (restored.stage, detail[:CHAMPION_TRUNC])

    budget = Budget(
        gpu_seconds_limit=config.gpu_budget_seconds,
        tokens_limit=config.token_budget,
    )
    orchestrator = Orchestrator(journal_path, budget)
    if stored_manifest is not None:
        # Allowed changes (budgets / candidate allowance) get explicit,
        # auditable journal events; the stored manifest stays the record of
        # what the run was originally created with.
        for field, (stored, requested) in sorted(
            stored_manifest.budget_revisions(run_manifest).items()
        ):
            orchestrator.journal.append(
                "manifest_revised",
                field=field,
                previous_value=stored,
                new_value=requested,
                reason="explicit resume override of a revisable manifest field",
            )
    else:
        # Persisted (atomically) before the first billable action of the run.
        _atomic_write_text(manifest_path, domain_dumps(run_manifest))

    state = "completed"
    champion_sha: str | None = None
    champion_path: Path | None = None
    feedback = ""
    if last_failure is not None:
        feedback = _feedback_note(*last_failure)
    stop = False

    def _persist_record(name: str, record: dict) -> None:
        _atomic_write_text(
            records_dir / f"{name}.json", json.dumps(record, indent=2, sort_keys=True)
        )

    def _mark_stage(name: str, stage: str) -> None:
        """Record the stage a candidate entered, twice: a lightweight
        heartbeat file (atomically replaced, RV02) for live status readers,
        and an append-only ``stage_started`` journal event so an external
        timeline can be reconstructed from journal.jsonl alone. The durable
        record written at the end of the attempt stays the source of truth
        for outcomes."""
        _atomic_write_text(
            records_dir / f"{name}.progress.json",
            json.dumps({"candidate": name, "stage": stage, "ts": time.time()}, sort_keys=True),
        )
        orchestrator.journal.append("stage_started", action_id=name, stage=stage, ts=time.time())

    def _load_record(name: str) -> dict | None:
        path = records_dir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _baseline_action():
        def run() -> dict:
            _mark_stage("baseline-eager", PipelineStage.TIMING.value)
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
            # Method planning (generator-design §4): replan from the current
            # attempt history before every generation so demotion and the
            # single-factor rotation happen without hidden state. The plan
            # proposes only; outcomes below stay with the trusted ports.
            selected_method_id: str | None = None
            if planner is not None:
                plan: MethodPlan = planner.plan(
                    GeneratorInput(
                        task=task_profile,
                        code_features=code_features,
                        attempts=tuple(attempt_history),
                        # Baseline NCU evidence when profiling collected it
                        # (None = not profiled: coverage stays static_only).
                        ncu_view=ncu_view,
                        hardware=HardwareFacts.sm89_reference(),
                    )
                )
                item = plan.first_actionable(tuple(attempt_history))
                selected_method_id = item.method_id if item is not None else None
                orchestrator.journal.append(
                    "method_plan",
                    action_id=name,
                    selected_method=selected_method_id,
                    ts=time.time(),
                    **plan.journal_fields(),
                )
                seed_note = compose_seed_note(
                    feedback, item.prompt_fragment if item is not None else None
                )
            else:
                seed_note = feedback
            _mark_stage(name, PipelineStage.GENERATE.value)
            gen = generate(problem_source, config.model_id, seed_note)
            if not gen["ok"]:
                record = {
                    "candidate": name,
                    "stage": "generation",
                    "status": "failed",
                    "detail": gen["reason"][:CHAMPION_TRUNC],
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="generation",
                        failure_class="parse",
                        note=gen["reason"][:CHAMPION_TRUNC],
                    )
                )
                feedback = _feedback_note("generation", gen["reason"])
                return {"result_ref": f"{name}:generation-failed", "tokens": gen["tokens"]}

            _mark_stage(name, PipelineStage.POLICY.value)
            source = gen["candidate_source"]
            policy = inspect_candidate_policy(source)
            if not policy.allowed:
                # Bounded one-line note (RV03 two-path parity): the resume
                # path rebuilds the next request's feedback text and the
                # planner's attempt note from the persisted record's
                # ``detail`` string, so it must be exactly the string the
                # uninterrupted run feeds back. The structured violations
                # stay alongside for evidence.
                policy_note = ("Policy violations: " + "; ".join(policy.violations))[
                    :CHAMPION_TRUNC
                ]
                record = {
                    "candidate": name,
                    "stage": "policy",
                    "status": "rejected",
                    "detail": policy_note,
                    "violations": list(policy.violations)[:8],
                    "candidate_sha256": gen["candidate_sha256"],
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="policy",
                        failure_class="policy",
                        note=policy_note,
                    )
                )
                feedback = _feedback_note("policy", policy_note)
                return {"result_ref": f"{name}:policy-rejected", "tokens": gen["tokens"]}

            _mark_stage(name, PipelineStage.EVALUATE.value)
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
                    "request_sha256": gen["request_sha256"],
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
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="evaluate",
                        failure_class="correctness",
                        note=reason[:CHAMPION_TRUNC],
                    )
                )
                feedback = _feedback_note("evaluate", reason)
                return {"result_ref": f"{name}:eval-failed", "tokens": gen["tokens"]}

            _mark_stage(name, PipelineStage.CORRECTNESS_PRO.value)
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
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="correctness_pro",
                        failure_class="correctness",
                        note=str(pro_note)[:CHAMPION_TRUNC],
                    )
                )
                feedback = _feedback_note("correctness_pro", str(pro_note))
                return {"result_ref": f"{name}:pro-failed", "tokens": gen["tokens"]}

            _mark_stage(name, PipelineStage.TIMING.value)
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
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="timing",
                        failure_class="timing",
                        note=reason,
                    )
                )
                feedback = _feedback_note("timing", reason)
                return {"result_ref": f"{name}:timing-invalid", "tokens": gen["tokens"]}
            if timing_result.get("async_leak"):
                record = {
                    "candidate": name,
                    "stage": "timing",
                    "status": "rejected",
                    "detail": "async leak integrity check failed",
                    "candidate_sha256": gen["candidate_sha256"],
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="timing",
                        failure_class="timing",
                        note="Async leak: work continued past the timed region.",
                    )
                )
                feedback = _feedback_note(
                    "timing", "Async leak: work continued past the timed region."
                )
                return {"result_ref": f"{name}:async-leak", "tokens": gen["tokens"]}

            _mark_stage(name, PipelineStage.CONFIRM.value)
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
                _atomic_write_text(champion_file, source)
                _atomic_write_text(
                    champion_dir / "champion.json",
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
                            "protocol_sha256": run_manifest.timing_protocol_sha256,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                )
                record = {
                    "candidate": name,
                    "stage": "confirm",
                    "status": "promoted",
                    "detail": decision.reason,
                    "candidate_sha256": gen["candidate_sha256"],
                    "champion_path": str(champion_file),
                    "ratio_ci_95": list(decision.ratio_ci_95 or ()),
                    "candidate_batches_ms": batches,
                    "method_id": selected_method_id,
                    "request_sha256": gen["request_sha256"],
                }
                _persist_record(name, record)
                attempt_history.append(
                    AttemptRecord(
                        method_id=selected_method_id,
                        stage="confirm",
                        failure_class="none",
                        outcome="promoted",
                        note=decision.reason,
                    )
                )
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
                "candidate_batches_ms": batches,
                "method_id": selected_method_id,
                "request_sha256": gen["request_sha256"],
            }
            _persist_record(name, record)
            attempt_history.append(
                AttemptRecord(
                    method_id=selected_method_id,
                    stage="confirm",
                    failure_class="no_gain",
                    note=decision.reason,
                )
            )
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
    # Baseline NCU evidence (B3): profile the reference once, between the
    # baseline timing and the first candidate, so the planner classifies
    # from measured Tier-A signals instead of static priors only. Any
    # failure degrades to a durable profile_not_run event - never a break.
    ncu_view = None
    if baseline["state"] == "budget_exhausted":
        state = "budget_exhausted"
        stop = True
    elif profiler is not None:
        profile_outcome = _run_baseline_profile(
            profiler=profiler,
            journal=orchestrator.journal,
            budget=orchestrator.budget,
            records_dir=records_dir,
            profile_dir=output / PROFILE_DIR_NAME,
            problem_sha256=problem_sha256,
        )
        ncu_view = profile_outcome.get("evidence_view")

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
        _load_record(path.stem)
        for path in sorted(records_dir.glob("candidate-*.json"))
        if not path.name.endswith(".progress.json")
    ]
    # A champion-less "completed" is an illegal terminal state (review R6):
    # never write such a report.
    validate_terminal(state, champion_sha)
    profile_section: dict | None = None
    if profiler is not None:
        baseline_record = _load_record("baseline-eager") or {}
        profile_field = baseline_record.get("profile")
        if not isinstance(profile_field, dict):
            profile_field = {"status": "not_run", "reason": "not_attempted"}
        profile_section = {
            "status": profile_field.get("status"),
            "reason": profile_field.get("reason"),
            "report": profile_field.get("report"),
            "evidence": profile_field.get("evidence"),
            "summary": profile_field.get("summary"),
            "gpu_wall_seconds": profile_field.get("gpu_wall_seconds"),
            "billed_against_gpu_budget": bool(profile_field.get("billed_against_gpu_budget")),
            "note": (
                "profiling is source=ncu_profile evidence for the method planner; it "
                "never enters formal timing and never decides promotion. Its measured "
                "GPU wall time is billed to the same durable gpu budget via the "
                "journal's budget_settled event for action 'profile-baseline'."
            ),
        }
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
        "run_manifest": {
            "path": str(manifest_path),
            "schema_version": run_manifest.schema_version,
            "identity": run_manifest.identity(),
        },
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
        "profile": profile_section,
        "journal_entries": len(orchestrator.journal.entries),
        "candidate_trust": "cooperative",
        "adversarially_secure": False,
    }
    report_path = output / "report.json"
    _atomic_write_text(report_path, json.dumps(report_payload, indent=2, sort_keys=True))
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
        "has_run_manifest": (output / MANIFEST_FILENAME).is_file(),
    }
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        summary["state"] = payload.get("state")
        summary["champion"] = payload.get("champion")
        summary["candidates"] = len(payload.get("candidates") or ())
    return summary
