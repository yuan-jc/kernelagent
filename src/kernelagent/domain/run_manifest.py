"""RunManifest: the versioned, validated identity of one optimization run.

The manifest is persisted before the first billable action of a run and is
the only thing a resumed process may trust about "what run am I continuing".
It pins:

- the problem identity (spec, pinned file name, full source, content hash);
- the benchmark source (benchmark name and snapshot it was resolved from);
- the backend and the evaluation/timing protocol identities (driver content
  hashes + the serialized timing protocol hash);
- the environment identity (pinned container image and the actual GPU device);
- the model provider configuration identity (model id, base URL and the
  *name* of the credential env var - never the credential itself, which is
  only ever read from the environment at call time);
- the generation strategy and budgets.

Identity policy (RV01 / review R1):

- *Identity fields* (``IDENTITY_FIELDS``) must match exactly for a resume to
  reuse any previous baseline/candidate result. A changed GPU device, image,
  protocol, problem, backend or model provider configuration is a REFUSAL:
  the caller raises a configuration error telling the user to start a new
  experiment with a fresh output directory. Old scores are never silently
  re-attributed to a new environment, and the manifest never "defaults to
  matching" - unknown manifest schema versions or missing fields are load
  errors, not implicit matches.
- *Revisable fields* (``REVISIBLE_FIELDS``: candidate allowance, repair
  rounds, budgets) are not identity. A resumed run may override them, but
  the caller must record every revision as an explicit journal event so the
  change is auditable.
- ``snapshot_root`` is recorded for convenience but is machine-local and not
  identity; the problem content hash carries that role.

Also defined here are the explicit pipeline stages a candidate moves through
and the terminal run states, so external tools (future timeline frontends)
can reconstruct each candidate's stage sequence from journal events alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kernelagent.domain._validation import (
    require_finite_number,
    require_non_negative_int,
    require_sha256,
    require_string,
    require_text,
)
from kernelagent.domain.errors import ContractError

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "run_manifest.json"


class PipelineStage(str, Enum):
    """Stages one candidate moves through, in frozen alpha order.

    Values are the journal/record spellings; the order of the members is the
    execution order of the loop."""

    GENERATE = "generate"
    POLICY = "policy"
    EVALUATE = "evaluate"
    CORRECTNESS_PRO = "correctness_pro"
    TIMING = "timing"
    CONFIRM = "confirm"


PIPELINE_STAGE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.GENERATE,
    PipelineStage.POLICY,
    PipelineStage.EVALUATE,
    PipelineStage.CORRECTNESS_PRO,
    PipelineStage.TIMING,
    PipelineStage.CONFIRM,
)


class RunState(str, Enum):
    """Durable run states; the terminal four are the honest outcomes."""

    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_IMPROVEMENT = "no_improvement"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRA_ERROR = "infra_error"


RUN_TERMINAL_STATES: frozenset[RunState] = frozenset(
    {
        RunState.COMPLETED,
        RunState.NO_IMPROVEMENT,
        RunState.BUDGET_EXHAUSTED,
        RunState.INFRA_ERROR,
    }
)

# Fields that must match exactly before a resume may reuse old results.
IDENTITY_FIELDS: tuple[str, ...] = (
    "run_protocol",
    "problem",
    "problem_sha256",
    "benchmark",
    "backend",
    "eval_driver_sha256",
    "timing_driver_sha256",
    "timing_protocol_sha256",
    "image_repo",
    "image_id",
    "gpu_device",
    "model_id",
    "base_url",
)

# Fields a resumed run may change, but only with an explicit revision event.
REVISIBLE_FIELDS: tuple[str, ...] = (
    "max_candidates",
    "max_repair_rounds",
    "gpu_budget_seconds",
    "token_budget",
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable identity card of one optimization run."""

    schema_version: str
    created_at: str
    run_protocol: str
    # Problem identity.
    problem: str
    problem_name: str
    problem_sha256: str
    problem_source: str
    # Benchmark source.
    benchmark: str
    snapshot: str
    snapshot_root: str
    # Backend and frozen evaluation/timing protocol identity.
    backend: str
    eval_driver_sha256: str
    timing_driver_sha256: str
    timing_protocol_sha256: str
    # Environment identity.
    image_repo: str
    image_id: str
    gpu_device: str
    # Model provider configuration identity (no secrets - the credential
    # env var is referenced by name only and read from the environment).
    model_id: str
    base_url: str
    api_key_env: str
    # Generation strategy and budgets (revisable on resume with events).
    max_candidates: int
    max_repair_rounds: int
    gpu_budget_seconds: float
    token_budget: int

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ContractError(
                f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}; got "
                f"{self.schema_version!r}; migrate the manifest explicitly"
            )
        require_text(self.created_at, "created_at")
        require_text(self.run_protocol, "run_protocol")
        require_text(self.problem, "problem")
        require_text(self.problem_name, "problem_name")
        require_sha256(self.problem_sha256, "problem_sha256")
        require_string(self.problem_source, "problem_source")
        require_text(self.benchmark, "benchmark")
        require_text(self.snapshot, "snapshot")
        require_string(self.snapshot_root, "snapshot_root")
        require_text(self.backend, "backend")
        require_sha256(self.eval_driver_sha256, "eval_driver_sha256")
        require_sha256(self.timing_driver_sha256, "timing_driver_sha256")
        require_sha256(self.timing_protocol_sha256, "timing_protocol_sha256")
        require_text(self.image_repo, "image_repo")
        require_text(self.image_id, "image_id")
        require_text(self.gpu_device, "gpu_device")
        require_text(self.model_id, "model_id")
        require_text(self.base_url, "base_url")
        require_text(self.api_key_env, "api_key_env")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int):
            raise ContractError(f"max_candidates must be an integer; got {self.max_candidates!r}")
        if self.max_candidates < 1:
            raise ContractError(f"max_candidates must be >= 1; got {self.max_candidates!r}")
        require_non_negative_int(self.max_repair_rounds, "max_repair_rounds")
        require_finite_number(self.gpu_budget_seconds, "gpu_budget_seconds", positive=True)
        if isinstance(self.token_budget, bool) or not isinstance(self.token_budget, int):
            raise ContractError(f"token_budget must be an integer; got {self.token_budget!r}")
        if self.token_budget < 1:
            raise ContractError(f"token_budget must be >= 1; got {self.token_budget!r}")

    def identity(self) -> dict[str, object]:
        """The identity subset as a plain dict (for reports and guards)."""
        return {field: getattr(self, field) for field in IDENTITY_FIELDS}

    def identity_mismatches(self, current: "RunManifest") -> dict[str, tuple[object, object]]:
        """Identity fields where ``current`` differs from this manifest.

        An empty result means the resume may reuse previous results; any
        entry means the caller must refuse (never re-attribute old scores
        to a new environment)."""
        mismatches: dict[str, tuple[object, object]] = {}
        for field in IDENTITY_FIELDS:
            stored = getattr(self, field)
            now = getattr(current, field)
            if stored != now:
                mismatches[field] = (stored, now)
        return mismatches

    def budget_revisions(self, current: "RunManifest") -> dict[str, tuple[object, object]]:
        """Revisible fields where ``current`` differs; each must be journaled."""
        revisions: dict[str, tuple[object, object]] = {}
        for field in REVISIBLE_FIELDS:
            stored = getattr(self, field)
            now = getattr(current, field)
            if stored != now:
                revisions[field] = (stored, now)
        return revisions
