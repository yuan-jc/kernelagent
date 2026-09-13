"""Input objects for the method generator (generator-design §1.4).

All inputs are plain frozen data - no torch, no triton, no model SDK.
Missing evidence is represented by ``None``/empty tuples (MISSING), never
by a zero or a fabricated default: the classifier and planner must be able
to tell "not measured" apart from "measured, unremarkable".

``AttemptRecord`` upgrades the loop's free-text feedback (REVIEW.md R5) into
structured history: which method was applied to which candidate, at which
pipeline stage it failed, and under what failure class. The optimization
loop persists ``method_id`` on each candidate record, so attempts survive a
resume and the feedback loop stays on the same search path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kernelagent.generators.features import CodeFeatures
from kernelagent.methods import TaskProfile

# Failure classes are the taxonomy the planner's success prior uses
# (generator-design §3.2, meta.failure_feedback_loop). ``none`` is only
# legal together with ``outcome="promoted"``.
FAILURE_CLASSES: frozenset[str] = frozenset(
    {"parse", "policy", "correctness", "timing", "no_gain", "infra", "none"}
)

# Pipeline stage (record spelling) -> failure class of a candidate rejected
# at that stage. Stages absent from the map have no method-attributable
# failure (e.g. an infra error is the environment's, not the method's).
STAGE_FAILURE_CLASS: dict[str, str] = {
    "generation": "parse",
    "policy": "policy",
    "evaluate": "correctness",
    "correctness_pro": "correctness",
    "timing": "timing",
    "confirm": "no_gain",
}


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One finished candidate attributed to the method plan item it used."""

    method_id: str | None
    stage: str
    failure_class: str
    note: str = ""
    outcome: str = "failed"  # failed | promoted

    def __post_init__(self) -> None:
        if self.method_id is not None and (
            not isinstance(self.method_id, str) or not self.method_id
        ):
            raise ValueError(
                f"method_id must be None or a non-empty string; got {self.method_id!r}"
            )
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError(f"stage must be a non-empty string; got {self.stage!r}")
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError(
                f"failure_class must be one of {sorted(FAILURE_CLASSES)}; "
                f"got {self.failure_class!r}"
            )
        if self.outcome not in ("failed", "promoted"):
            raise ValueError(f"outcome must be 'failed' or 'promoted'; got {self.outcome!r}")
        if (self.failure_class == "none") != (self.outcome == "promoted"):
            raise ValueError(
                "failure_class='none' and outcome='promoted' must be used together; got "
                f"failure_class={self.failure_class!r} outcome={self.outcome!r}"
            )
        if not isinstance(self.note, str):
            raise ValueError(f"note must be a string; got {self.note!r}")


@dataclass(frozen=True, slots=True)
class HardwareFacts:
    """Minimal hardware projection (design §5.2). ``None`` = MISSING: the
    cc>=9.0 guard must refuse on unknown, never assume support."""

    compute_capability: float | None = None
    sm_count: int | None = None
    shared_memory_per_sm_bytes: int | None = None
    dram_bandwidth_gbps: float | None = None

    @classmethod
    def sm89_reference(cls) -> "HardwareFacts":
        """Current hardware target (RTX 4060 Laptop 8GB). SM count and
        bandwidth stay MISSING until a local microbenchmark calibrates them
        (methods.yaml header: "待运行微基准核实")."""
        return cls(compute_capability=8.9)


@dataclass(frozen=True, slots=True)
class GeneratorInput:
    """Everything the planner may look at - and nothing else (§1.4)."""

    task: TaskProfile
    code_features: CodeFeatures
    attempts: tuple[AttemptRecord, ...] = ()
    ncu_view: Mapping[str, Any] | None = None  # evidence_view() output; None = not profiled
    hardware: HardwareFacts = field(default_factory=HardwareFacts)


def attempt_from_record(record: Mapping[str, Any]) -> AttemptRecord | None:
    """Rebuild an AttemptRecord from a persisted candidate record dict.

    Returns ``None`` for records that carry no method-attributable outcome
    (missing stage/status); infrastructure errors map to failure class
    ``infra`` and never count against a method's success prior."""
    stage = record.get("stage")
    status = record.get("status")
    if not isinstance(stage, str) or not stage or not isinstance(status, str) or not status:
        return None
    method_id = record.get("method_id")
    if not isinstance(method_id, str) or not method_id:
        method_id = None
    note = record.get("detail")
    note_text = note if isinstance(note, str) else ""
    if status == "promoted":
        return AttemptRecord(
            method_id=method_id,
            stage=stage,
            failure_class="none",
            outcome="promoted",
            note=note_text,
        )
    failure_class = "infra" if status == "infra_error" else STAGE_FAILURE_CLASS.get(stage)
    if failure_class is None:
        return None
    return AttemptRecord(
        method_id=method_id, stage=stage, failure_class=failure_class, note=note_text
    )
