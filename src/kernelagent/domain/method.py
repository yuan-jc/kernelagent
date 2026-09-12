"""Methods and proposals: turning optimization ideas into falsifiable plans.

A proposal is a plan, never a result. Whether the plan produced a correct or
faster implementation is decided exclusively by EvaluationResult records the
candidate cannot write itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelagent.domain._validation import (
    require_finite_number,
    require_instance,
    require_sha256,
    require_text,
    require_tuple,
)
from kernelagent.domain.evidence import EvidenceRef


@dataclass(frozen=True, slots=True)
class Hypothesis:
    statement: str
    supporting_evidence: tuple[EvidenceRef, ...] = ()
    predicted_observations: tuple[str, ...] = ()
    falsification_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.statement, "statement")
        require_tuple(self.supporting_evidence, "supporting_evidence")
        for index, ref in enumerate(self.supporting_evidence):
            require_instance(ref, EvidenceRef, f"supporting_evidence[{index}]")
        require_tuple(self.predicted_observations, "predicted_observations")
        for index, observation in enumerate(self.predicted_observations):
            require_text(observation, f"predicted_observations[{index}]")
        require_tuple(self.falsification_conditions, "falsification_conditions")
        for index, condition in enumerate(self.falsification_conditions):
            require_text(condition, f"falsification_conditions[{index}]")


@dataclass(frozen=True, slots=True)
class MethodProposal:
    """One planned application of an OptimizationMethod (registered in T13+)."""

    method_id: str
    method_version: str
    hypothesis: Hypothesis
    parameter_space_sha256: str
    estimated_gpu_seconds: float
    target_workload_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.method_id, "method_id")
        require_text(self.method_version, "method_version")
        require_instance(self.hypothesis, Hypothesis, "hypothesis")
        require_sha256(self.parameter_space_sha256, "parameter_space_sha256")
        require_finite_number(
            self.estimated_gpu_seconds, "estimated_gpu_seconds", positive=True, allow_zero=True
        )
        require_tuple(self.target_workload_ids, "target_workload_ids")
        for index, workload_id in enumerate(self.target_workload_ids):
            require_text(workload_id, f"target_workload_ids[{index}]")
