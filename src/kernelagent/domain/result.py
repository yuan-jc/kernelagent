"""Execution and task outcomes.

``EvaluationResult`` is one execution of one candidate; ``TaskResult`` is the
terminal state of a whole search. "Search finished" and "found something
faster" are different facts and get different terminal states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kernelagent.domain._validation import (
    require_instance,
    require_one_of,
    require_sha256,
    require_text,
    require_tuple,
)
from kernelagent.domain.errors import ContractError
from kernelagent.domain.evidence import EvidenceRef
from kernelagent.domain.implementation import Implementation

EvaluationStatus = Literal[
    "passed",
    "incorrect",
    "build_failed",
    "timeout",
    "resource_exceeded",
    "unsupported",
    "inconclusive",
]
EVALUATION_STATUSES: frozenset[str] = frozenset(
    {
        "passed",
        "incorrect",
        "build_failed",
        "timeout",
        "resource_exceeded",
        "unsupported",
        "inconclusive",
    }
)

TaskTerminalState = Literal[
    "completed_improved",
    "completed_no_improvement",
    "failed",
    "unsupported",
    "baseline_failed",
    "cancelled",
    "budget_exhausted",
]
TASK_TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "completed_improved",
        "completed_no_improvement",
        "failed",
        "unsupported",
        "baseline_failed",
        "cancelled",
        "budget_exhausted",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    candidate_id: str
    protocol_sha256: str
    environment_sha256: str
    status: EvaluationStatus
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.candidate_id, "candidate_id")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        require_sha256(self.environment_sha256, "environment_sha256")
        require_one_of(self.status, "status", EVALUATION_STATUSES)
        require_tuple(self.evidence, "evidence")
        for index, ref in enumerate(self.evidence):
            require_instance(ref, EvidenceRef, f"evidence[{index}]")


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    terminal_state: TaskTerminalState
    champion: Implementation | None = None
    best_result: EvaluationResult | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.task_id, "task_id")
        require_one_of(self.terminal_state, "terminal_state", TASK_TERMINAL_STATES)
        if self.champion is not None:
            require_instance(self.champion, Implementation, "champion")
        if self.best_result is not None:
            require_instance(self.best_result, EvaluationResult, "best_result")
        require_tuple(self.evidence, "evidence")
        for index, ref in enumerate(self.evidence):
            require_instance(ref, EvidenceRef, f"evidence[{index}]")
        if self.terminal_state == "completed_improved" and self.champion is None:
            raise ContractError(
                "terminal_state 'completed_improved' requires a champion implementation"
            )
