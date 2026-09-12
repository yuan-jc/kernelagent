"""What the agent is asked to optimize, on which inputs, under which track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kernelagent.domain._validation import (
    require_instance,
    require_one_of,
    require_text,
    require_tuple,
    require_unique,
)
from kernelagent.domain.errors import ContractError
from kernelagent.domain.operator import OperatorSpec
from kernelagent.domain.workload import Workload

Track = Literal["upstream_compatible", "robustness_extended", "production_reuse"]
TRACKS: frozenset[str] = frozenset(
    {"upstream_compatible", "robustness_extended", "production_reuse"}
)


@dataclass(frozen=True, slots=True)
class OptimizationTask:
    """Optimize ``workloads`` of one operator under one benchmark track.

    Budgets, workers and promotion rules are runtime concerns (T10+); a task
    only pins the target and the scoring track.
    """

    task_id: str
    operator: OperatorSpec
    workloads: tuple[Workload, ...]
    track: Track = "upstream_compatible"

    def __post_init__(self) -> None:
        require_text(self.task_id, "task_id")
        require_one_of(self.track, "track", TRACKS)
        require_instance(self.operator, OperatorSpec, "operator")
        require_tuple(self.workloads, "workloads")
        if not self.workloads:
            raise ContractError("OptimizationTask.workloads must not be empty")
        for index, workload in enumerate(self.workloads):
            require_instance(workload, Workload, f"workloads[{index}]")
        require_unique(tuple(w.workload_id for w in self.workloads), "workload ids")
        expected_inputs = len(self.operator.inputs)
        for workload in self.workloads:
            if workload.operator_id != self.operator.operator_id:
                raise ContractError(
                    f"workload {workload.workload_id!r} targets operator "
                    f"{workload.operator_id!r}, expected {self.operator.operator_id!r}"
                )
            if len(workload.shapes) != expected_inputs:
                raise ContractError(
                    f"workload {workload.workload_id!r} has {len(workload.shapes)} shapes, "
                    f"operator declares {expected_inputs} inputs"
                )
