"""Operator semantics: what a callable computes, never how fast it runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kernelagent.domain._validation import (
    require_dims,
    require_instance,
    require_one_of,
    require_string,
    require_text,
    require_tuple,
    require_unique,
)
from kernelagent.domain.errors import ContractError

Granularity = Literal["operator", "subgraph", "model"]
GRANULARITIES: frozenset[str] = frozenset({"operator", "subgraph", "model"})


@dataclass(frozen=True, slots=True)
class IOPort:
    """One declared input/output. ``shape`` uses -1 for dynamic dims, () for scalars."""

    name: str
    dtype: str
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        require_text(self.dtype, "dtype")
        require_dims(self.shape, "shape", allow_dynamic=True)


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Mathematical and calling semantics of one optimization target.

    Speed, "current best implementation" and benchmark scores are not part of
    the spec; they live in results and promotion decisions.
    """

    operator_id: str
    granularity: Granularity
    inputs: tuple[IOPort, ...]
    outputs: tuple[IOPort, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        require_text(self.operator_id, "operator_id")
        require_one_of(self.granularity, "granularity", GRANULARITIES)
        require_string(self.notes, "notes")
        require_tuple(self.inputs, "inputs")
        require_tuple(self.outputs, "outputs")
        if not self.outputs:
            raise ContractError("OperatorSpec.outputs must not be empty")
        for index, port in enumerate(self.inputs):
            require_instance(port, IOPort, f"inputs[{index}]")
        for index, port in enumerate(self.outputs):
            require_instance(port, IOPort, f"outputs[{index}]")
        require_unique(tuple(p.name for p in self.inputs), "input names")
        require_unique(tuple(p.name for p in self.outputs), "output names")
