"""Concrete inputs for one execution of an operator."""

from __future__ import annotations

from dataclasses import dataclass

from kernelagent.domain._validation import (
    require_dims,
    require_finite_number,
    require_non_negative_int,
    require_text,
    require_tuple,
)
from kernelagent.domain.errors import ContractError


@dataclass(frozen=True, slots=True)
class Workload:
    """A fully concrete input configuration: shapes/dtypes per positional input.

    Unlike an OperatorSpec port, workload dimensions are concrete positive
    integers; dynamic dimensions (-1) are resolved before a workload exists.
    """

    workload_id: str
    operator_id: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    seed: int | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        require_text(self.workload_id, "workload_id")
        require_text(self.operator_id, "operator_id")
        require_tuple(self.shapes, "shapes")
        require_tuple(self.dtypes, "dtypes")
        if len(self.shapes) != len(self.dtypes):
            raise ContractError(
                f"shapes and dtypes must have the same length; got {len(self.shapes)} and "
                f"{len(self.dtypes)}"
            )
        for index, shape in enumerate(self.shapes):
            require_dims(shape, f"shapes[{index}]", allow_dynamic=False)
        for index, dtype in enumerate(self.dtypes):
            require_text(dtype, f"dtypes[{index}]")
        if self.seed is not None:
            require_non_negative_int(self.seed, "seed")
        require_finite_number(self.weight, "weight", positive=True)
