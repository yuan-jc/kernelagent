"""Cost accounting: token usage is a first-class budget object (design 12)."""

from __future__ import annotations

from dataclasses import dataclass

from kernelagent.adapters.models.client import ModelResponse
from kernelagent.adapters.models.errors import BudgetExceededError


@dataclass(frozen=True, slots=True)
class UsageTotals:
    prompt_tokens: int
    completion_tokens: int
    requests: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostLedger:
    """Accumulates usage per model id from completed responses."""

    def __init__(self) -> None:
        self._by_model: dict[str, list[int]] = {}
        self._requests = 0

    def record(self, response: ModelResponse) -> None:
        entry = self._by_model.setdefault(response.model_id, [0, 0, 0])
        entry[0] += response.usage.prompt_tokens
        entry[1] += response.usage.completion_tokens
        entry[2] += 1
        self._requests += 1

    def totals(self, model_id: str | None = None) -> dict[str, UsageTotals]:
        if model_id is not None:
            selected = {model_id: self._by_model[model_id]} if model_id in self._by_model else {}
        else:
            selected = self._by_model
        return {
            name: UsageTotals(
                prompt_tokens=values[0], completion_tokens=values[1], requests=values[2]
            )
            for name, values in selected.items()
        }

    def total_tokens(self) -> int:
        return sum(totals.total_tokens for totals in self.totals().values())

    def total_requests(self) -> int:
        return self._requests


class TokenBudget:
    """Hard ceiling on cumulative tokens: exceeding it is an explicit error,
    never a silent continue."""

    def __init__(self, max_total_tokens: int):
        if isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int):
            raise ValueError("max_total_tokens must be an integer")
        if max_total_tokens < 1:
            raise ValueError(f"max_total_tokens must be >= 1; got {max_total_tokens!r}")
        self.max_total_tokens = max_total_tokens

    def ensure_allowed(self, ledger: CostLedger, estimated_next_tokens: int = 0) -> None:
        if isinstance(estimated_next_tokens, bool) or not isinstance(estimated_next_tokens, int):
            raise ValueError("estimated_next_tokens must be an integer")
        if estimated_next_tokens < 0:
            raise ValueError("estimated_next_tokens must be >= 0")
        projected = ledger.total_tokens() + estimated_next_tokens
        if projected > self.max_total_tokens:
            raise BudgetExceededError(
                f"projected token usage {projected} exceeds budget "
                f"{self.max_total_tokens} (used so far {ledger.total_tokens()})"
            )
