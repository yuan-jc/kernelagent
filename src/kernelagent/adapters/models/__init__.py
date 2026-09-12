"""Model client adapters: the offline contract layer for LLM access.

T12a scope: request/response contracts, scripted and recorded offline
clients, bounded retry, cost accounting and structured output parsing.
Real provider HTTP clients belong to T12 proper (credentials + LIVE_MODEL);
nothing here imports a provider SDK or touches the network."""

from kernelagent.adapters.models.client import (
    MODEL_CLIENT_PROTOCOL,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RecordedModelClient,
    RetryingModelClient,
    ScriptedModelClient,
)
from kernelagent.adapters.models.costing import CostLedger, TokenBudget
from kernelagent.adapters.models.errors import (
    BudgetExceededError,
    ModelClientError,
    ParseError,
    PermanentModelError,
    RecordingMissError,
    TransientModelError,
)
from kernelagent.adapters.models.parsing import extract_json_payload, parse_structured

__all__ = [
    "MODEL_CLIENT_PROTOCOL",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "RecordedModelClient",
    "RetryingModelClient",
    "ScriptedModelClient",
    "CostLedger",
    "TokenBudget",
    "BudgetExceededError",
    "ModelClientError",
    "ParseError",
    "PermanentModelError",
    "RecordingMissError",
    "TransientModelError",
    "extract_json_payload",
    "parse_structured",
]
