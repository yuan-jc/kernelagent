"""Model request/response contracts and offline client implementations.

Identity: a request's content hash (``request_sha256``) is its replay and
idempotency key, computed over the canonical JSON of model, messages and
sampling parameters - never over wall-clock time or transport details."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from kernelagent.adapters.models.errors import RecordingMissError, TransientModelError

MODEL_CLIENT_PROTOCOL = "model-client-v1"
_ROLES = ("system", "user", "assistant")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        for field in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer; got {value!r}")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def plus(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A complete, replayable model request. ``purpose`` classifies the call
    for budget/ledger reporting; it does not affect the request hash."""

    model_id: str
    messages: tuple[tuple[str, str], ...]
    temperature: float = 0.0
    max_tokens: int | None = None
    purpose: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not self.messages:
            raise ValueError("messages must not be empty")
        for index, message in enumerate(self.messages):
            if (
                not isinstance(message, tuple)
                or len(message) != 2
                or message[0] not in _ROLES
                or not isinstance(message[1], str)
                or not message[1]
            ):
                raise ValueError(
                    f"messages[{index}] must be a (role, text) tuple with role in {_ROLES}"
                )
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(f"temperature must be within [0, 2]; got {self.temperature!r}")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
                raise ValueError("max_tokens must be an integer or None")
            if self.max_tokens < 1:
                raise ValueError(f"max_tokens must be >= 1; got {self.max_tokens!r}")

    @property
    def request_sha256(self) -> str:
        payload = {
            "model_id": self.model_id,
            "messages": [[role, text] for role, text in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelResponse:
    request_sha256: str
    model_id: str
    content: str
    finish_reason: str
    usage: ModelUsage

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("content must be a non-empty string")
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError("finish_reason must be a non-empty string")


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


class ScriptedModelClient:
    """Offline double: returns queued responses/exceptions in order.

    Order-based on purpose: lets tests exercise branch sequences (error,
    error, success) without constructing request hashes."""

    def __init__(self, script: Sequence[ModelResponse | Exception]):
        self._script = list(script)
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self._script:
            raise RecordingMissError("scripted client exhausted; add another response")
        self.calls += 1
        entry = self._script.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if entry.request_sha256 != request.request_sha256:
            raise ValueError("scripted response was built for a different request")
        return entry


class RecordedModelClient:
    """Offline replay: exact responses keyed by request content hash.

    A miss is an explicit error; recordings verify control flow and never
    stand in for real generation capability."""

    def __init__(self, responses: Mapping[str, ModelResponse]):
        self._responses = dict(responses)
        self.hits = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        key = request.request_sha256
        if key not in self._responses:
            raise RecordingMissError(
                f"no recorded response for request {key[:12]}… "
                f"({len(self._responses)} recording(s) available)"
            )
        self.hits += 1
        return self._responses[key]


class RetryingModelClient:
    """Bounded retry wrapper: only TransientModelError is retried, with
    exponential backoff through an injectable sleep (tests pass a recorder).
    PermanentModelError propagates immediately."""

    def __init__(
        self,
        inner: ModelClient,
        max_attempts: int = 3,
        sleep=time.sleep,
        backoff_seconds: float = 0.5,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._backoff = backoff_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: TransientModelError | None = None
        for attempt in range(self._max_attempts):
            if attempt:
                self._sleep(self._backoff * (2 ** (attempt - 1)))
            try:
                return self._inner.complete(request)
            except TransientModelError as exc:
                last_error = exc
        raise last_error if last_error else TransientModelError("retry loop did not run")
