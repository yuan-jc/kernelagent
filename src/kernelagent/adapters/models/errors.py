"""Model client error taxonomy.

TransientModelError subclasses are retry candidates; PermanentModelError
subclasses must surface immediately. Neither ever stands in for a result."""

from __future__ import annotations


class ModelClientError(ValueError):
    """Base class for model client contract violations."""


class TransientModelError(ModelClientError):
    """A retryable provider condition (rate limit, timeout, server error)."""


class PermanentModelError(ModelClientError):
    """A non-retryable provider condition (auth, malformed request, policy)."""


class RecordingMissError(ModelClientError):
    """No recorded response exists for this request (offline replay only)."""


class ParseError(ModelClientError):
    """Model output could not be parsed into the required structure."""


class BudgetExceededError(ModelClientError):
    """The proposed request would exceed the configured token budget."""
