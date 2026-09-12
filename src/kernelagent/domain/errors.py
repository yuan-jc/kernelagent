"""Domain error taxonomy.

These exceptions are part of the contract: callers branch on them instead of
inspecting messages, and they never stand in for successful results.
"""

from __future__ import annotations


class DomainError(ValueError):
    """Base class for domain contract violations."""


class ContractError(DomainError):
    """A field value violates an invariant of a domain object."""


class SchemaVersionError(DomainError):
    """A serialized payload declares an unsupported schema version."""
