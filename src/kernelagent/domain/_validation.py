"""Shared field validation helpers for frozen contract objects.

Each helper either passes silently or raises :class:`ContractError`; none of
them coerce values. Validation runs in ``__post_init__`` so constructing and
deserializing go through the same rules.
"""

from __future__ import annotations

import math
import re

from kernelagent.domain.errors import ContractError

_SHA256 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_DRIVE = re.compile("[A-Za-z]:")


def require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string; got {value!r}")


def require_string(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string; got {value!r}")


def require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase 64-hex sha256 string; got {value!r}")


def require_one_of(value: object, field: str, allowed: frozenset[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ContractError(f"{field} must be one of {sorted(allowed)}; got {value!r}")


def require_finite_number(
    value: object, field: str, *, positive: bool, allow_zero: bool = False
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a number; got {value!r}")
    if not math.isfinite(value):
        raise ContractError(f"{field} must be finite; got {value!r}")
    kind = "positive" if positive else "non-negative"
    if value < 0 or (value == 0 and not allow_zero):
        raise ContractError(f"{field} must be {kind}; got {value!r}")


def require_non_negative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer; got {value!r}")


def require_dims(shape: object, field: str, *, allow_dynamic: bool) -> None:
    """Validate one shape tuple; ``-1`` marks a dynamic dimension when allowed."""
    if not isinstance(shape, tuple):
        raise ContractError(f"{field} must be a tuple of ints; got {shape!r}")
    dynamic_note = " or -1 (dynamic)" if allow_dynamic else ""
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ContractError(f"{field} dimensions must be integers{dynamic_note}; got {dim!r}")
        if dim == -1 and allow_dynamic:
            continue
        if dim < 1:
            raise ContractError(f"{field} dimensions must be positive{dynamic_note}; got {dim!r}")


def require_relative_path(value: object, field: str) -> None:
    """Require a normalized relative POSIX path (no drive letters, '..', '.', '')."""
    require_text(value, field)
    assert isinstance(value, str)
    if "\\" in value:
        raise ContractError(f"{field} must use '/' separators; got {value!r}")
    if _WINDOWS_DRIVE.match(value):
        raise ContractError(f"{field} must not be an absolute Windows path; got {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ContractError(f"{field} must be a normalized relative path; got {value!r}")


def require_unique(names: object, field: str) -> None:
    if not isinstance(names, tuple):
        raise ContractError(f"{field} must be a tuple; got {names!r}")
    seen: set[str] = set()
    for item in names:
        if item in seen:
            raise ContractError(f"{field} contains duplicate entries: {item!r}")
        seen.add(item)


def require_tuple(value: object, field: str) -> None:
    """Reject mutable sequence lookalikes at frozen contract boundaries."""
    if not isinstance(value, tuple):
        raise ContractError(f"{field} must be a tuple; got {value!r}")


def require_instance(value: object, expected: type, field: str) -> None:
    if not isinstance(value, expected):
        raise ContractError(f"{field} must be a {expected.__name__}; got {type(value).__name__}")
