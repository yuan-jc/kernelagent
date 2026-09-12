"""Structured output parsing: model text in, validated JSON object out.

Failures are explicit ParseErrors with the offending detail - prose, broken
JSON, missing keys or wrong types never become silent successes."""

from __future__ import annotations

import json
import re
from typing import Mapping

from kernelagent.adapters.models.errors import ParseError

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)

_TYPE_CHECKS: Mapping[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "object": dict,
}


def extract_json_payload(text: str) -> str:
    """Return the JSON payload from model output: a fenced ```json block when
    present, otherwise the text itself (callers validate by parsing)."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("model output is empty")
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_structured(text: str, required: Mapping[str, str] | None = None) -> dict:
    """Parse model output into a dict; enforce required keys and their simple
    types (``str``/``int``/``float``/``bool``/``list``/``object``)."""
    payload = extract_json_payload(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ParseError(f"expected a JSON object; got {type(parsed).__name__}")
    for key, expected in (required or {}).items():
        if expected not in _TYPE_CHECKS:
            raise ParseError(f"unknown expected type {expected!r} for key {key!r}")
        if key not in parsed:
            raise ParseError(f"missing required key {key!r}")
        value = parsed[key]
        if expected in ("int", "float") and isinstance(value, bool):
            raise ParseError(f"key {key!r} must be {expected}; got bool")
        if not isinstance(value, _TYPE_CHECKS[expected]):
            raise ParseError(f"key {key!r} must be {expected}; got {type(value).__name__}")
    return parsed
