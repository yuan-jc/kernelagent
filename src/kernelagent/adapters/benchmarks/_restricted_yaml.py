"""Restricted YAML subset parser shared by benchmark adapters.

The control plane has deliberately no YAML dependency (pyproject keeps a
zero-runtime-dependency policy), but benchmark metadata files (GPU-MODE
``task.yml``/set yamls, user-bench ``suite.yaml``/``problem.yaml``) use a
tiny YAML subset. This module parses exactly that subset and rejects
anything else rather than guessing:

- nested mappings by indentation, lists of scalars and of mappings;
- JSON-compatible flow values (``[...]`` / ``{...}``), optionally followed
  by a trailing ``# comment``;
- block scalars (``|`` / ``>``) kept as raw text;
- ``#`` comments outside block scalars;
- plain unquoted strings without YAML structure markers.

Anything with other YAML structure (anchors, multi-docs, exotic scalars)
is a hard error.
"""

from __future__ import annotations

import json
import re


class RestrictedYamlError(ValueError):
    """Input uses syntax outside the supported restricted YAML subset."""


def _parse_scalar(text: str) -> object:
    text = text.strip()
    if not text:
        return None
    if text and text[0] in "[{":
        # flow value possibly followed by a trailing comment
        depth, in_string, end = 0, None, None
        for pos, char in enumerate(text):
            if in_string:
                if char == in_string:
                    in_string = None
                continue
            if char in "\"'":
                in_string = char
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    break
        if end is None:
            raise RestrictedYamlError(f"unterminated flow value {text!r}")
        flow, remainder = text[:end], text[end:].strip()
        if remainder and not remainder.startswith("#"):
            raise RestrictedYamlError(f"unsupported content after flow value {text!r}")
        try:
            return json.loads(flow)
        except json.JSONDecodeError as exc:
            raise RestrictedYamlError(f"unsupported flow value {flow!r}: {exc}") from exc
    if (text[0] == '"' and text.endswith('"')) or (text[0] == "'" and text.endswith("'")):
        return text[1:-1]
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[+-]?([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?", text):
        return float(text)
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~"):
        return None
    if not any(marker in text for marker in (":", "{", "}", "[", "]", ",", "#", "&", "*", "!")):
        return text
    raise RestrictedYamlError(f"unsupported scalar {text!r}")


def _key_line(text: str):
    return re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", text)


def _child_indent(lines: list[str], index: int, parent_indent: int) -> int:
    """Indentation of the child block after a bare ``key:`` line (-1 = empty)."""
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = len(lines[index]) - len(lines[index].lstrip(" "))
        return indent if indent > parent_indent else -1
    return -1


def _consume_mapping(
    lines: list[str], index: int, mapping: dict, *, inner_indent: int, stop_at_dash: int | None
) -> int:
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        line_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if line_indent < inner_indent:
            break
        if stop_at_dash is not None and line_indent == stop_at_dash and stripped.startswith("- "):
            break
        if line_indent > inner_indent:
            raise RestrictedYamlError(f"unexpected indentation at {stripped!r}")
        match = _key_line(stripped)
        if not match:
            raise RestrictedYamlError(f"unsupported line {stripped!r}")
        key, rest = match.group(1), match.group(2)
        if rest in ("|", ">"):
            block: list[str] = []
            block_indent = None
            index += 1
            while index < len(lines):
                follow = lines[index]
                if not follow.strip():
                    block.append("")
                    index += 1
                    continue
                follow_indent = len(follow) - len(follow.lstrip(" "))
                if block_indent is None:
                    block_indent = follow_indent
                if follow_indent < block_indent:
                    break
                block.append(follow[block_indent:])
                index += 1
            while block and not block[-1]:
                block.pop()
            mapping[key] = "\n".join(block)
            continue
        if not rest:
            child_indent = _child_indent(lines, index + 1, inner_indent)
            if child_indent == -1:
                mapping[key] = None
                index += 1
                continue
            child, index = _parse_block(lines, index + 1, child_indent)
            mapping[key] = child
            continue
        mapping[key] = _parse_scalar(rest)
        index += 1
    return index


def _parse_block(lines: list[str], index: int, indent: int):
    """Parse the block starting at ``lines[index]`` whose entries sit at
    ``indent`` (a mapping or a list). Returns (value, next_index)."""
    while index < len(lines) and (not lines[index].strip() or lines[index].strip().startswith("#")):
        index += 1
    if index >= len(lines):
        return None, index
    if lines[index].lstrip().startswith("- "):
        items: list = []
        while index < len(lines):
            stripped = lines[index].strip()
            line_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if line_indent != indent or not stripped.startswith("- "):
                break
            content = stripped[2:].strip()
            key_match = _key_line(content)
            if key_match is not None and key_match.group(2) not in ("|", ">"):
                # list-of-mappings item: its own keys live at the dash indent
                item: dict = {}
                item[key_match.group(1)] = _parse_scalar(key_match.group(2))
                index += 1
                index = _consume_mapping(
                    lines, index, item, inner_indent=indent + 2, stop_at_dash=indent
                )
                items.append(item)
                continue
            items.append(_parse_scalar(content))
            index += 1
        return items, index
    mapping: dict = {}
    index = _consume_mapping(lines, index, mapping, inner_indent=indent, stop_at_dash=None)
    return mapping, index


def parse_restricted_yaml(text: str) -> dict:
    """Parse the restricted YAML subset into plain Python data. The top
    level must be a mapping."""
    lines = text.splitlines()
    value, index = _parse_block(lines, 0, 0)
    if index < len(lines):
        raise RestrictedYamlError(f"unconsumed content at line {index + 1}")
    if not isinstance(value, dict):
        raise RestrictedYamlError("input must parse to a top-level mapping")
    return value
