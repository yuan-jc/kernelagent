"""Immutable candidate code packages and their §11.1 identity.

identity is derived from content, never stored: a stored hash could disagree
with the stored bytes, and that disagreement is exactly the corruption this
project must not be able to express silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from kernelagent.domain._validation import (
    require_instance,
    require_relative_path,
    require_sha256,
    require_string,
    require_text,
    require_tuple,
    require_unique,
)
from kernelagent.domain.errors import ContractError


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One UTF-8 text file of a code package; binary payloads are out of scope."""

    path: str
    content: str

    def __post_init__(self) -> None:
        require_relative_path(self.path, "path")
        require_text(self.content, "content")


@dataclass(frozen=True, slots=True)
class BuildSpec:
    """How the package is built. ``flags`` keep order (it is semantic); ``env``
    keys are unique and canonicalized (sorted) when hashed."""

    toolchain: str
    flags: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        require_text(self.toolchain, "toolchain")
        require_tuple(self.flags, "flags")
        for index, flag in enumerate(self.flags):
            require_text(flag, f"flags[{index}]")
        require_tuple(self.env, "env")
        for index, pair in enumerate(self.env):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ContractError(f"env[{index}] must be a (key, value) tuple; got {pair!r}")
            require_text(pair[0], f"env[{index}] key")
            require_text(pair[1], f"env[{index}] value")
        require_unique(tuple(key for key, _ in self.env), "env keys")


@dataclass(frozen=True, slots=True)
class Implementation:
    """A candidate code package. It makes no claim about its own correctness
    or speed; those are EvaluationResults produced by the trusted evaluator."""

    operator_id: str
    backend: str
    entry_point: str
    source_files: tuple[SourceFile, ...]
    build_spec: BuildSpec | None = None
    applicability_guard: str = ""
    parent_ids: tuple[str, ...] = ()
    origin: str = ""

    def __post_init__(self) -> None:
        require_text(self.operator_id, "operator_id")
        require_text(self.backend, "backend")
        require_text(self.entry_point, "entry_point")
        require_string(self.applicability_guard, "applicability_guard")
        require_string(self.origin, "origin")
        require_tuple(self.source_files, "source_files")
        if not self.source_files:
            raise ContractError("Implementation.source_files must not be empty")
        for index, source_file in enumerate(self.source_files):
            require_instance(source_file, SourceFile, f"source_files[{index}]")
        require_unique(tuple(f.path for f in self.source_files), "source file paths")
        if self.build_spec is not None:
            require_instance(self.build_spec, BuildSpec, "build_spec")
        require_tuple(self.parent_ids, "parent_ids")
        for parent in self.parent_ids:
            require_sha256(parent, "parent_ids entry")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def implementation_id(implementation: Implementation) -> str:
    """SHA-256 over canonical file list, file bytes, build spec, entry point and
    guard (design §11.1). Deliberately excludes operator_id, backend, parents
    and origin: they describe usage and provenance, not the built artifact."""
    files = sorted(
        (
            {
                "path": source.path,
                "sha256": hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
            }
            for source in implementation.source_files
        ),
        key=lambda item: item["path"],
    )
    build = None
    if implementation.build_spec is not None:
        build = {
            "toolchain": implementation.build_spec.toolchain,
            "flags": list(implementation.build_spec.flags),
            "env": sorted([key, value] for key, value in implementation.build_spec.env),
        }
    payload = {
        "entry_point": implementation.entry_point,
        "files": files,
        "build": build,
        "guard": implementation.applicability_guard,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()
