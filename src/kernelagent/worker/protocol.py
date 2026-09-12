"""Worker request/outcome contracts (protocol worker-v1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

_WORKER_STATUSES = ("completed", "failed", "timeout")


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """One isolated execution of an untrusted payload.

    ``env_extra`` pairs are merged into the scrubbed child environment after
    deny-filtering; they cannot re-introduce secret-shaped names."""

    request_id: str
    argv: tuple[str, ...]
    timeout_seconds: float
    workspace_root: Path
    env_extra: tuple[tuple[str, str], ...] = ()
    keep_workdir_on_failure: bool = False
    output_tail_chars: int = 4000

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ValueError("argv must be a non-empty tuple")
        for index, arg in enumerate(self.argv):
            if not isinstance(arg, str) or not arg:
                raise ValueError(f"argv[{index}] must be a non-empty string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                f"timeout_seconds must be a finite positive number; got {self.timeout_seconds!r}"
            )
        if not isinstance(self.workspace_root, Path):
            raise ValueError("workspace_root must be a Path")
        for index, pair in enumerate(self.env_extra):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"env_extra[{index}] must be a (name, value) tuple")
            name, value = pair
            if not isinstance(name, str) or not name or "=" in name:
                raise ValueError(f"env_extra[{index}] name must be non-empty without '='")
            if not isinstance(value, str):
                raise ValueError(f"env_extra[{index}] value must be a string")
        if isinstance(self.output_tail_chars, bool) or not isinstance(self.output_tail_chars, int):
            raise ValueError("output_tail_chars must be an integer")
        if self.output_tail_chars < 1:
            raise ValueError(f"output_tail_chars must be >= 1; got {self.output_tail_chars!r}")


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Structured result of one isolated execution. ``workdir`` is set only
    when the private directory was retained (failure policy); otherwise the
    directory was removed and nothing outside it was written by the runner."""

    request_id: str
    status: str
    exit_code: int | None
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    killed: bool
    workdir: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _WORKER_STATUSES:
            raise ValueError(f"status must be one of {list(_WORKER_STATUSES)}; got {self.status!r}")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ValueError("exit_code must be an integer or None")
        for field in ("stdout_tail", "stderr_tail"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"{field} must be a string")
        if isinstance(self.killed, bool) is False:
            raise ValueError("killed must be a bool")
