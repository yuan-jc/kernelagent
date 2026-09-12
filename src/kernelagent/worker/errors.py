"""Worker error taxonomy."""

from __future__ import annotations


class WorkerError(ValueError):
    """Base class for worker boundary violations."""


class PathOutsideWorkspaceError(WorkerError):
    """A path escapes the workspace root it must stay inside."""


class TrustedDirModifiedError(WorkerError):
    """A trusted directory changed between snapshots (tamper detection)."""
