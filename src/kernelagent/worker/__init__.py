"""Trusted worker boundary: process isolation for untrusted payloads.

T04a scope: private work directories, scrubbed child environments, hard
timeouts that kill the whole process tree, structured outcomes, and
detection of trusted-directory tampering. This is process-boundary and
detection machinery, NOT a sandbox: OS-enforced write isolation (containers)
belongs to the parent T04 package on the target Ubuntu environment."""

from kernelagent.worker.errors import (
    PathOutsideWorkspaceError,
    TrustedDirModifiedError,
    WorkerError,
)
from kernelagent.worker.process import (
    WorkerOutcome,
    WorkerRequest,
    assert_dir_unchanged,
    build_child_env,
    ensure_within,
    execute,
    process_exists,
    snapshot_dir_hashes,
)

__all__ = [
    "PathOutsideWorkspaceError",
    "TrustedDirModifiedError",
    "WorkerError",
    "WorkerOutcome",
    "WorkerRequest",
    "assert_dir_unchanged",
    "build_child_env",
    "ensure_within",
    "execute",
    "process_exists",
    "snapshot_dir_hashes",
]
