"""Trusted worker boundary: process and container isolation for untrusted
payloads.

T04a scope: private work directories, scrubbed child environments, hard
timeouts that kill the whole process tree, structured outcomes, and
detection of trusted-directory tampering. This is process-boundary and
detection machinery, NOT a sandbox.

T04 scope (ADR-0001): a container executor with OS-enforced boundaries -
PID namespace, no network, read-only rootfs and inputs, size-capped tmpfs
output, cgroup memory/CPU/PID limits, non-root user - driven through the
same WorkerRequest/WorkerOutcome contract."""

from kernelagent.worker.container import (
    ContainerSpec,
    build_container_env,
    container_exists,
    execute_container,
    list_worker_containers,
    remove_container,
)
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
    is_secret_name,
    process_exists,
    snapshot_dir_hashes,
)

__all__ = [
    "ContainerSpec",
    "PathOutsideWorkspaceError",
    "TrustedDirModifiedError",
    "WorkerError",
    "WorkerOutcome",
    "WorkerRequest",
    "assert_dir_unchanged",
    "build_child_env",
    "build_container_env",
    "container_exists",
    "ensure_within",
    "execute",
    "execute_container",
    "is_secret_name",
    "list_worker_containers",
    "process_exists",
    "remove_container",
    "snapshot_dir_hashes",
]
