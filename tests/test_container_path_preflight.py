"""Daemon-free boundary tests for the container worker's preflight
validation (RV06, R7).

Refusing a mount source that resolves outside the workspace must be pure
path/config validation: no Docker daemon, no pinned image cache, no
network. The rejection must happen before any container resource
operation - including the docker binary lookup and the image inspect -
so a dangerous request is refused on a machine that has no Docker at
all. These tests therefore run unconditionally on every platform and in
every CI matrix; a failure here is a real regression, never an
environment gap. Real-Docker integration tests live in
``test_container_worker.py`` and skip with a recorded reason when Docker
or the pinned image is absent; a skip there is a coverage gap, never a
pass."""

from pathlib import Path

import pytest

from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    execute_container,
)

# A docker command that cannot exist: if the implementation ever moves the
# path validation back behind a Docker operation, these tests fail with the
# docker-binary error instead of the boundary message.
NO_DOCKER = ("ka-no-such-docker-binary",)


def make_request(tmp_path: Path, **overrides) -> WorkerRequest:
    values = {
        "request_id": "req-1",
        "argv": ("/bin/sh", "-c", "true"),
        "timeout_seconds": 60.0,
        "workspace_root": tmp_path / "workspace",
    }
    values.update(overrides)
    return WorkerRequest(**values)


def make_spec(mounts: tuple[tuple[str, Path], ...]) -> ContainerSpec:
    return ContainerSpec(
        image="docker.m.daocloud.io/library/python",
        image_digest="sha256:" + "5" * 64,
        read_only_mounts=mounts,
    )


def outcome_for(tmp_path: Path, mounts: tuple[tuple[str, Path], ...]):
    return execute_container(make_request(tmp_path), make_spec(mounts), docker_command=NO_DOCKER)


# -- rejections: every escape variant must be refused before any Docker
#    resource operation, with the boundary message ---------------------------


def test_absolute_path_outside_workspace_rejected(tmp_path):
    outcome = outcome_for(tmp_path, (("/task", tmp_path / "outside"),))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_parent_traversal_outside_workspace_rejected(tmp_path):
    escape = tmp_path / "workspace" / "inputs" / ".." / ".." / "escaped"
    outcome = outcome_for(tmp_path, (("/task", escape),))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_deep_traversal_outside_workspace_rejected(tmp_path):
    escape = tmp_path / "workspace" / "a" / "b" / "c" / ".." / ".." / ".." / ".." / "x"
    outcome = outcome_for(tmp_path, (("/task", escape),))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_symlinked_file_outside_workspace_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    link = workspace / "reference.txt"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except OSError:  # no symlink privilege (e.g. Windows runner) - fixture only
        pytest.skip("symlink creation unavailable on this platform")
    outcome = outcome_for(tmp_path, (("/task", link),))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_symlinked_directory_outside_workspace_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    link = workspace / "inputs"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(tmp_path)  # resolves to the tmp root, outside workspace
    except OSError:  # no symlink privilege (e.g. Windows runner) - fixture only
        pytest.skip("symlink creation unavailable on this platform")
    outcome = outcome_for(tmp_path, (("/task", link),))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


def test_mount_source_that_is_a_file_rejected_as_missing(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    plain_file = workspace / "not-a-dir.txt"
    plain_file.write_text("data", encoding="utf-8")
    outcome = outcome_for(tmp_path, (("/task", plain_file),))
    assert outcome.status == "infra_error"
    assert "does not exist" in outcome.stderr_tail, (
        "a non-directory mount source must be refused by the same preflight"
    )


def test_every_mount_is_validated_before_any_docker_operation(tmp_path):
    # The first mount is fine; the second escapes. Both must be checked
    # before the docker binary lookup happens.
    workspace = tmp_path / "workspace"
    valid = workspace / "inputs"
    valid.mkdir(parents=True)
    outcome = outcome_for(tmp_path, (("/data", valid), ("/task", Path("/etc"))))
    assert outcome.status == "infra_error"
    assert "resolves outside workspace" in outcome.stderr_tail


# -- positive control: legitimate mounts pass the preflight and the run
#    proceeds toward the (absent) docker binary -------------------------------


def test_valid_mount_passes_preflight(tmp_path):
    workspace = tmp_path / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    outcome = outcome_for(tmp_path, (("/task", inputs),))
    assert outcome.status == "infra_error"
    assert "not found" in outcome.stderr_tail, "run must proceed to the docker step"
    assert "resolves outside workspace" not in outcome.stderr_tail


def test_preflight_rejection_leaves_no_workdir(tmp_path):
    outcome = outcome_for(tmp_path, (("/task", Path("/etc")),))
    assert outcome.workdir is None, (
        "a request refused before any resource operation must not create a work directory"
    )
