"""Container-backed trusted worker (parent package T04; ADR-0001).

Runs the untrusted payload inside a Docker container with OS-enforced
boundaries the process executor (``kernelagent.worker.process``) cannot
provide: a PID namespace whose descendants cannot escape reclamation
(closing the documented setsid gap), ``--network none``, a read-only
rootfs, read-only bind-mounted trusted inputs, a tmpfs /tmp, and cgroup
limits for memory/CPU/PIDs. The container user is a non-root uid;
capabilities are dropped; privilege escalation is disabled.

Output channel: the payload writes results into ``/out``, a read-write
bind mount of a parent-owned directory inside the private work directory
(container uid 20000, directory mode 0777). A tmpfs output directory was
rejected after real observation: a container's tmpfs dies with its mount
namespace, so ``docker cp`` of a stopped container silently yields an
empty tree (exit 0) - results would be unrecoverable after exit. The
output size cap is therefore parent-enforced by a 0.5 s monitoring loop
that kills the container on breach, exactly like the stdout/stderr log
cap; the kernel-enforced memory/CPU/PID limits bound the damage between
polls.

Lifecycle contract: the container gets an independent random name (never
the request id), runs until the request budget ends it - killed on
timeout or cancel - and is then force-removed; removal is confirmed via
inspect and a failed reclamation is reported, never claimed clean. The
payload's exit code, stdout, or files it writes do not constitute
success: the parent verifies results itself from the output directory.

Every infrastructure step (image presence, start, kill, removal) is
checked and a failure produces a structured ``infra_error`` outcome -
never a silent ``completed``. Images must already exist locally under the
exact digest; the executor refuses to auto-pull. A side-car
``container-record.json`` in the private work directory captures the
inspect state, image, mounts, and reclamation result for evidence.

The private work directory is always retained and reported in
``WorkerOutcome.workdir`` (it holds the logs and the output directory the
parent must consume); unlike the process executor there is no
failure-only retention policy. Tests gate on Docker availability and skip
with a recorded reason; skips never count as passes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from kernelagent.worker.errors import PathOutsideWorkspaceError
from kernelagent.worker.process import (
    _read_tail,
    ensure_within,
    is_secret_name,
)
from kernelagent.worker.protocol import WorkerOutcome, WorkerRequest


def _raise_infra(message: str) -> None:
    raise RuntimeError(message)


CONTAINER_LABEL = "kernelagent.worker=1"
DEFAULT_USER = "20000:20000"
DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_CPUS = 1.0
DEFAULT_PIDS_LIMIT = 64
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
DEFAULT_TMP_TMPFS_BYTES = 64 * 1024 * 1024
DEFAULT_LOG_LIMIT_BYTES = 32 * 1024 * 1024
OUTPUT_MOUNT_POINT = "/out"
TMP_MOUNT_POINT = "/tmp"
TASK_WORKDIR = "/task"
_MONITOR_INTERVAL_SECONDS = 0.5
_LOG_TAIL_BYTES = 4000
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# Injected on top of the image environment so naive payloads work inside a
# read-only rootfs; every entry can be overridden through ``env_extra``.
_BASE_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": TMP_MOUNT_POINT,
    "TMPDIR": TMP_MOUNT_POINT,
}


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


class ContainerSpec:
    """Immutable description of the isolation boundary for one request.

    The image is pinned by ``image_digest`` (``repo @ sha256:...`` manifest
    digest) or - for locally built images that have no registry digest - by
    ``image_id`` (the content-addressed config digest). Exactly one must be
    given; the executor runs exactly that reference and refuses anything
    looser. Trusted input mount sources must resolve inside the request's
    workspace root; ``/out`` (parent-monitored size cap) and ``/tmp``
    (tmpfs) are reserved mount points the spec cannot override."""

    __slots__ = (
        "image",
        "image_digest",
        "image_id",
        "user",
        "memory_bytes",
        "cpus",
        "pids_limit",
        "output_limit_bytes",
        "tmp_tmpfs_bytes",
        "log_limit_bytes",
        "read_only_mounts",
        "gpu_devices",
    )

    def __init__(
        self,
        image: str,
        image_digest: str | None = None,
        *,
        image_id: str | None = None,
        user: str = DEFAULT_USER,
        memory_bytes: int = DEFAULT_MEMORY_BYTES,
        cpus: float = DEFAULT_CPUS,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
        tmp_tmpfs_bytes: int = DEFAULT_TMP_TMPFS_BYTES,
        log_limit_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
        read_only_mounts: tuple[tuple[str, Path], ...] = (),
        gpu_devices: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(image, str) or not image or not re.fullmatch(r"[A-Za-z0-9._/-]+", image):
            raise ValueError(f"image must be a plain repository path; got {image!r}")
        if image_digest is None and image_id is None:
            raise ValueError("either image_digest or image_id must pin the image")
        if image_digest is not None and image_id is not None:
            raise ValueError("image_digest and image_id are mutually exclusive")
        if image_digest is not None and not _DIGEST_PATTERN.fullmatch(image_digest):
            raise ValueError(f"image_digest must be 'sha256:<64 hex>'; got {image_digest!r}")
        if image_id is not None and not _DIGEST_PATTERN.fullmatch(image_id):
            raise ValueError(f"image_id must be 'sha256:<64 hex>'; got {image_id!r}")
        if not isinstance(user, str) or not re.fullmatch(r"[0-9]{1,5}:[0-9]{1,5}", user):
            raise ValueError(f"user must be '<uid>:<gid>'; got {user!r}")
        _require_positive_int(memory_bytes, "memory_bytes")
        if isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or cpus <= 0:
            raise ValueError(f"cpus must be a positive number; got {cpus!r}")
        _require_positive_int(pids_limit, "pids_limit")
        _require_positive_int(output_limit_bytes, "output_limit_bytes")
        _require_positive_int(tmp_tmpfs_bytes, "tmp_tmpfs_bytes")
        _require_positive_int(log_limit_bytes, "log_limit_bytes")
        if not isinstance(read_only_mounts, tuple):
            raise ValueError("read_only_mounts must be a tuple of (container_path, host_path)")
        seen_destinations: set[str] = {OUTPUT_MOUNT_POINT, TMP_MOUNT_POINT}
        for entry in read_only_mounts:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("read_only_mounts entries must be (container_path, host_path)")
            destination, source = entry
            if not isinstance(destination, str) or not destination.startswith("/"):
                raise ValueError(f"mount destination must be absolute; got {destination!r}")
            if destination in seen_destinations:
                raise ValueError(f"mount destination {destination!r} is reserved or duplicated")
            if not isinstance(source, Path):
                raise ValueError(f"mount source for {destination!r} must be a Path")
            seen_destinations.add(destination)
        if not isinstance(gpu_devices, tuple) or not all(
            isinstance(device, str) and device and "=" in device for device in gpu_devices
        ):
            raise ValueError("gpu_devices must be a tuple of CDI device names ('vendor/gpu=id')")
        self.image = image
        self.image_digest = image_digest
        self.image_id = image_id
        self.user = user
        self.memory_bytes = memory_bytes
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.output_limit_bytes = output_limit_bytes
        self.tmp_tmpfs_bytes = tmp_tmpfs_bytes
        self.log_limit_bytes = log_limit_bytes
        self.read_only_mounts = tuple(read_only_mounts)
        self.gpu_devices = tuple(gpu_devices)

    @property
    def reference(self) -> str:
        """The exact content-addressed image reference this spec runs."""
        return self.image_id if self.image_id is not None else f"{self.image}@{self.image_digest}"


def _docker(
    docker_command: tuple[str, ...], *args: str, timeout: float = 60.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*docker_command, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def container_exists(docker_command: tuple[str, ...], name: str) -> bool:
    completed = _docker(docker_command, "inspect", "--type", "container", name)
    return completed.returncode == 0


def list_worker_containers(docker_command: tuple[str, ...] = ("docker",)) -> list[str]:
    """Names of all kernelagent worker containers, including stopped ones;
    the leak-detection primitive for tests and reclamation audits."""
    completed = _docker(
        docker_command,
        "ps",
        "-a",
        "--filter",
        f"label={CONTAINER_LABEL}",
        "--format",
        "{{.Names}}",
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line]


def remove_container(docker_command: tuple[str, ...], name: str) -> bool:
    completed = _docker(docker_command, "rm", "-f", name, timeout=30.0)
    return completed.returncode == 0


def build_container_env(env_extra: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Base environment plus validated extras; extras cannot re-introduce
    secret-shaped names. The parent's own environment is never inherited:
    the container only receives these explicit pairs."""
    env: dict[str, str] = dict(_BASE_ENV)
    for name, value in env_extra:
        if is_secret_name(name):
            raise ValueError(f"env_extra name {name!r} matches the secret denylist")
        env[name] = value
    return tuple(env.items())


def _run_args(
    request: WorkerRequest,
    spec: ContainerSpec,
    name: str,
    workspace_root: Path,
    output_host: Path,
) -> list[str]:
    args = [
        "run",
        "--name",
        name,
        "--label",
        CONTAINER_LABEL,
        "--user",
        spec.user,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--network",
        "none",
        "--memory",
        str(spec.memory_bytes),
        "--cpus",
        str(spec.cpus),
        "--pids-limit",
        str(spec.pids_limit),
        "--mount",
        f"type=tmpfs,dst={TMP_MOUNT_POINT},tmpfs-size={spec.tmp_tmpfs_bytes},tmpfs-mode=01777",
        "--mount",
        f"type=bind,src={output_host},dst={OUTPUT_MOUNT_POINT}",
        "--workdir",
        TASK_WORKDIR,
    ]
    for destination, source in spec.read_only_mounts:
        source_resolved = ensure_within(workspace_root, source)
        if not source_resolved.is_dir():
            raise RuntimeError(f"mount source {str(source_resolved)!r} does not exist")
        args += ["--mount", f"type=bind,src={source_resolved},dst={destination},readonly"]
    for env_name, env_value in build_container_env(request.env_extra):
        args += ["--env", f"{env_name}={env_value}"]
    for device in spec.gpu_devices:
        args += ["--device", device]
    args.append(f"{spec.image}@{spec.image_digest}")
    args.extend(request.argv)
    return args


def _tree_size(root: Path, stop_after: int | None = None) -> int:
    """Total bytes of every regular file under ``root``; when ``stop_after``
    is given, early-exits once the sum exceeds it (the monitor uses this so
    a runaway output directory costs bounded work per poll)."""
    total = 0
    for current, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
            if stop_after is not None and total > stop_after:
                return total
    return total


def _monitor(
    process: subprocess.Popen[str],
    log_paths: tuple[Path, Path],
    log_limit_bytes: int,
    output_dir: Path,
    output_limit_bytes: int,
    timeout_seconds: float,
    cancel_event: threading.Event | None,
) -> tuple[str | None, str | None]:
    """Watch the running client until it exits; returns a (status, reason)
    breach verdict of ``("timeout", ...)`` / ``("log_limit", ...)`` /
    ``("output_limit", ...)`` / ``("cancelled", ...)`` or
    ``("completed", None)``. The caller must still map "completed" to the
    final exit-code-dependent status."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled", "cancel requested by parent"
        try:
            log_bytes = sum(path.stat().st_size for path in log_paths)
        except OSError:
            log_bytes = 0
        if log_bytes > log_limit_bytes:
            return "log_limit", f"log/output stream exceeded {log_limit_bytes} bytes"
        if (
            output_limit_bytes
            and _tree_size(output_dir, stop_after=output_limit_bytes) > output_limit_bytes
        ):
            return "output_limit", f"output directory exceeded {output_limit_bytes} bytes"
        if process.poll() is not None:
            return "completed", None
        if time.monotonic() >= deadline:
            return "timeout", f"container exceeded {timeout_seconds}s budget"
        time.sleep(_MONITOR_INTERVAL_SECONDS)


def _tail_of(path: Path | None, limit: int) -> str:
    if path is None or not path.exists():
        return ""
    return _read_tail(path, limit)


def execute_container(
    request: WorkerRequest,
    spec: ContainerSpec,
    *,
    docker_command: tuple[str, ...] = ("docker",),
    cancel_event: threading.Event | None = None,
) -> WorkerOutcome:
    """Run one request inside the ADR-0001 boundary. See the module
    docstring for the lifecycle and evidence contract."""
    started = time.monotonic()
    status = "infra_error"
    exit_code: int | None = None
    killed = False
    notes: list[str] = []
    stdout_tail = ""
    stderr_tail = ""
    workdir: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    container_name = f"ka-{uuid.uuid4().hex}"
    record: dict = {
        "container_name": container_name,
        "image": spec.reference,
        "gpu_devices": list(spec.gpu_devices),
        "user": spec.user,
        "request_id": request.request_id,
    }
    try:
        if shutil.which(docker_command[0]) is None and not Path(docker_command[0]).exists():
            _raise_infra(f"docker command {docker_command[0]!r} not found")
        digest_ref = spec.reference
        inspect = _docker(docker_command, "image", "inspect", digest_ref)
        if inspect.returncode != 0:
            _raise_infra(
                f"image {digest_ref} not present locally; refusing to auto-pull "
                f"(stderr: {inspect.stderr.strip()[-200:]})"
            )
        record["image_present"] = True
        workdir = ensure_within(
            request.workspace_root, request.workspace_root / f"container-{uuid.uuid4().hex}"
        )
        try:
            workdir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise RuntimeError(f"work directory creation failed: {exc}") from exc
        stdout_path = workdir / "stdout.log"
        stderr_path = workdir / "stderr.log"
        output_dir = workdir / "out"
        output_dir.mkdir(exist_ok=True)
        os.chmod(output_dir, 0o777)  # container uid 20000 writes; parent owns the directory
        args = _run_args(request, spec, container_name, request.workspace_root, output_dir)
        try:
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                process = subprocess.Popen(
                    [*docker_command, *args],
                    cwd=str(workdir),
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
        except OSError as exc:
            remove_container(docker_command, container_name)
            raise RuntimeError(f"docker client start failed: {exc}") from exc
        record["docker_client_pid"] = process.pid
        record["output_dir"] = str(output_dir)
        verdict, verdict_reason = _monitor(
            process,
            (stdout_path, stderr_path),
            spec.log_limit_bytes,
            output_dir,
            spec.output_limit_bytes,
            request.timeout_seconds,
            cancel_event,
        )
        if verdict != "completed":
            # The request budget ends the container: kill makes the client
            # return, and every PID-namespace descendant dies with it.
            killed = _docker(docker_command, "kill", container_name, timeout=30.0).returncode == 0
            if not killed:
                notes.append("container kill could not be confirmed; forcing removal")
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
                notes.append("docker client did not exit after container kill")
            exit_code = process.returncode
            if verdict == "timeout":
                status = "timeout"
            elif verdict == "cancelled":
                status = "timeout"
                notes.append(f"[cancelled] {verdict_reason}")
            elif verdict == "log_limit":
                status = "failed"
                notes.append(f"[log_limit] {verdict_reason}")
            else:
                status = "failed"
                notes.append(f"[output_limit] {verdict_reason}")
        else:
            exit_code = process.returncode
            status = "completed" if exit_code == 0 else "failed"
        # The container may be stopped/created/exited here; state inspection
        # is the trusted record of what actually happened inside.
        state = _docker(docker_command, "inspect", "--type", "container", container_name)
        if state.returncode == 0:
            try:
                info = json.loads(state.stdout)[0]
                container_state = info.get("State", {})
                record["container_id"] = info.get("Id")
                record["state"] = {
                    "status": container_state.get("Status"),
                    "exit_code": container_state.get("ExitCode"),
                    "oom_killed": container_state.get("OOMKilled"),
                }
                if container_state.get("OOMKilled"):
                    if status == "completed":
                        status = "failed"
                    notes.append("[oom_killed] memory cgroup limit triggered")
                if exit_code is None:
                    exit_code = container_state.get("ExitCode")
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                notes.append(f"container inspect parse failed: {exc}")
        else:
            notes.append("container inspect failed after run; state unrecorded")
        # The output bind mount lands directly in the private work
        # directory; record its size as the trusted accounting entry.
        record["output_bytes"] = _tree_size(output_dir)
    except (RuntimeError, OSError, subprocess.TimeoutExpired, PathOutsideWorkspaceError) as exc:
        status = "infra_error"
        notes.append(f"[infra] {exc}")
    finally:
        try:
            if container_exists(docker_command, container_name):
                remove_container(docker_command, container_name)
            if container_exists(docker_command, container_name):
                status = "infra_error"
                notes.append("[infra] container removal could not be confirmed")
                record["reclaimed"] = False
            else:
                record["reclaimed"] = True
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = "infra_error"
            notes.append(f"[infra] container reclamation check failed: {exc}")
            record["reclaimed"] = False
        if workdir is not None:
            try:
                (workdir / "container-record.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
                )
            except OSError as exc:
                notes.append(f"[infra] container record write failed: {exc}")
    stdout_tail = _tail_of(stdout_path, request.output_tail_chars)
    stderr_tail = _tail_of(stderr_path, request.output_tail_chars)
    if notes:
        stderr_tail = "\n".join([*notes, stderr_tail]).strip("\n")
    return WorkerOutcome(
        request_id=request.request_id,
        status=status,
        exit_code=exit_code,
        duration_seconds=time.monotonic() - started,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        killed=killed,
        workdir=str(workdir) if workdir is not None and workdir.exists() else None,
    )
