"""Process mechanics: scrubbed environments, tree kill, execution, and
trusted-directory tamper detection.

Lifecycle contract: every execution runs inside a private work directory
created under the workspace (random name, boundary-checked), the whole
process tree ends with the request - terminated on timeout and reaped after
the payload exits (Windows Job Object with kill-on-close; POSIX process-group
kill) - and cleanup is verified: a directory that could not be deleted is
reported in the outcome instead of being claimed clean. Startup failures
become ``infra_error`` outcomes, so the orchestrator always receives a
structured record.

Platform limits, documented honestly: a POSIX descendant that escapes its
process group (setsid) survives the best-effort killpg, and a Windows
payload could theoretically race the Job Object assignment. OS-enforced
isolation (containers/cgroups with real boundaries) is the parent T04
package on the target Ubuntu environment; this module is not a sandbox."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kernelagent.worker.errors import (
    PathOutsideWorkspaceError,
    TrustedDirModifiedError,
)
from kernelagent.worker.protocol import WorkerOutcome, WorkerRequest

SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PASSWD")
EXTRA_DENIED_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ZHIPUAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SECRET_ACCESS_KEY",
    }
)
_WORKDIR_PREFIX = "payload-"
_HASH_CHUNK_BYTES = 1024 * 1024


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS) or name in EXTRA_DENIED_NAMES


def build_child_env(
    extra: tuple[tuple[str, str], ...] = (),
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Scrub the parent environment (deny secret-shaped names) and merge
    validated extras. Extras cannot re-introduce denied names."""
    source = dict(os.environ if base is None else base)
    env = {name: value for name, value in source.items() if not _is_secret_name(name)}
    for name, value in extra:
        if _is_secret_name(name):
            raise ValueError(f"env_extra name {name!r} matches the secret denylist")
        env[name] = value
    return env


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and require it to stay inside ``base``."""
    base_resolved = Path(base).resolve()
    target_resolved = Path(target).resolve()
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise PathOutsideWorkspaceError(
            f"{str(target)!r} resolves outside workspace {str(base_resolved)!r}"
        ) from exc
    return target_resolved


def snapshot_dir_hashes(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every file under ``root``, read in bounded
    chunks: whole-file reads would let a large reference file exhaust the
    trusted parent's memory."""
    root = Path(root)
    hashes: dict[str, str] = {}
    if not root.exists():
        return hashes
    for current, _dirs, files in os.walk(root):
        for name in files:
            file_path = Path(current) / name
            digest = hashlib.sha256()
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                    digest.update(chunk)
            hashes[file_path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def assert_dir_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if changed:
        raise TrustedDirModifiedError(f"trusted directory changed: {changed}")


def process_exists(pid: int) -> bool:
    """Existence check for an arbitrary pid, without killing it."""
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class _WindowsJob:
    """Best-effort Job Object with KILL_ON_JOB_CLOSE: every descendant the
    payload spawns (after assignment) dies when the job is terminated or
    closed, even if the direct payload already exited."""

    def __init__(self) -> None:
        self.handle = None
        if sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.windll.kernel32

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    (name, ctypes.c_uint64)
                    for name in (
                        "ReadOperationCount",
                        "WriteOperationCount",
                        "OtherOperationCount",
                        "ReadTransferCount",
                        "WriteTransferCount",
                        "OtherTransferCount",
                    )
                ]

            class BASIC_LIMIT_INFO(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class EXTENDED_LIMIT_INFO(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BASIC_LIMIT_INFO),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = EXTENDED_LIMIT_INFO()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                kernel32.CloseHandle(handle)
                return
            self.handle = handle
        except Exception:  # noqa: BLE001 - the job is an enhancement, never a crash
            self.handle = None

    def assign(self, process: subprocess.Popen) -> None:
        if self.handle:
            ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(process._handle))

    def terminate(self) -> None:
        if self.handle:
            ctypes.windll.kernel32.TerminateJobObject(self.handle, 0x40000001)

    def close(self) -> None:
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


def _kill_process_group(pgid: int | None, pid: int) -> None:
    """Kill the captured process group (POSIX) or the process tree via
    taskkill (Windows), then fall back to killing the direct child."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _read_tail(path: Path, limit: int) -> str:
    """Read at most ``limit`` bytes from the end of the file. A multi-gigabyte
    candidate log must never be loaded whole into the trusted parent."""
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        data = handle.read(limit)
    return data.decode("utf-8", errors="replace")


def execute(request: WorkerRequest) -> WorkerOutcome:
    """Run the payload in an isolated child with a scrubbed environment and a
    private, boundary-checked work directory. The whole process tree ends
    with the request budget; startup failures become ``infra_error``
    outcomes; an undeletable work directory is reported, never claimed
    clean."""
    started = time.monotonic()
    workdir = ensure_within(
        request.workspace_root,
        request.workspace_root / f"{_WORKDIR_PREFIX}{uuid.uuid4().hex}",
    )
    workdir.mkdir(parents=True, exist_ok=False)
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    env = build_child_env(extra=request.env_extra)

    status = "infra_error"
    exit_code: int | None = None
    killed = False
    startup_error = ""
    job = _WindowsJob()
    pgid: int | None = None
    process: subprocess.Popen | None = None
    try:
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            try:
                process = subprocess.Popen(
                    list(request.argv),
                    cwd=str(workdir),
                    env=env,
                    stdout=out,
                    stderr=err,
                    **({"start_new_session": True} if sys.platform != "win32" else {}),
                )
            except OSError as exc:
                startup_error = f"{type(exc).__name__}: {exc}"
            else:
                job.assign(process)
                if sys.platform != "win32":
                    try:
                        pgid = os.getpgid(process.pid)
                    except ProcessLookupError:
                        pgid = None
                try:
                    try:
                        exit_code = process.wait(timeout=request.timeout_seconds)
                        status = "completed" if exit_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        killed = True
                        status = "timeout"
                        if sys.platform == "win32":
                            job.terminate()
                        _kill_process_group(pgid, process.pid)
                        try:
                            exit_code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            exit_code = process.wait(timeout=30)
                finally:
                    if sys.platform != "win32" and pgid is not None:
                        # The request budget covers descendants: reap anything
                        # left in the process group (best effort, same group;
                        # a setsid escapee is a documented parent-T04 limit).
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
    finally:
        job.close()  # kill-on-close ends any Windows descendants still alive

    duration = time.monotonic() - started
    stdout_tail = _read_tail(stdout_path, request.output_tail_chars)
    stderr_tail = _read_tail(stderr_path, request.output_tail_chars)
    if status == "infra_error":
        stderr_tail = stderr_tail or startup_error
    should_keep = request.keep_workdir_on_failure and status != "completed"
    if not should_keep:
        shutil.rmtree(workdir, ignore_errors=True)
    kept = str(workdir) if workdir.exists() else None
    outcome = WorkerOutcome(
        request_id=request.request_id,
        status=status,
        exit_code=exit_code,
        duration_seconds=duration,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        killed=killed,
        workdir=kept,
    )
    return outcome
