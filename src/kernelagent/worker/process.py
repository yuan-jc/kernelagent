"""Process mechanics: scrubbed environments, tree kill, execution, and
trusted-directory tamper detection.

Lifecycle contract: every execution runs inside a private work directory
created under the workspace (random name, boundary-checked), the whole
process tree ends with the request - terminated on timeout and reaped after
the payload exits (Windows Job Object with kill-on-close; POSIX process-group
kill) - and cleanup is verified: a directory that could not be deleted is
reported in the outcome instead of being claimed clean.

Every infrastructure step that could break the lifecycle guarantee (Job
creation, configuration, process attribution, reclamation, log access) is
checked: failure refuses the run or produces a structured ``infra_error``
outcome with the reason, never a silent ``completed``. Platform limits,
documented honestly: a POSIX descendant that escapes its process group
(setsid) survives the best-effort killpg. Windows payloads remain suspended
until Job Object assignment succeeds. OS-enforced isolation
(containers/cgroups with real boundaries) is the parent T04 package on the
target Ubuntu environment; this module is not a sandbox."""

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


class _InfraFailure(Exception):
    """An infrastructure step failed; the run must not report completed."""


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS) or name in EXTRA_DENIED_NAMES


# Backward-compatible private alias.
_is_secret_name = is_secret_name


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
    """Job Object with KILL_ON_JOB_CLOSE: every descendant the payload spawns
    (after assignment) dies when the job is terminated or closed, even if the
    direct payload already exited. Win32 results are checked and failures are
    recorded in ``error`` so the caller can refuse an unmanaged run."""

    def __init__(self) -> None:
        self.handle = None
        self.error = ""
        if sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.windll.kernel32

            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            kernel32.SetInformationJobObject.restype = ctypes.c_int
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel32.AssignProcessToJobObject.restype = ctypes.c_int
            kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.TerminateJobObject.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.GetLastError.restype = ctypes.c_uint32

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
                self.error = f"CreateJobObjectW failed (GetLastError={kernel32.GetLastError()})"
                return
            info = EXTENDED_LIMIT_INFO()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                self.error = (
                    f"SetInformationJobObject failed (GetLastError={kernel32.GetLastError()})"
                )
                kernel32.CloseHandle(handle)
                return
            self.handle = handle
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced as infra failure
            self.error = f"{type(exc).__name__}: {exc}"
            self.handle = None

    def assign(self, process: subprocess.Popen) -> bool:
        if not self.handle:
            return False
        result = ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(process._handle))
        if not result:
            last_error = ctypes.windll.kernel32.GetLastError()
            self.error = f"AssignProcessToJobObject failed (GetLastError={last_error})"
        return bool(result)

    def terminate(self) -> bool:
        if not self.handle:
            return False
        result = ctypes.windll.kernel32.TerminateJobObject(self.handle, 0x40000001)
        if not result:
            last_error = ctypes.windll.kernel32.GetLastError()
            self.error = f"TerminateJobObject failed (GetLastError={last_error})"
        return bool(result)

    def close(self) -> bool:
        if self.handle:
            if not ctypes.windll.kernel32.CloseHandle(self.handle):
                self.error = (
                    f"CloseHandle failed (GetLastError={ctypes.windll.kernel32.GetLastError()})"
                )
                return False
            self.handle = None
        return True


def _resume_process_threads(pid: int) -> bool:
    """Resume every thread of a CREATE_SUSPENDED payload; returns whether at
    least one thread was resumed."""
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPTHREAD = 0x4
    THREAD_SUSPEND_RESUME = 0x2
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenThread.restype = ctypes.c_void_p
    kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
    kernel32.ResumeThread.restype = ctypes.c_uint32

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ThreadID", ctypes.c_uint32),
            ("th32OwnerProcessID", ctypes.c_uint32),
            ("tpBasePri", ctypes.c_int32),
            ("tpDeltaPri", ctypes.c_int32),
            ("dwFlags", ctypes.c_uint32),
        ]

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        return False
    resumed = False
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        for name in ("Thread32First", "Thread32Next"):
            function = getattr(kernel32, name)
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
            function.restype = ctypes.c_int
        ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if thread:
                    try:
                        if kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                            return False
                        resumed = True
                    finally:
                        kernel32.CloseHandle(thread)
            ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return resumed


def _kill_process_group(pgid: int | None, pid: int) -> bool:
    """Kill the captured process group (POSIX) or the process tree via
    taskkill (Windows); returns whether the reclamation was confirmed.
    Falls back to killing the direct child."""
    if sys.platform == "win32":
        try:
            completed = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True  # nothing left in the group
        except PermissionError:
            return False
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False


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
    with the request budget; infrastructure failures (Job Object, log files,
    process start, reclamation, log read-back) become structured
    ``infra_error`` outcomes with the reason, and an undeletable work
    directory is reported, never claimed clean."""
    started = time.monotonic()
    status = "infra_error"
    exit_code: int | None = None
    killed = False
    infra_reason = ""
    stdout_tail = ""
    stderr_tail = ""
    workdir: Path | None = None
    job = _WindowsJob()
    pgid: int | None = None
    process: subprocess.Popen | None = None
    try:
        if sys.platform == "win32" and job.handle is None:
            # Refusing here is the guarantee: without the job there is no
            # descendant reclamation, and a completed report would be a lie.
            raise _InfraFailure(f"Windows Job Object unavailable: {job.error or 'unknown'}")
        try:
            env = build_child_env(extra=request.env_extra)
        except ValueError as exc:
            raise _InfraFailure(f"invalid child environment: {exc}") from exc
        workdir = ensure_within(
            request.workspace_root,
            request.workspace_root / f"{_WORKDIR_PREFIX}{uuid.uuid4().hex}",
        )
        try:
            workdir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise _InfraFailure(f"work directory creation failed: {exc}") from exc
        stdout_path = workdir / "stdout.log"
        stderr_path = workdir / "stderr.log"
        out_handle = None
        try:
            out_handle = stdout_path.open("wb")
            err_handle = stderr_path.open("wb")
        except OSError as exc:
            if out_handle is not None:
                out_handle.close()
            raise _InfraFailure(f"log file open failed: {exc}") from exc
        with out_handle as out, err_handle as err:
            win32_flags = {"creationflags": 0x4} if sys.platform == "win32" else {}
            # CREATE_SUSPENDED on Windows: the payload is attributed to the
            # job (or terminated) before a single instruction runs, so an
            # attribution failure can never leave running descendants.
            try:
                process = subprocess.Popen(
                    list(request.argv),
                    cwd=str(workdir),
                    env=env,
                    stdout=out,
                    stderr=err,
                    **({"start_new_session": True} if sys.platform != "win32" else {}),
                    **win32_flags,
                )
            except OSError as exc:
                raise _InfraFailure(f"process start failed: {exc}") from exc
            if sys.platform == "win32":
                if not job.assign(process):
                    # Child is still suspended: terminate it before it ever
                    # runs, so no descendant can escape the failed
                    # attribution.
                    process.kill()
                    process.wait(timeout=30)
                    raise _InfraFailure(
                        "process attribution failed: "
                        f"{job.error or 'AssignProcessToJobObject returned failure'}"
                    )
                if not _resume_process_threads(process.pid):
                    process.kill()
                    raise _InfraFailure("failed to resume the suspended payload threads")
            else:
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
                    reclaimed = job.terminate() if sys.platform == "win32" else False
                    if not reclaimed:
                        reclaimed = _kill_process_group(pgid, process.pid)
                    try:
                        exit_code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        exit_code = process.wait(timeout=30)
                    if not reclaimed:
                        raise _InfraFailure(
                            "process tree reclamation could not be confirmed after timeout; "
                            "unreclaimed state reported instead of a completed run"
                        )
                finally:
                    if sys.platform != "win32" and pgid is not None:
                        # The request budget covers descendants: reap anything
                        # left in the process group (best effort, same group;
                        # a setsid escapee is a documented parent-T04 limit).
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
            except OSError as exc:
                raise _InfraFailure(f"process reclamation failed: {exc}") from exc
        try:
            stdout_tail = _read_tail(stdout_path, request.output_tail_chars)
            stderr_tail = _read_tail(stderr_path, request.output_tail_chars)
        except OSError as exc:
            stdout_tail, stderr_tail = "", f"[tail-read-error] {type(exc).__name__}: {exc}"
            raise _InfraFailure("log read-back failed") from exc
    except (_InfraFailure, OSError, subprocess.TimeoutExpired) as exc:
        infra_reason = str(exc)
        status = "infra_error"
    finally:
        # Failures before/during wait must still reap the direct payload,
        # including suspended Windows processes that could not be resumed.
        if process is not None and process.poll() is None:
            try:
                process.kill()
                killed = True
                exit_code = process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                status = "infra_error"
                infra_reason += f"; direct process reclamation failed: {exc}"
        if not job.close():
            status = "infra_error"
            infra_reason += f"; {job.error or 'job close failed'}"
    if infra_reason:
        stderr_tail = "\n".join(part for part in (infra_reason, stderr_tail) if part)
    duration = time.monotonic() - started
    should_keep = request.keep_workdir_on_failure and status != "completed"
    if workdir is not None and not should_keep:
        shutil.rmtree(workdir, ignore_errors=True)
    kept = str(workdir) if (workdir is not None and workdir.exists()) else None
    return WorkerOutcome(
        request_id=request.request_id,
        status=status,
        exit_code=exit_code,
        duration_seconds=duration,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        killed=killed,
        workdir=kept,
    )
