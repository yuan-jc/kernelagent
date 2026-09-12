"""Process mechanics: scrubbed environments, tree kill, execution, and
trusted-directory tamper detection.

Platform notes: the tree kill uses taskkill /T /F on Windows and
setsid + killpg(SIGKILL) on POSIX. Environment scrubbing denies names
containing secret-shaped markers case-insensitively (Windows env names are
case-insensitive). This module does not pretend to be a sandbox: OS-level
write enforcement belongs to the parent T04 package (containers)."""

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
    """Relative path -> sha256 for every file under ``root`` (deterministic)."""
    root = Path(root)
    hashes: dict[str, str] = {}
    if not root.exists():
        return hashes
    for current, _dirs, files in os.walk(root):
        for name in files:
            file_path = Path(current) / name
            relative = file_path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(file_path.read_bytes()).hexdigest()
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


def _kill_tree(pid: int) -> None:
    """Kill the whole process tree. Best effort: a failed fallback kill is
    still attempted and the caller reports the outcome via the child's
    reap step."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _read_tail(path: Path, limit: int) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def execute(request: WorkerRequest) -> WorkerOutcome:
    """Run the payload in an isolated child with a scrubbed environment and a
    private work directory; on timeout, kill the whole process tree."""
    started = time.monotonic()
    workdir = (
        Path(request.workspace_root)
        / f"{_WORKDIR_PREFIX}{request.request_id}-{uuid.uuid4().hex[:8]}"
    )
    workdir.mkdir(parents=True, exist_ok=False)
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    env = build_child_env(extra=request.env_extra)
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True  # own process group for killpg
    killed = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(
            list(request.argv),
            cwd=str(workdir),
            env=env,
            stdout=out,
            stderr=err,
            **popen_kwargs,
        )
        try:
            exit_code = process.wait(timeout=request.timeout_seconds)
            status = "completed" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            killed = True
            status = "timeout"
            exit_code = None
            _kill_tree(process.pid)
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=30)
    duration = time.monotonic() - started
    outcome = WorkerOutcome(
        request_id=request.request_id,
        status=status,
        exit_code=exit_code,
        duration_seconds=duration,
        stdout_tail=_read_tail(stdout_path, request.output_tail_chars),
        stderr_tail=_read_tail(stderr_path, request.output_tail_chars),
        killed=killed,
        workdir=str(workdir)
        if (status != "completed" and request.keep_workdir_on_failure)
        else None,
    )
    if outcome.workdir is None:
        # The payload may have left arbitrary files in its work directory.
        shutil.rmtree(workdir, ignore_errors=True)
    return outcome
