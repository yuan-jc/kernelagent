"""HTTP server for the local web console (standard library only).

Endpoints:
- ``GET /``                      the console (SPA dist when built, else the
  legacy single-file page from ``webapp/static``)
- ``GET /assets/<path>``         hashed Vite build assets (MIME whitelist)
- ``GET /<route>``               SPA history-route fallback to index.html
  (only when the SPA dist exists; otherwise unmatched paths stay 404)
- ``GET /api/health``            liveness + active run
- ``GET /api/problems``          bench/level/problem tree from the pinned snapshot
- ``GET /api/runs``              run history (durable state per run directory)
- ``GET /api/runs/<id>``         one run snapshot (state/budget/stages/speedups)
- ``GET /api/runs/<id>/report``  report.json verbatim (404 when missing)
- ``GET /api/runs/<id>/journal?after=N``  journal tail from line N+1 on
  (cursor polling; entries pass through verbatim, plus a 1-based ``seq``)
- ``GET /api/runs/<id>/records``  read-only record listing
- ``GET /api/runs/<id>/records/<action>``          one record, secret-scrubbed
- ``GET /api/runs/<id>/records/<action>/progress``  one progress record
- ``GET /api/runs/<id>/workspace``                workspace top-level listing
- ``GET /api/runs/<id>/workspace/file?path=...``   one text file (whitelist,
  256 KiB cap, traversal/symlink-escape rejected)
- ``POST /api/runs``        start a run: {mode, problem spec, model,
  base_url, api_key, budgets}; the key stays in memory for that job only.
- ``POST /api/models``      list the models a key can use (key in memory only).

One GPU means one active run; a second start is a 409. Every durable fact
shown in the UI comes from the run directory (journal, records, report) -
the server never invents status. The SPA dist location defaults to
``<repo>/frontend/dist`` and can be overridden with ``KA_FRONTEND_DIST``."""

from __future__ import annotations

import json
import os
import posixpath
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlsplit

from kernelagent.adapters.models.errors import PermanentModelError, TransientModelError
from kernelagent.adapters.models.openai_compat import list_models
from kernelagent.config import API_KEY_ENV
from kernelagent.optimization import (
    OptimizationConfig,
    OptimizationConfigError,
    default_gpu_device,
    optimize,
    parse_problem_spec,
    reconstruct_budget,
    resolve_problem,
)
from kernelagent.orchestrator import Journal
from kernelagent.webapp.jobs import MODES, build_generator, list_problems

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_FIELDS_DEFAULTS = {
    "max_candidates": 5,
    "max_repair_rounds": 2,
    "gpu_budget_seconds": 1800.0,
    "token_budget": 200000,
}

# -- static assets (P1) -----------------------------------------------------

ASSET_MIME: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".txt": "text/plain; charset=utf-8",
}

# -- workspace text files (P5) -----------------------------------------------

WORKSPACE_MAX_BYTES = 256 * 1024
WORKSPACE_TEXT_EXTENSIONS = frozenset(
    {
        ".bash",
        ".c",
        ".cfg",
        ".cpp",
        ".css",
        ".csv",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".ts",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ACTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# -- record redaction (P4) ---------------------------------------------------
# The API key lives only in the generator client's memory, but records are
# third-party-shaped artifacts, so anything that can echo a credential is
# scrubbed before it leaves the server:
# 1. dict keys that are credential-ish get their string value replaced;
# 2. any string equal to a whole environment variable value (len >= 8) is
#    replaced (catches exact echoes, e.g. a pasted key or token);
# 3. values of env vars whose name looks secret-bearing are also scrubbed as
#    substrings (catches "Bearer <key>" style embedding).

REDACTED = "[REDACTED]"
_SECRET_ENV_NAME_RE = re.compile(r"key|secret|token|password|passwd|credential|auth", re.IGNORECASE)
_SECRET_FIELD_RE = re.compile(r"^(api_?key|authorization|secret|password|passwd)$", re.IGNORECASE)
_MIN_SECRET_LEN = 8


class HttpError(Exception):
    """A readable error mapped to an HTTP status by the handler."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def frontend_dist_dir() -> Path | None:
    """The built SPA directory (with index.html), or None when absent.

    ``KA_FRONTEND_DIST`` overrides the default ``<repo>/frontend/dist`` so
    tests and alternative deployments can point elsewhere."""
    env = os.environ.get("KA_FRONTEND_DIST")
    root = Path(env).expanduser() if env else REPO_ROOT / "frontend" / "dist"
    return root if (root / "index.html").is_file() else None


def _int_arg(query: dict[str, list[str]], name: str, default: int) -> int:
    raw = (query.get(name) or [None])[0]
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise HttpError(400, f"query parameter {name!r} must be an integer; got {raw!r}") from None
    if value < 0:
        raise HttpError(400, f"query parameter {name!r} must be >= 0")
    return value


def _query_arg(query: dict[str, list[str]], name: str) -> str:
    return (query.get(name) or [""])[0]


class ActiveRunError(RuntimeError):
    """Raised when a start is requested while another run holds the GPU."""


class WebApp:
    """Shared state: roots, the single-GPU run lock, and job threads."""

    def __init__(
        self,
        runs_root: Path,
        snapshot_root: Path,
        fixtures_root: Path = Path("configs/kernelbench/eval-fixtures"),
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root = Path(snapshot_root)
        self.fixtures_root = Path(fixtures_root)
        self.gpu_device = default_gpu_device()
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    # -- starting runs ----------------------------------------------------

    def active_run_id(self) -> str | None:
        for run_id, thread in self._threads.items():
            if thread.is_alive():
                return run_id
        return None

    def start_run(self, payload: dict) -> dict:
        mode = str(payload.get("mode") or "live")
        if mode not in MODES:
            raise OptimizationConfigError(f"mode must be one of {MODES}; got {mode!r}")
        api_key = str(payload.get("api_key") or "")
        problem = str(payload.get("problem") or "")
        spec = parse_problem_spec(problem)
        resolve_problem(spec, self.snapshot_root)
        with self._lock:
            active = self.active_run_id()
            if active:
                raise ActiveRunError(f"run {active} is still using the GPU")
            run_id = time.strftime("%Y%m%d-%H%M%S")
            config = OptimizationConfig(
                problem=problem,
                backend=str(payload.get("backend") or "triton"),
                model_id=str(payload.get("model") or "glm-4.5"),
                base_url=str(payload.get("base_url") or ""),
                max_candidates=int(
                    payload.get("max_candidates", _JOB_FIELDS_DEFAULTS["max_candidates"])
                ),
                max_repair_rounds=int(
                    payload.get("max_repair_rounds", _JOB_FIELDS_DEFAULTS["max_repair_rounds"])
                ),
                gpu_budget_seconds=float(
                    payload.get("gpu_budget_seconds", _JOB_FIELDS_DEFAULTS["gpu_budget_seconds"])
                ),
                token_budget=int(payload.get("token_budget", _JOB_FIELDS_DEFAULTS["token_budget"])),
                output=self.runs_root / run_id,
                snapshot_root=self.snapshot_root,
                gpu_device=self.gpu_device,
            )
            (config.output).mkdir(parents=True, exist_ok=True)
            # job.json is the UI's durable config record: never contains the key.
            (config.output / "job.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "mode": mode,
                        "config": {
                            "problem": config.problem,
                            "backend": config.backend,
                            "model_id": config.model_id,
                            "base_url": config.base_url,
                            "max_candidates": config.max_candidates,
                            "max_repair_rounds": config.max_repair_rounds,
                            "gpu_budget_seconds": config.gpu_budget_seconds,
                            "token_budget": config.token_budget,
                            "gpu_device": config.gpu_device,
                        },
                        "started_at": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            thread = threading.Thread(
                target=self._run_job,
                args=(mode, config, api_key, run_id, bool(payload.get("disable_thinking"))),
                daemon=True,
                name=f"optimize-{run_id}",
            )
            self._threads[run_id] = thread
            thread.start()
        return {"run_id": run_id, "state": "running"}

    def _run_job(
        self,
        mode: str,
        config: OptimizationConfig,
        api_key: str,
        run_id: str,
        disable_thinking: bool = False,
    ) -> None:
        try:
            generator = build_generator(
                mode,
                config.base_url,
                api_key,
                disable_thinking=disable_thinking,
                fixtures_root=self.fixtures_root,
            )
            optimize(config, generator=generator)
        except Exception:  # noqa: BLE001 - the job's error channel is the run dir
            (config.output / "error.txt").write_text(
                traceback.format_exc()[-4000:], encoding="utf-8"
            )

    # -- read models -------------------------------------------------------

    def problems(self) -> dict:
        return list_problems(self.snapshot_root)

    def runs(self) -> list[dict]:
        summaries = []
        for run_dir in sorted(self.runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            snap = run_snapshot(run_dir)
            if snap is None:
                continue
            summaries.append(
                {
                    "run_id": snap["run_id"],
                    "state": snap["state"],
                    "mode": (snap.get("job") or {}).get("mode"),
                    "problem": ((snap.get("job") or {}).get("config") or {}).get("problem"),
                    # report.json writes champion: null for no_improvement runs;
                    # .get(key, {}) does not replace an existing null value.
                    "champion": (snap.get("champion") or {}).get("candidate_sha256"),
                    "started_at": run_started_at(run_dir, snap),
                }
            )
        return summaries


def run_snapshot(run_dir: Path) -> dict | None:
    """Durable status of one run directory; the UI's only source of truth."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    snap: dict = {
        "run_id": run_dir.name,
        "state": "unknown",
        "budget": None,
        "in_flight": [],
        "progress": {},
        "candidates": [],
        "champion": None,
        "error_tail": None,
        "job": _read_json(run_dir / "job.json"),
    }
    report = _read_json(run_dir / "report.json")
    if report:
        snap["state"] = report.get("state", "unknown")
        snap["champion"] = report.get("champion")
        snap["candidates"] = report.get("candidates") or []
        snap["budget"] = report.get("budget")
        snap["candidate_trust"] = report.get("candidate_trust")
        snap["adversarially_secure"] = report.get("adversarially_secure")
    error_file = run_dir / "error.txt"
    if error_file.is_file():
        snap["error_tail"] = error_file.read_text(encoding="utf-8")[-1500:]
        if snap["state"] in ("unknown", "running"):
            snap["state"] = "infra_error"

    journal_path = run_dir / "journal.jsonl"
    if journal_path.is_file():
        try:
            journal = Journal(journal_path)
        except ValueError:
            snap["state"] = "journal_corrupt"
            return snap
        durable = reconstruct_budget(journal.entries)
        started: list[str] = []
        finished: set[str] = set()
        for entry in journal.entries:
            if entry["kind"] == "action_started":
                started.append(entry.get("action_id") or "")
            elif entry["kind"] in ("action_finished", "action_interrupted"):
                name = entry.get("action_id") or ""
                if name in started:
                    started.remove(name)
                finished.add(name)
        snap["in_flight"] = [name for name in started if name]
        for name in snap["in_flight"]:
            progress = _read_json(run_dir / "records" / f"{name}.progress.json")
            if progress:
                snap["progress"][name] = progress.get("stage")
        if snap["budget"] is None:
            limits = ((snap.get("job") or {}).get("config")) or {}
            snap["budget"] = {
                "gpu_seconds_limit": limits.get("gpu_budget_seconds"),
                "tokens_limit": limits.get("token_budget"),
                "settled_gpu_seconds": durable.settled_gpu_seconds,
                "settled_tokens": durable.settled_tokens,
                "reserved_gpu_seconds": durable.reserved_gpu_seconds,
                "reserved_tokens": durable.reserved_tokens,
            }
        if not report:
            # No report yet: a journal with any activity means a live or
            # interrupted run; the UI shows it as running until a terminal
            # report exists.
            snap["state"] = "running"
    return snap


TERMINAL_STATES = {"completed", "no_improvement", "budget_exhausted", "infra_error"}


def _read_json(path: Path) -> dict | None:
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# -- run report (P2) ----------------------------------------------------------


def run_report(run_id: str, run_dir: Path) -> dict:
    """report.json verbatim: the identity chain (commit/protocol/sha) the UI
    shows in the evidence tab. Missing is 404; a corrupt file is a 500, not a
    fabricated report."""
    _require_run_dir(run_id, run_dir)
    path = run_dir / "report.json"
    if not path.is_file():
        raise HttpError(404, f"run {run_id} has no report.json yet")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HttpError(500, f"report.json for run {run_id} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HttpError(500, f"report.json for run {run_id} must contain a JSON object")
    return data


# -- journal tail (P3) ---------------------------------------------------------


def run_journal_tail(run_id: str, run_dir: Path, after: int) -> dict:
    """Journal entries after the client's cursor (0-based consumed-line count).

    Entries pass through verbatim (all kinds, including ``stage_started`` and
    ``manifest_revised``) plus a 1-based ``seq`` equal to the file line number.
    A partially written final line (torn append) is withheld until complete:
    the cursor only advances over whole JSON lines."""
    _require_run_dir(run_id, run_dir)
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        raise HttpError(404, f"run {run_id} has no journal.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    start = min(max(after, 0), total)
    entries: list[dict] = []
    for index in range(start, total):
        line = lines[index]
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == total - 1:
                total = index  # torn tail: stop before it, cursor stays back
                break
            raise HttpError(
                500, f"journal entry at line {index + 1} of run {run_id} is not valid JSON"
            ) from exc
        entry = dict(entry)
        entry["seq"] = index + 1
        entries.append(entry)
    return {"run_id": run_id, "next_after": start + len(entries), "entries": entries}


# -- records (P4) ---------------------------------------------------------------


def run_records_list(run_id: str, run_dir: Path) -> dict:
    """Read-only listing of the run's ``records/`` directory."""
    _require_run_dir(run_id, run_dir)
    records_dir = run_dir / "records"
    if not records_dir.is_dir():
        raise HttpError(404, f"run {run_id} has no records directory")
    records = []
    for record_file in sorted(records_dir.glob("*.json")):
        if not record_file.is_file():
            continue
        name = record_file.name
        variant = "progress" if name.endswith(".progress.json") else "final"
        stem = name[: -len(".progress.json")] if variant == "progress" else name[: -len(".json")]
        records.append(
            {
                "record": stem,
                "variant": variant,
                "file": name,
                "size": record_file.stat().st_size,
            }
        )
    return {"run_id": run_id, "records": records}


def run_record(run_id: str, run_dir: Path, action_id: str) -> dict:
    """One ``records/<action>.json``, scrubbed of credential-shaped content."""
    data = _read_record(run_id, run_dir, action_id, ".json")
    return _scrub_secrets(data)


def run_record_progress(run_id: str, run_dir: Path, action_id: str) -> dict:
    """One ``records/<action>.progress.json`` ({stage, ts}), secret-scrubbed."""
    data = _read_record(run_id, run_dir, action_id, ".progress.json")
    return _scrub_secrets(data)


def _read_record(run_id: str, run_dir: Path, action_id: str, suffix: str) -> dict:
    _require_run_dir(run_id, run_dir)
    if not _ACTION_ID_RE.fullmatch(action_id or ""):
        raise HttpError(404, f"invalid record id {action_id!r}")
    path = run_dir / "records" / f"{action_id}{suffix}"
    if not path.is_file():
        raise HttpError(404, f"run {run_id} has no record {action_id!r} ({suffix.strip('.')})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HttpError(
            500, f"record {action_id}{suffix} of run {run_id} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise HttpError(500, f"record {action_id}{suffix} of run {run_id} must be a JSON object")
    return data


def _scrub_secrets(node: object) -> object:
    """Return ``node`` with credential-shaped strings replaced by REDACTED."""
    exact: set[str] = set()
    substrings: list[str] = []
    for name, value in os.environ.items():
        if len(value) < _MIN_SECRET_LEN:
            continue
        exact.add(value)
        if _SECRET_ENV_NAME_RE.search(name):
            substrings.append(value)

    def walk(item: object) -> object:
        if isinstance(item, dict):
            return {
                key: (
                    REDACTED
                    if isinstance(value, str) and _SECRET_FIELD_RE.fullmatch(str(key))
                    else walk(value)
                )
                for key, value in item.items()
            }
        if isinstance(item, list):
            return [walk(entry) for entry in item]
        if isinstance(item, str):
            if item in exact:
                return REDACTED
            for secret in substrings:
                if secret in item:
                    item = item.replace(secret, REDACTED)
            return item
        return item

    return walk(node)


# -- workspace read-only access (P5) --------------------------------------------


def workspace_listing(run_id: str, run_dir: Path) -> dict:
    """Top-level workspace entries: dirs with their direct files, plus files."""
    _require_run_dir(run_id, run_dir)
    root = run_dir / "workspace"
    if not root.is_dir():
        raise HttpError(404, f"run {run_id} has no workspace directory")
    entries = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            continue  # never advertise links: the file endpoint rejects them anyway
        if child.is_dir():
            files = sorted(item.name for item in child.iterdir() if item.is_file())
            entries.append({"path": child.name, "type": "dir", "files": files})
        elif child.is_file():
            entries.append({"path": child.name, "type": "file", "size": child.stat().st_size})
    return {"run_id": run_id, "entries": entries}


def workspace_file(run_id: str, run_dir: Path, rel_path: str) -> dict:
    """One text file under the run's workspace, read-only.

    Defenses, in order: the path must be relative and contain no ``..``
    component after normalization; the resolved target (symlinks followed)
    must stay inside the resolved workspace; only whitelisted text extensions
    are served, up to WORKSPACE_MAX_BYTES, with a NUL-byte binary check."""
    _require_run_dir(run_id, run_dir)
    root = run_dir / "workspace"
    if not root.is_dir():
        raise HttpError(404, f"run {run_id} has no workspace directory")
    raw = (rel_path or "").strip()
    if not raw:
        raise HttpError(400, "missing required query parameter: path")
    norm = posixpath.normpath(raw)
    if posixpath.isabs(norm) or norm.startswith("\\") or (len(norm) > 1 and norm[1] == ":"):
        raise HttpError(400, f"absolute paths are not allowed: {raw!r}")
    parts = norm.split("/")
    if any(part == ".." for part in parts):
        raise HttpError(400, f"path traversal is not allowed: {raw!r}")
    if norm == ".":
        raise HttpError(400, f"not a file path: {raw!r}")
    resolved_root = root.resolve()
    target = (root / norm).resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise HttpError(400, f"path escapes the workspace: {raw!r}")
    if not target.exists():
        raise HttpError(404, f"no such workspace file: {norm}")
    if not target.is_file():
        raise HttpError(400, f"not a regular file: {norm}")
    suffix = target.suffix.lower()
    if suffix not in WORKSPACE_TEXT_EXTENSIONS:
        raise HttpError(415, f"unsupported file type {suffix!r}: only whitelisted text files")
    size = target.stat().st_size
    if size > WORKSPACE_MAX_BYTES:
        raise HttpError(413, f"file exceeds the {WORKSPACE_MAX_BYTES} byte limit: {norm}")
    data = target.read_bytes()
    if b"\x00" in data[:8192]:
        raise HttpError(415, f"binary content is not served: {norm}")
    content = data.decode("utf-8", errors="replace")
    return {"run_id": run_id, "path": norm, "size": size, "content": content}


# -- run summary timestamp -------------------------------------------------------


def run_started_at(run_dir: Path, snap: dict) -> float | None:
    """Best-effort start time: job.json first, else the journal's first
    ``stage_started`` timestamp. Never invented: unknown stays None."""
    job = snap.get("job") or {}
    ts = job.get("started_at")
    if isinstance(ts, (int, float)):
        return float(ts)
    return _journal_first_stage_ts(run_dir)


def _journal_first_stage_ts(run_dir: Path) -> float | None:
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kind") == "stage_started" and isinstance(entry.get("ts"), (int, float)):
            return float(entry["ts"])
    return None


def _require_run_dir(run_id: str, run_dir: Path) -> None:
    if not Path(run_dir).is_dir():
        raise HttpError(404, f"unknown run {run_id!r}")


def make_handler(app: WebApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "kernelagent-web/0.1"

        def log_message(self, fmt: str, *args) -> None:  # quiet by default
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            split = urlsplit(self.path)
            path = unquote(split.path)
            query = parse_qs(split.query)
            if path in ("/", "/index.html"):
                self._send_index()
            elif path.startswith("/api/"):
                try:
                    self._dispatch_api(path, query)
                except HttpError as exc:
                    self._send_json({"error": exc.message}, exc.status)
            else:
                self._dispatch_static(path)

        def _dispatch_api(self, path: str, query: dict[str, list[str]]) -> None:
            parts = [segment for segment in path.split("/") if segment]
            if parts == ["api", "health"]:
                self._send_json(
                    {
                        "ok": True,
                        "gpu_device": app.gpu_device,
                        "active_run_id": app.active_run_id(),
                    }
                )
                return
            if parts == ["api", "problems"]:
                try:
                    self._send_json(app.problems())
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"error": str(exc)}, 500)
                return
            if parts[:2] != ["api", "runs"]:
                raise HttpError(404, "not found")
            rest = parts[2:]
            if not rest:
                self._send_json({"runs": app.runs()})
                return
            run_id = rest[0]
            if not _RUN_ID_RE.fullmatch(run_id):
                raise HttpError(404, f"unknown run {run_id!r}")
            run_dir = app.runs_root / run_id
            route = rest[1:]
            if not route:
                snap = run_snapshot(run_dir)
                if snap is None:
                    raise HttpError(404, f"unknown run {run_id!r}")
                self._send_json(snap)
            elif route == ["report"]:
                self._send_json(run_report(run_id, run_dir))
            elif route == ["journal"]:
                self._send_json(
                    run_journal_tail(run_id, run_dir, _int_arg(query, "after", default=0))
                )
            elif route == ["records"]:
                self._send_json(run_records_list(run_id, run_dir))
            elif len(route) == 2 and route[0] == "records":
                self._send_json(run_record(run_id, run_dir, route[1]))
            elif len(route) == 3 and route[0] == "records" and route[2] == "progress":
                self._send_json(run_record_progress(run_id, run_dir, route[1]))
            elif route == ["workspace"]:
                self._send_json(workspace_listing(run_id, run_dir))
            elif route == ["workspace", "file"]:
                self._send_json(workspace_file(run_id, run_dir, _query_arg(query, "path")))
            else:
                raise HttpError(404, "not found")

        # -- SPA static hosting (P1) --------------------------------------

        def _dispatch_static(self, path: str) -> None:
            dist = frontend_dist_dir()
            if dist is None:
                # Legacy behaviour: only "/" serves the single-file console.
                self._send_json({"error": "not found"}, 404)
                return
            rel = path.lstrip("/")
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts:
                self._send_json({"error": f"invalid path {path!r}"}, 404)
                return
            if rel and self._try_send_dist_file(dist, rel):
                return
            if rel.startswith("assets/"):
                # Asset references are never SPA routes: a miss is a hard 404.
                self._send_json({"error": f"no such asset: {path}"}, 404)
                return
            self._send_dist_index(dist)

        def _send_index(self) -> None:
            dist = frontend_dist_dir()
            if dist is not None:
                self._send_dist_index(dist)
            else:
                self._send_static()

        def _send_dist_index(self, dist: Path) -> None:
            body = (dist / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _try_send_dist_file(self, dist: Path, rel: str) -> bool:
            """Serve one whitelisted dist file; False means "not an asset"."""
            resolved_root = dist.resolve()
            target = (dist / rel).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                return False
            if not target.is_file():
                return False
            mime = ASSET_MIME.get(target.suffix.lower())
            if mime is None:
                return False
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(body)
            return True

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path not in ("/api/runs", "/api/models"):
                self._send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                self._send_json({"error": f"invalid JSON body: {exc}"}, 400)
                return
            if path == "/api/models":
                self._send_models(payload)
                return
            try:
                self._send_json(app.start_run(payload))
            except ActiveRunError as exc:
                self._send_json({"error": str(exc)}, 409)
            except OptimizationConfigError as exc:
                self._send_json({"error": str(exc)}, 400)
            except (TypeError, ValueError) as exc:
                self._send_json({"error": f"invalid request: {exc}"}, 400)

        def _send_models(self, payload: dict) -> None:
            """List the models a key can use. The key lives in memory for
            this one upstream call - it is never written anywhere."""
            base_url = str(payload.get("base_url") or "")
            api_key = str(payload.get("api_key") or "") or os.environ.get(API_KEY_ENV, "")
            try:
                models = list_models(base_url, api_key)
            except PermanentModelError as exc:
                self._send_json({"error": str(exc), "models": []}, 400)
                return
            except TransientModelError as exc:
                self._send_json({"error": f"provider unreachable: {exc}", "models": []}, 502)
                return
            self._send_json({"models": models})

        def _send_static(self) -> None:
            index = STATIC_DIR / "index.html"
            body = index.read_bytes() if index.is_file() else b"console missing"
            self.send_response(200 if index.is_file() else 500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8501,
    runs_root: Path = Path("artifacts/webui"),
    snapshot_root: Path = Path("research/sources/ScalingIntelligence__KernelBench"),
    fixtures_root: Path = Path("configs/kernelbench/eval-fixtures"),
) -> None:
    app = WebApp(runs_root, snapshot_root, fixtures_root)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"kernelagent console: http://{host}:{port} (runs: {app.runs_root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
