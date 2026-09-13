"""HTTP server for the local web console (standard library only).

Endpoints:
- ``GET /``                 the single-page console
- ``GET /api/health``       liveness + active run
- ``GET /api/problems``     bench/level/problem tree from the pinned snapshot
- ``GET /api/runs``         run history (durable state per run directory)
- ``GET /api/runs/<id>``    one run snapshot (state/budget/stages/speedups)
- ``POST /api/runs``        start a run: {mode, problem spec, model,
  base_url, api_key, budgets}; the key stays in memory for that job only.

One GPU means one active run; a second start is a 409. Every durable fact
shown in the UI comes from the run directory (journal, records, report) -
the server never invents status."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
_JOB_FIELDS_DEFAULTS = {
    "max_candidates": 5,
    "max_repair_rounds": 2,
    "gpu_budget_seconds": 1800.0,
    "token_budget": 200000,
}


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
                args=(mode, config, api_key, run_id),
                daemon=True,
                name=f"optimize-{run_id}",
            )
            self._threads[run_id] = thread
            thread.start()
        return {"run_id": run_id, "state": "running"}

    def _run_job(self, mode: str, config: OptimizationConfig, api_key: str, run_id: str) -> None:
        try:
            generator = build_generator(
                mode, config.base_url, api_key, fixtures_root=self.fixtures_root
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
                    "mode": snap.get("job", {}).get("mode"),
                    "problem": snap.get("job", {}).get("config", {}).get("problem"),
                    "champion": snap.get("champion", {}).get("candidate_sha256"),
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
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_static()
            elif path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "gpu_device": app.gpu_device,
                        "active_run_id": app.active_run_id(),
                    }
                )
            elif path == "/api/problems":
                try:
                    self._send_json(app.problems())
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"error": str(exc)}, 500)
            elif path == "/api/runs":
                self._send_json({"runs": app.runs()})
            elif path.startswith("/api/runs/"):
                run_id = path.rsplit("/", 1)[-1]
                snap = run_snapshot(app.runs_root / run_id)
                if snap is None:
                    self._send_json({"error": f"unknown run {run_id!r}"}, 404)
                else:
                    self._send_json(snap)
            else:
                self._send_json({"error": "not found"}, 404)

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
