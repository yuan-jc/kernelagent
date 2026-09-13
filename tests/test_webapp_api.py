"""HTTP-level tests for the webapp API (frontend spec proposals P1-P5).

Every test drives a real ThreadingHTTPServer on an ephemeral port, so the
routing, SPA fallback, error bodies and traversal defences are exercised
exactly as a browser would see them. CPU-only: run directories are seeded
fixtures, never live GPU jobs."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from kernelagent.orchestrator import Journal
from kernelagent.webapp.server import WebApp, make_handler

SECRET = "sk-test-1234567890abcdef"
RUN_ID = "20260914-010000"


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    level1 = root / "KernelBench" / "level1"
    level1.mkdir(parents=True)
    (level1 / "40_LayerNorm.py").write_text("# p\n", encoding="utf-8")
    return root


def _fixtures(tmp_path: Path) -> Path:
    fx = tmp_path / "fx"
    fx.mkdir(exist_ok=True)
    (fx / "triton_l1_p40_layernorm.py").write_text("GOOD\n", encoding="utf-8")
    (fx / "wrong_l1_p40_layernorm.py").write_text("BAD\n", encoding="utf-8")
    return fx


class _Server:
    """A live handler over one WebApp; tests seed run dirs before GETs."""

    def __init__(self, runs_root: Path, snapshot_root: Path, fixtures_root: Path):
        self.runs_root = runs_root
        self.app = WebApp(runs_root, snapshot_root, fixtures_root=fixtures_root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def get(self, path: str) -> tuple[int, dict, bytes]:
        request = urllib.request.Request(self.base + path)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:  # 4xx/5xx bodies still matter
            return exc.code, dict(exc.headers), exc.read()

    def get_json(self, path: str) -> tuple[int, dict]:
        status, _, body = self.get(path)
        return status, json.loads(body)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Default to "no dist" so P2-P5 tests run in legacy static mode; the P1
    # tests override KA_FRONTEND_DIST themselves.
    monkeypatch.setenv("KA_FRONTEND_DIST", str(tmp_path / "no-dist-built"))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    yield server
    server.stop()


def _seed_run(runs: Path, run_id: str = RUN_ID, *, started_at: float | None = 1789400000.0) -> Path:
    """One run dir with a 6-line journal, records, report, job and workspace."""
    run_dir = runs / run_id
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "workspace" / "container-deadbeef").mkdir(parents=True)
    journal = Journal(run_dir / "journal.jsonl")
    journal.append("budget_reserved", action_id="candidate-000", gpu_seconds=300.0, tokens=2048)
    journal.append(
        "action_started", action_id="candidate-000", input_hash="h1", budget_reservation="r1"
    )
    journal.append("budget_settled", action_id="candidate-000", gpu_seconds=300.0, tokens=30)
    journal.append("action_finished", action_id="candidate-000", input_hash="h1", result_ref="x")
    # New event kinds from the RV01 pipeline stages must pass through verbatim.
    journal.append("stage_started", action_id="candidate-001", stage="timing", ts=1789400123.5)
    journal.append(
        "manifest_revised",
        field="token_budget",
        from_value=200000,
        to_value=300000,
        reason="explicit resume override",
    )
    (run_dir / "records" / "candidate-000.json").write_text(
        json.dumps({"candidate": "candidate-000", "stage": "timing", "status": "retained"}),
        encoding="utf-8",
    )
    (run_dir / "records" / "candidate-000.progress.json").write_text(
        json.dumps({"candidate": "candidate-000", "stage": "timing", "ts": 1789400100.0}),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps({"state": "completed", "champion": {"candidate_sha256": "cafe1234"}}),
        encoding="utf-8",
    )
    job: dict = {"run_id": run_id, "mode": "live"}
    if started_at is not None:
        job["started_at"] = started_at
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (run_dir / "workspace" / "container-deadbeef" / "stdout.log").write_text(
        "hello\n", encoding="utf-8"
    )
    (run_dir / "workspace" / "container-deadbeef" / "candidate.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    (run_dir / "workspace" / "notes.md").write_text("# notes\n", encoding="utf-8")
    return run_dir


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
    (dist / "assets" / "app-a1b2c3.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "assets" / "style-d4e5.css").write_text("body{}", encoding="utf-8")
    (dist / "favicon.ico").write_bytes(b"ico")
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    return dist


# --- P1: SPA static hosting and history-route fallback -----------------------


def test_spa_root_and_fallback_serve_dist_index(tmp_path, monkeypatch):
    dist = _make_dist(tmp_path)
    monkeypatch.setenv("KA_FRONTEND_DIST", str(dist))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    try:
        for route in ("/", "/index.html", "/runs/20260914-010000", "/benchmarks"):
            status, headers, body = server.get(route)
            assert status == 200, route
            assert body == b"<html>SPA</html>", route
            assert headers["Content-Type"].startswith("text/html")
            assert headers["Cache-Control"] == "no-cache"
    finally:
        server.stop()


def test_spa_assets_served_with_mime_and_immutable_cache(tmp_path, monkeypatch):
    dist = _make_dist(tmp_path)
    monkeypatch.setenv("KA_FRONTEND_DIST", str(dist))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    try:
        status, headers, body = server.get("/assets/app-a1b2c3.js")
        assert status == 200
        assert body == b"console.log(1)"
        assert headers["Content-Type"].startswith("text/javascript")
        assert "max-age=31536000" in headers["Cache-Control"]
        status, headers, _ = server.get("/favicon.ico")
        assert status == 200
        assert headers["Content-Type"] == "image/x-icon"
    finally:
        server.stop()


def test_spa_missing_asset_is_404_not_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("KA_FRONTEND_DIST", str(_make_dist(tmp_path)))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    try:
        status, _, body = server.get("/assets/nope-000.js")
        assert status == 404
        assert b"no such asset" in body
    finally:
        server.stop()


def test_spa_mime_whitelist_rejects_unknown_extension(tmp_path, monkeypatch):
    dist = _make_dist(tmp_path)
    (dist / "assets" / "payload.exe").write_bytes(b"MZ")
    (dist / "payload.exe").write_bytes(b"MZ")
    monkeypatch.setenv("KA_FRONTEND_DIST", str(dist))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    try:
        # Under /assets/ a non-whitelisted extension is a hard 404.
        status, _, body = server.get("/assets/payload.exe")
        assert status == 404
        assert b"MZ" not in body
        # Outside /assets/ it is treated as an SPA route (index fallback):
        # the exe bytes are never served with any content type.
        status, headers, body = server.get("/payload.exe")
        assert status == 200
        assert body == b"<html>SPA</html>"
        assert headers["Content-Type"].startswith("text/html")
    finally:
        server.stop()


@pytest.mark.parametrize(
    "route",
    [
        "/assets/..%2F..%2Fsecret.txt",  # encoded .. segments inside assets
        "/assets/%2e%2e/secret.txt",
        "/../secret.txt",
        "/..%2fsecret.txt",
    ],
)
def test_spa_traversal_attacks_rejected(tmp_path, monkeypatch, route):
    monkeypatch.setenv("KA_FRONTEND_DIST", str(_make_dist(tmp_path)))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    try:
        status, _, body = server.get(route)
        assert status in (400, 404)
        assert SECRET.encode() not in body
    finally:
        server.stop()


def test_legacy_static_page_kept_when_no_dist(tmp_path, monkeypatch):
    monkeypatch.setenv("KA_FRONTEND_DIST", str(tmp_path / "still-not-built"))
    server = _Server(tmp_path / "runs", _snapshot(tmp_path), _fixtures(tmp_path))
    legacy = Path(__file__).resolve().parents[1] / "src/kernelagent/webapp/static/index.html"
    try:
        status, _, body = server.get("/")
        assert status == 200
        assert body == legacy.read_bytes()
        # Without a dist there is no SPA fallback: unknown routes stay 404 JSON.
        status, _, body = server.get("/runs/whatever")
        assert status == 404
        assert json.loads(body)["error"] == "not found"
    finally:
        server.stop()


# --- P2: run report -----------------------------------------------------------


def test_report_returns_report_json_verbatim(api):
    run_dir = _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/report")
    assert status == 200
    assert body == json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def test_report_missing_is_404_with_clear_error(api):
    run_dir = _seed_run(api.runs_root)
    (run_dir / "report.json").unlink()
    status, body = api.get_json(f"/api/runs/{RUN_ID}/report")
    assert status == 404
    assert "report.json" in body["error"]


def test_report_corrupt_is_500_not_a_fake_report(api):
    run_dir = _seed_run(api.runs_root)
    (run_dir / "report.json").write_text("{not json", encoding="utf-8")
    status, body = api.get_json(f"/api/runs/{RUN_ID}/report")
    assert status == 500
    assert "not valid JSON" in body["error"]


def test_report_unknown_run_is_404(api):
    status, body = api.get_json("/api/runs/19990101-000000/report")
    assert status == 404
    assert "unknown run" in body["error"]


# --- P3: journal incremental reads ---------------------------------------------


def test_journal_cursor_semantics(api):
    run_dir = _seed_run(api.runs_root)

    status, full = api.get_json(f"/api/runs/{RUN_ID}/journal?after=0")
    assert status == 200
    assert full["run_id"] == RUN_ID
    assert [entry["seq"] for entry in full["entries"]] == [1, 2, 3, 4, 5, 6]
    assert full["next_after"] == 6

    status, tail = api.get_json(f"/api/runs/{RUN_ID}/journal?after=2")
    assert status == 200
    assert [entry["seq"] for entry in tail["entries"]] == [3, 4, 5, 6]
    assert tail["next_after"] == 6

    # Quiescent poll: cursor at end yields nothing and stays put.
    status, quiet = api.get_json(f"/api/runs/{RUN_ID}/journal?after=6")
    assert status == 200
    assert quiet["entries"] == []
    assert quiet["next_after"] == 6

    # New events become visible exactly once, after the stored cursor.
    Journal(run_dir / "journal.jsonl").append(
        "stage_started", action_id="candidate-002", stage="evaluate", ts=1789400456.0
    )
    status, delta = api.get_json(f"/api/runs/{RUN_ID}/journal?after=6")
    assert status == 200
    assert [entry["kind"] for entry in delta["entries"]] == ["stage_started"]
    assert delta["entries"][0]["seq"] == 7
    assert delta["next_after"] == 7

    # Cursor past the end (e.g. a restarted run with a fresh journal) is safe.
    status, past = api.get_json(f"/api/runs/{RUN_ID}/journal?after=999")
    assert status == 200
    assert past["entries"] == []
    assert past["next_after"] == 7


def test_journal_passes_new_event_kinds_through_verbatim(api):
    _seed_run(api.runs_root)
    _, full = api.get_json(f"/api/runs/{RUN_ID}/journal?after=0")
    stage = next(entry for entry in full["entries"] if entry["kind"] == "stage_started")
    assert stage["action_id"] == "candidate-001"
    assert stage["stage"] == "timing"
    assert stage["ts"] == 1789400123.5
    assert "entry_hash" in stage and "prev_hash" in stage
    revised = next(entry for entry in full["entries"] if entry["kind"] == "manifest_revised")
    assert revised["field"] == "token_budget"
    assert revised["to_value"] == 300000


def test_journal_torn_tail_line_is_withheld(api):
    run_dir = _seed_run(api.runs_root)
    # Simulate a torn append: a final line that is not valid JSON yet.
    with (run_dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "stage_started", "sta')
    status, body = api.get_json(f"/api/runs/{RUN_ID}/journal?after=6")
    assert status == 200
    assert body["entries"] == []
    assert body["next_after"] == 6
    # Once the append completes, the line is served.
    with (run_dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('ge": "confirm", "ts": 1.0}\n')
    status, body = api.get_json(f"/api/runs/{RUN_ID}/journal?after=6")
    assert status == 200
    assert body["entries"][0]["kind"] == "stage_started"
    assert body["next_after"] == 7


def test_journal_rejects_bad_cursor_and_unknown_run(api):
    _seed_run(api.runs_root)
    for cursor in ("after=abc", "after=-1"):
        status, body = api.get_json(f"/api/runs/{RUN_ID}/journal?{cursor}")
        assert status == 400, cursor
        assert "error" in body
    status, body = api.get_json("/api/runs/19990101-000000/journal?after=0")
    assert status == 404
    assert "unknown run" in body["error"]


# --- P4: records (read-only, secret-scrubbed) -----------------------------------


def test_records_listing_lists_final_and_progress(api):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/records")
    assert status == 200
    by_key = {(r["record"], r["variant"]): r for r in body["records"]}
    assert by_key[("candidate-000", "final")]["file"] == "candidate-000.json"
    assert by_key[("candidate-000", "final")]["size"] > 0
    assert by_key[("candidate-000", "progress")]["file"] == "candidate-000.progress.json"


def test_record_single_and_progress(api):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/records/candidate-000")
    assert status == 200
    assert body["status"] == "retained"
    status, body = api.get_json(f"/api/runs/{RUN_ID}/records/candidate-000/progress")
    assert status == 200
    assert body["stage"] == "timing"


def test_record_missing_and_invalid_ids(api):
    _seed_run(api.runs_root)
    status, _ = api.get_json(f"/api/runs/{RUN_ID}/records/ghost")
    assert status == 404
    # Encoded traversal in the action id must never reach the filesystem.
    status, body = api.get_json(f"/api/runs/{RUN_ID}/records/..%2F..%2Fjob")
    assert status == 404
    assert "config" not in body  # job.json content is not echoed
    # A lone ".." segment is rejected as an invalid id, not resolved.
    status, body = api.get_json(f"/api/runs/{RUN_ID}/records/%2E%2E")
    assert status == 404
    assert "invalid record id" in body["error"]


def test_record_scrubs_environment_secrets(api, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER_API_KEY", SECRET)
    run_dir = _seed_run(api.runs_root)
    (run_dir / "records" / "candidate-000.json").write_text(
        json.dumps(
            {
                "candidate": "candidate-000",
                "config": {"api_key": SECRET},
                "note": f"provider replied 401 for Bearer {SECRET}",
                "exactly": SECRET,
                "cwd_echo": str(api.runs_root),
                "public": "unchanged",
            }
        ),
        encoding="utf-8",
    )
    status, _, raw = api.get(f"/api/runs/{RUN_ID}/records/candidate-000")
    assert status == 200
    text = raw.decode("utf-8")
    assert SECRET not in text, "the environment secret must never leave the server"
    body = json.loads(text)
    assert body["config"]["api_key"] == "[REDACTED]"
    assert body["note"] == "provider replied 401 for Bearer [REDACTED]"
    assert body["exactly"] == "[REDACTED]"
    assert body["public"] == "unchanged"


# --- P5: workspace read-only access ----------------------------------------------


def test_workspace_listing(api):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace")
    assert status == 200
    entries = {entry["path"]: entry for entry in body["entries"]}
    assert entries["container-deadbeef"]["type"] == "dir"
    assert entries["container-deadbeef"]["files"] == ["candidate.py", "stdout.log"]
    assert entries["notes.md"]["type"] == "file"
    assert entries["notes.md"]["size"] == len("# notes\n")


@pytest.mark.parametrize(
    "query,expected",
    [
        ("container-deadbeef/stdout.log", "hello\n"),
        ("container-deadbeef/sub/../stdout.log", "hello\n"),  # inner .. normalizes inside
        ("notes.md", "# notes\n"),
    ],
)
def test_workspace_file_serves_text(api, query, expected):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path={query}")
    assert status == 200
    assert body["content"] == expected
    assert body["size"] == len(expected.encode())
    assert ".." not in body["path"]


@pytest.mark.parametrize(
    "query",
    [
        "../../job.json",
        "..%2F..%2Fjob.json",
        "container-deadbeef/../../../etc/passwd",
    ],
)
def test_workspace_traversal_rejected(api, query):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path={query}")
    assert status == 400
    assert "traversal" in body["error"]


def test_workspace_absolute_path_rejected(api):
    _seed_run(api.runs_root)
    for query in ("/etc/passwd", "%2Fetc%2Fpasswd"):
        status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path={query}")
        assert status == 400, query
        assert "absolute" in body["error"]


def test_workspace_symlink_escape_rejected(api, tmp_path):
    run_dir = _seed_run(api.runs_root)
    workspace = run_dir / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text(SECRET, encoding="utf-8")
    (workspace / "innocent.log").symlink_to(outside)
    (workspace / "dirlink").symlink_to(tmp_path)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=innocent.log")
    assert status == 400
    assert "escapes" in body["error"]
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=dirlink/outside.txt")
    assert status == 400
    assert "escapes" in body["error"]
    # Symlinks are not advertised in the listing either.
    _, body = api.get_json(f"/api/runs/{RUN_ID}/workspace")
    assert "innocent.log" not in [e["path"] for e in body["entries"]]


def test_workspace_binary_and_unknown_extension_rejected(api):
    run_dir = _seed_run(api.runs_root)
    workspace = run_dir / "workspace"
    (workspace / "blob.json").write_bytes(b'{"a": "\x00\x01"}')
    (workspace / "payload.bin").write_bytes(b"\x00\x01binary")
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=blob.json")
    assert status == 415
    assert "binary" in body["error"]
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=payload.bin")
    assert status == 415
    assert "unsupported file type" in body["error"]


def test_workspace_size_limit(api):
    run_dir = _seed_run(api.runs_root)
    big = run_dir / "workspace" / "huge.log"
    big.write_text("x" * (256 * 1024 + 1), encoding="utf-8")
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=huge.log")
    assert status == 413
    assert "limit" in body["error"]


def test_workspace_missing_targets(api):
    _seed_run(api.runs_root)
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=nope.txt")
    assert status == 404
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=container-deadbeef")
    assert status == 400
    assert "not a regular file" in body["error"]
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file")
    assert status == 400
    assert "path" in body["error"]


def test_workspace_without_directory_is_404(api):
    run_dir = _seed_run(api.runs_root)
    (run_dir / "workspace").rename(run_dir / "workspace-bak")
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace")
    assert status == 404
    status, body = api.get_json(f"/api/runs/{RUN_ID}/workspace/file?path=notes.md")
    assert status == 404


# --- runs summary: started_at + snapshot routing ---------------------------------


def test_runs_summary_started_at_from_job(api):
    _seed_run(api.runs_root)
    status, body = api.get_json("/api/runs")
    assert status == 200
    summary = next(run for run in body["runs"] if run["run_id"] == RUN_ID)
    assert summary["started_at"] == 1789400000.0


def test_runs_summary_started_at_falls_back_to_journal(api):
    run_dir = _seed_run(api.runs_root, started_at=None)
    (run_dir / "job.json").unlink()
    status, body = api.get_json("/api/runs")
    summary = next(run for run in body["runs"] if run["run_id"] == RUN_ID)
    # First stage_started ts in the seeded journal.
    assert summary["started_at"] == 1789400123.5


def test_runs_summary_started_at_null_when_unknown(api):
    run_dir = _seed_run(api.runs_root, started_at=None)
    (run_dir / "job.json").unlink()
    (run_dir / "journal.jsonl").write_text("", encoding="utf-8")
    status, body = api.get_json("/api/runs")
    summary = next(run for run in body["runs"] if run["run_id"] == RUN_ID)
    assert summary["started_at"] is None


def test_snapshot_unknown_run_is_404_json(api):
    status, body = api.get_json("/api/runs/19990101-000000")
    assert status == 404
    assert "unknown run" in body["error"]
