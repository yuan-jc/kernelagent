"""Web console (local UI): CPU-only checks of the durable-state plumbing.

The server must never persist or echo the API key, must derive every
status from the run directory (journal/records/report), and must refuse a
second run while the GPU is busy."""

import json
import time
from pathlib import Path

import pytest

from kernelagent.optimization import OptimizationConfigError
from kernelagent.orchestrator import Journal
from kernelagent.webapp.jobs import build_generator, list_problems
from kernelagent.webapp.server import ActiveRunError, WebApp, run_snapshot


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    level1 = root / "KernelBench" / "level1"
    level1.mkdir(parents=True)
    (level1 / "40_LayerNorm.py").write_text("# p\n", encoding="utf-8")
    (level1 / "1_Square_matrix_multiplication_.py").write_text("# p\n", encoding="utf-8")
    (root / "KernelBench" / "level2").mkdir()
    return root


def _fixtures(tmp_path: Path) -> Path:
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "triton_l1_p40_layernorm.py").write_text("GOOD\n", encoding="utf-8")
    (fx / "wrong_l1_p40_layernorm.py").write_text("BAD\n", encoding="utf-8")
    return fx


def test_list_problems_enumerates_levels_and_specs(tmp_path):
    data = list_problems(_snapshot(tmp_path))
    assert data["bench"] == "kernelbench"
    level1 = next(level for level in data["levels"] if level["level"] == 1)
    specs = [problem["spec"] for problem in level1["problems"]]
    assert "kernelbench:l1:40" in specs
    assert "kernelbench:l1:1" in specs
    assert data["levels"][0]["level"] == 1


def test_build_generator_demo_modes_use_fixtures(tmp_path):
    fx = _fixtures(tmp_path)
    assert build_generator("demo-correct", "http://x", "", fixtures_root=fx) is not None
    assert build_generator("demo-wrong", "http://x", "", fixtures_root=fx) is not None
    with pytest.raises(OptimizationConfigError):
        build_generator("no-such-mode", "http://x", "", fixtures_root=fx)


def test_build_generator_live_requires_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER_API_KEY", raising=False)
    with pytest.raises(OptimizationConfigError):
        build_generator("live", "http://x", "", fixtures_root=_fixtures(tmp_path))


def _seed_run(runs: Path, run_id: str = "20260913-000000", *, in_flight: bool) -> Path:
    run_dir = runs / run_id
    records = run_dir / "records"
    records.mkdir(parents=True)
    journal = Journal(run_dir / "journal.jsonl")
    reservation = journal.append(
        "budget_reserved", action_id="candidate-000", gpu_seconds=300.0, tokens=2048
    )
    journal.append(
        "action_started",
        action_id="candidate-000",
        input_hash="h",
        budget_reservation=reservation["entry_hash"],
    )
    if not in_flight:
        journal.append("budget_settled", action_id="candidate-000", gpu_seconds=300.0, tokens=30)
        journal.append("action_finished", action_id="candidate-000", input_hash="h", result_ref="x")
    (records / "candidate-000.progress.json").write_text(
        json.dumps({"candidate": "candidate-000", "stage": "timing", "ts": 1.0}),
        encoding="utf-8",
    )
    (records / "candidate-000.json").write_text(
        json.dumps(
            {
                "candidate": "candidate-000",
                "stage": "evaluate",
                "status": "failed",
                "detail": "boom",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": "live",
                "config": {
                    "problem": "kernelbench:l1:40",
                    "gpu_budget_seconds": 1800.0,
                    "token_budget": 200000,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_run_snapshot_running_shows_inflight_stage(tmp_path):
    run_dir = _seed_run(tmp_path, in_flight=True)
    snap = run_snapshot(run_dir)
    assert snap["state"] == "running"
    assert snap["in_flight"] == ["candidate-000"]
    assert snap["progress"]["candidate-000"] == "timing"
    assert snap["budget"]["settled_tokens"] == 0
    assert snap["budget"]["tokens_limit"] == 200000


def test_run_snapshot_finished_reads_report(tmp_path):
    run_dir = _seed_run(tmp_path, in_flight=False)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "state": "no_improvement",
                "champion": {"candidate_sha256": None, "path": None},
                "candidates": [
                    {
                        "candidate": "candidate-000",
                        "stage": "evaluate",
                        "status": "failed",
                        "detail": "boom",
                    }
                ],
                "budget": {
                    "gpu_seconds_limit": 1800.0,
                    "tokens_limit": 100,
                    "settled_gpu_seconds": 300.0,
                    "settled_tokens": 30,
                },
                "candidate_trust": "cooperative",
                "adversarially_secure": False,
            }
        ),
        encoding="utf-8",
    )
    snap = run_snapshot(run_dir)
    assert snap["state"] == "no_improvement"
    assert snap["candidates"][0]["status"] == "failed"
    assert snap["budget"]["settled_tokens"] == 30
    assert snap["adversarially_secure"] is False


def test_start_run_strips_api_key_and_reports_failure(tmp_path, monkeypatch):
    import kernelagent.webapp.server as server_module

    monkeypatch.delenv("MODEL_PROVIDER_API_KEY", raising=False)

    def failing_optimize(config, generator=None):
        raise RuntimeError("environment variable MODEL_PROVIDER_API_KEY is not set")

    monkeypatch.setattr(server_module, "optimize", failing_optimize)
    app = WebApp(tmp_path / "runs", _snapshot(tmp_path), fixtures_root=_fixtures(tmp_path))
    result = app.start_run(
        {
            "mode": "live",
            "problem": "kernelbench:l1:40",
            "api_key": "sk-secret",
            "base_url": "http://x",
        }
    )
    run_dir = tmp_path / "runs" / result["run_id"]
    app._threads[result["run_id"]].join(timeout=5)
    job_text = (run_dir / "job.json").read_text(encoding="utf-8")
    assert "sk-secret" not in job_text, "the API key must never be persisted"
    error_text = (run_dir / "error.txt").read_text(encoding="utf-8")
    assert "sk-secret" not in error_text, "the API key must never reach the error channel"
    assert "MODEL_PROVIDER_API_KEY" in error_text


def test_second_start_while_gpu_busy_is_conflicted(tmp_path, monkeypatch):
    import kernelagent.webapp.server as server_module

    app = WebApp(tmp_path / "runs", _snapshot(tmp_path), fixtures_root=_fixtures(tmp_path))

    def slow_optimize(config, generator=None):
        time.sleep(0.5)

    monkeypatch.setattr(server_module, "optimize", slow_optimize)
    first = app.start_run({"mode": "demo-correct", "problem": "kernelbench:l1:40"})
    with pytest.raises(ActiveRunError):
        app.start_run({"mode": "demo-correct", "problem": "kernelbench:l1:40"})
    app._threads[first["run_id"]].join(timeout=5)
    assert app.active_run_id() is None
