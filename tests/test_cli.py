"""CLI surface: the optimize/resume/status product entry (launch plan
Task 5). Offline only: config errors and exit-code mapping; real GPU runs
go through examples/alpha_run.py (Task 6)."""

import json

import pytest

from kernelagent.cli import main


def test_cli_help_lists_product_commands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("optimize", "resume", "status"):
        assert command in out


def test_optimize_without_base_url_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER_API_KEY", raising=False)
    code = main(
        [
            "optimize",
            "--problem",
            "kernelbench:l1:40",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert code == 3


def test_optimize_without_credentials_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER_API_KEY", raising=False)
    code = main(
        [
            "optimize",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert code == 3, "missing credentials must be a user-fixable config error, not a crash"


def test_status_of_missing_run_is_a_config_error(tmp_path):
    code = main(["status", "--output", str(tmp_path / "does-not-exist")])
    assert code == 3


def test_resume_without_a_run_is_a_config_error(tmp_path):
    code = main(
        [
            "resume",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--output",
            str(tmp_path / "never-started"),
        ]
    )
    assert code == 3


# --- RV01: manifest-driven resume semantics --------------------------------


def _seed_completed_run(tmp_path):
    """Finish one real offline loop (injected ports) with a NON-default
    problem/model/budget so restoring defaults is observable."""
    from test_optimization_loop import (
        GOOD_CANDIDATE,
        RecordingLedger,
        ScriptedResponder,
        _fast_candidate_timing,
        _passing_evaluate,
        _stage_problem_with_extra,
    )

    from kernelagent.adapters.models.generation import CandidateGenerator
    from kernelagent.optimization import OptimizationConfig, optimize

    snapshot = _stage_problem_with_extra(tmp_path, (41,))
    responder = ScriptedResponder([GOOD_CANDIDATE])
    generator = CandidateGenerator(responder, ledger=RecordingLedger())
    config = OptimizationConfig(
        problem="kernelbench:l1:41",
        backend="triton",
        model_id="original-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=1,
        max_repair_rounds=0,
        # Non-default budget, large enough for the RV04 stage-timeout
        # floors (baseline 1200 + evaluate 900 + timing 1200).
        gpu_budget_seconds=41234.5,
        token_budget=100000,
        output=tmp_path / "run",
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    result = optimize(
        config,
        generator=generator,
        evaluate=_passing_evaluate,
        timing=_fast_candidate_timing,
        correctness_pro=None,
    )
    assert result.state == "completed"
    return result


def test_resume_with_only_output_and_base_url_restores_original_config(tmp_path, monkeypatch):
    """`resume --output X --base-url Y` (nothing else) must restore the
    original problem/model/budgets from the run manifest, skip all finished
    actions, and never rewrite the manifest."""
    monkeypatch.setenv("MODEL_PROVIDER_API_KEY", "cli-offline-dummy")
    result1 = _seed_completed_run(tmp_path)
    report1 = json.loads(result1.report_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest_before = manifest_path.read_text(encoding="utf-8")

    code = main(
        [
            "resume",
            "--output",
            str(tmp_path / "run"),
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )
    assert code == 0
    report2 = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert report2["config"]["problem"] == "kernelbench:l1:41", (
        "resume must not fall back to the CLI default problem"
    )
    assert report2["config"]["model_id"] == "original-model"
    assert report2["budget"]["gpu_seconds_limit"] == 41234.5
    assert report2["champion"]["candidate_sha256"] == report1["champion"]["candidate_sha256"]
    assert report2["budget"]["settled_tokens"] == report1["budget"]["settled_tokens"], (
        "identical-identity resume must not re-execute or re-bill finished actions"
    )
    assert manifest_path.read_text(encoding="utf-8") == manifest_before


def test_resume_with_explicit_identity_override_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER_API_KEY", "cli-offline-dummy")
    _seed_completed_run(tmp_path)
    code = main(
        [
            "resume",
            "--output",
            str(tmp_path / "run"),
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--gpu-device",
            "nvidia.com/gpu=GPU-swapped",
        ]
    )
    assert code == 3, "an identity override on resume must refuse with exit 3"


def test_resume_with_invalid_override_exits_3(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER_API_KEY", "cli-offline-dummy")
    _seed_completed_run(tmp_path)
    code = main(
        [
            "resume",
            "--output",
            str(tmp_path / "run"),
            "--max-candidates",
            "0",
        ]
    )
    assert code == 3


def test_resume_of_run_without_manifest_is_config_error(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "journal.jsonl").write_text("", encoding="utf-8")
    code = main(
        [
            "resume",
            "--output",
            str(run),
            "--base-url",
            "http://127.0.0.1:9/v1",
        ]
    )
    assert code == 3, "a journal without its manifest cannot prove identity"


def test_optimize_into_existing_output_dir_is_config_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER_BASE_URL", "http://127.0.0.1:9/v1")
    run = tmp_path / "run"
    run.mkdir()
    (run / "leftover.txt").write_text("stale", encoding="utf-8")
    code = main(["optimize", "--output", str(run)])
    assert code == 3, "a fresh optimize must refuse an occupied output directory"
