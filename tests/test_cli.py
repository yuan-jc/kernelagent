"""CLI surface: the optimize/resume/status product entry (launch plan
Task 5). Offline only: config errors and exit-code mapping; real GPU runs
go through examples/alpha_run.py (Task 6)."""

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
