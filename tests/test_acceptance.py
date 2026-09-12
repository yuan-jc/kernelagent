"""Exercise real subprocess outcomes, including pytest's all-skipped exit zero."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from kernelagent.acceptance import _counts, run_checks

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/cpu-test-run.schema.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("source", "status", "passed", "skipped"),
    [
        ("def test_ok(): assert 2 + 2 == 4", "PASS", 1, 0),
        ("def test_bad(): assert False", "FAIL", 0, 0),
        (
            "import pytest\n@pytest.mark.skip(reason='no hardware')\ndef test_skip(): pass",
            "FAIL",
            0,
            1,
        ),
        ("# No tests here", "FAIL", 0, 0),
        ("raise RuntimeError('collection failed')", "FAIL", 0, 0),
    ],
)
def test_actual_outcomes(tmp_path, source, status, passed, skipped):
    tests = tmp_path / "test_sample.py"
    tests.write_text(source + "\n", encoding="utf-8")
    report = run_checks(tests, tmp_path / "evidence")
    jsonschema.validate(report, SCHEMA)
    assert report["status"] == status
    assert report["exit_code"] == (0 if status == "PASS" else 1)
    assert report["counts"]["passed"] == passed
    assert report["counts"]["skipped"] == skipped
    if skipped:
        assert report["pytest_exit_code"] == 0
    for artifact in report["artifacts"]:
        raw = (Path(report["report_path"]).parent / artifact["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]


def test_cli_failure_returns_nonzero(tmp_path):
    test = tmp_path / "test_bad.py"
    test.write_text("def test_failure(): assert False\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kernelagent",
            "check",
            "--tests",
            str(test),
            "--output",
            str(tmp_path / "evidence"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_timeout_fails_and_does_not_reuse_prior_report(tmp_path):
    test = tmp_path / "test_sample.py"
    test.write_text("def test_ok(): assert True\n", encoding="utf-8")
    first = run_checks(test, tmp_path / "evidence")
    assert first["status"] == "PASS"
    test.write_text("import time\ndef test_slow(): time.sleep(30)\n", encoding="utf-8")
    second = run_checks(test, tmp_path / "evidence", timeout=0.5)
    jsonschema.validate(second, SCHEMA)
    assert second["status"] == "FAIL"
    assert "timed out" in second["reason"]
    assert first["run_id"] != second["run_id"]


def test_schema_rejects_false_pass():
    schema = jsonschema.Draft202012Validator(SCHEMA)
    schema.check_schema(SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"schema_version": "1.0", "status": "PASS"}, SCHEMA)


def test_counts_reject_unrelated_xml(tmp_path):
    file = tmp_path / "junit.xml"
    file.write_text("<unrelated />", encoding="utf-8")
    with pytest.raises(ValueError):
        _counts(file)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError):
        run_checks(tmp_path, tmp_path / "out", timeout)
