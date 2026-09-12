"""Extended correctness track (T06): control-plane verdict semantics.

Pure CPU: the overall verdict must require every core check AND an
explicit sanitizer pass; ``not_run`` is a coverage gap, never a pass;
the upstream evaluator's verdict is passed through untouched."""

from kernelagent.adapters.evals import overall_verdict


def _payload(**overrides) -> dict:
    checks = {
        "baseline": {"passed": True},
        "shape_dtype": {"passed": True},
        "input_mutation": {"passed": True},
        "constant_output": {"passed": True},
        "tail_check": {"passed": True, "tail_ok": True},
        "state_check": {"passed": True},
        "sanitizer": {"status": "pass"},
    }
    checks.update(overrides)
    return {"protocol": "t06-correctness-pro-v1", "checks": checks}


def test_all_green_passes():
    verdict, reason = overall_verdict(_payload())
    assert verdict is True and "all core" in reason


def test_any_core_failure_blocks_overall():
    for name in (
        "baseline",
        "shape_dtype",
        "input_mutation",
        "constant_output",
        "tail_check",
        "state_check",
    ):
        verdict, _ = overall_verdict(_payload(**{name: {"passed": False}}))
        assert verdict is False, f"{name} failure must block the overall verdict"


def test_missing_core_check_blocks_overall():
    payload = _payload()
    del payload["checks"]["tail_check"]
    verdict, reason = overall_verdict(payload)
    assert verdict is False
    assert "tail_check" in reason


def test_sanitizer_fail_is_not_a_pass():
    verdict, _ = overall_verdict(_payload(sanitizer={"status": "fail", "returncode": 9}))
    assert verdict is False


def test_sanitizer_not_run_is_coverage_gap_never_pass():
    verdict, reason = overall_verdict(_payload(sanitizer={"status": "not_run"}))
    assert verdict is False
    assert "not a pass" in reason or "coverage gap" in reason


def test_unknown_sanitizer_status_rejected():
    verdict, _ = overall_verdict(_payload(sanitizer={"status": "skipped"}))
    assert verdict is False


def test_missing_payload_is_never_a_pass():
    for payload in (None, {}, {"checks": "nope"}):
        verdict, _ = overall_verdict(payload)
        assert verdict is False
