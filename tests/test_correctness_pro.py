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


# --- Launch plan Task 6 (U1 blocker): triton candidates must be staged
# with their backend and loaded from a real file (tempfile loader), because
# @triton.jit cannot take its source from exec()-style strings. The
# integration gap was exposed by the first real-GPU optimization loop run
# (pro 'baseline' check failed for the known-good triton fixture). ---


def test_run_pro_case_stages_backend(tmp_path, monkeypatch):
    from pathlib import Path

    from kernelagent.adapters.evals.correctness_pro import run_pro_case
    from kernelagent.worker import WorkerOutcome

    snapshot = Path(tmp_path) / "snapshot"
    (snapshot / "src" / "kernelbench").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (snapshot / "src" / "kernelbench" / harness).write_text("# stub\n", encoding="utf-8")
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text("# p\n")
    captured = {}

    def fake_execute(request, spec, *, docker_command):
        case_json = Path(spec.read_only_mounts[0][1]) / "case.json"
        import json as _json

        captured["backend"] = _json.loads(case_json.read_text(encoding="utf-8"))["backend"]
        return WorkerOutcome(
            request_id="pro-x",
            status="completed",
            exit_code=0,
            duration_seconds=0.1,
            stdout_tail="",
            stderr_tail="",
            killed=False,
            workdir=str(tmp_path / "wd"),
        )

    monkeypatch.setattr(
        "kernelagent.adapters.evals.correctness_pro.execute_container", fake_execute
    )
    (Path(tmp_path) / "wd" / "out").mkdir(parents=True)
    run_pro_case(
        case_id="backend-staging",
        level=1,
        problem_id=40,
        problem_name="40_LayerNorm.py",
        candidate_name="c",
        candidate_source="class ModelNew:\n    pass\n",
        snapshot_root=snapshot,
        workspace_root=Path(tmp_path) / "ws",
        gpu_devices=("nvidia.com/gpu=GPU-fake",),
        backend="triton",
    )
    assert captured["backend"] == "triton"


def test_pro_driver_loads_triton_candidates_from_real_file():
    """The driver must use the tempfile loader for triton-family backends:
    @triton.jit needs real source files, exec-strings break it."""
    import kernelagent.adapters.evals.correctness_pro as module

    driver = module._PRO_DRIVER_PATH.read_text(encoding="utf-8")
    assert "load_custom_model_with_tempfile" in driver
    assert 'case.get("backend"' in driver
