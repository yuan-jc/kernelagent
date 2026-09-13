"""Baseline NCU profiling pipeline (B3): evidence production, honest
degradation, loop wiring and CLI.

Everything here runs offline: NCU evidence comes from a fixed raw-CSV
fixture (parsed with the real T15 parser), the profiler is injected into
``optimize()`` as a fake, and no GPU / docker / ncu binary is touched.
Live NCU validation is a separate, NOT_RUN-until-executed GPU smoke."""

import json
import subprocess
from pathlib import Path

import pytest

from kernelagent.adapters.models.client import ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.adapters.profiling import (
    MetricCatalog,
    evidence_view,
    parse_raw_csv,
    pipeline,
)
from kernelagent.adapters.profiling.pipeline import (
    BASELINE_EVIDENCE_NAME,
    ProfileCapture,
    profile_baseline,
    stage_profile_inputs,
    summarize_evidence,
)
from kernelagent.cli import main as cli_main
from kernelagent.generators.catalog import load_default as load_method_catalog
from kernelagent.generators.classify import CORE_TIER_A_METRICS
from kernelagent.generators.features import (
    extract_code_features,
    task_profile_from_problem,
)
from kernelagent.generators.input import GeneratorInput, HardwareFacts
from kernelagent.generators.plan import MethodPlanner
from kernelagent.optimization import (
    _PROFILER_UNSET,
    OptimizationConfig,
    optimize,
)

# ---------------------------------------------------------------------------
# Fixed NCU raw-CSV fixture: one LayerNorm-like launch carrying the full
# Tier-A catalog (row 0 = names, row 1 = units, row 2 = the launch).
# ---------------------------------------------------------------------------

_RAW_CSV_FULL = (
    '"ID","Kernel Name","Grid Size","Block Size","Device","CC",'
    '"gpu__time_duration.sum",'
    '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
    '"dram__throughput.avg.pct_of_peak_sustained_elapsed",'
    '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
    '"launch__grid_size","launch__block_size","launch__registers_per_thread",'
    '"launch__occupancy_limit_registers",'
    '"sm__warps_active.avg.pct_of_peak_sustained_active"\n'
    '"","","","","","","us","%","%","%","blocks","threads","registers","%","%"\n'
    '"0","vectorized_layer_norm_kernel","(4096, 1, 1)","(1024, 1, 1)","0","8.9",'
    '"85.2","71.68","72.5","38.4","4096","1024","12","20","58.6"\n'
)

# Same launch but the three occupancy/throughput metrics were not collected.
_RAW_CSV_PARTIAL = (
    '"ID","Kernel Name","Grid Size","Block Size","Device","CC",'
    '"gpu__time_duration.sum",'
    '"launch__grid_size","launch__block_size","launch__registers_per_thread"\n'
    '"","","","","","","us","blocks","threads","registers"\n'
    '"0","vectorized_layer_norm_kernel","(4096, 1, 1)","(1024, 1, 1)","0","8.9",'
    '"85.2","4096","1024","12"\n'
)

LAYERNORM_TASK = task_profile_from_problem("40_LayerNorm.py", 1, 40)
LAYERNORM_SOURCE = (
    "import torch\nimport torch.nn as nn\n"
    "class Model(nn.Module):\n"
    "    def __init__(self, normalized_shape: tuple):\n"
    "        super().__init__()\n"
    "        self.ln = nn.LayerNorm(normalized_shape)\n"
    "    def forward(self, x):\n"
    "        return self.ln(x)\n"
)


def _fixture_view(csv_text: str) -> dict:
    """Real evidence_view built from the fixed fixture CSV (no GPU)."""
    return evidence_view(parse_raw_csv(csv_text), MetricCatalog.default())


def _planner_input(ncu_view: dict | None) -> GeneratorInput:
    return GeneratorInput(
        task=LAYERNORM_TASK,
        code_features=extract_code_features(LAYERNORM_SOURCE),
        attempts=(),
        ncu_view=ncu_view,
        hardware=HardwareFacts(compute_capability=8.9, sm_count=24),
    )


# ---------------------------------------------------------------------------
# ProfileCapture: controllable capture parameters.
# ---------------------------------------------------------------------------


def test_capture_requires_exactly_one_of_set_or_metrics():
    with pytest.raises(ValueError):
        ProfileCapture(set_name="basic", metrics=("gpu__time_duration.sum",))
    with pytest.raises(ValueError):
        ProfileCapture(set_name=None, metrics=None)


def test_capture_rejects_nonpositive_launch_count_and_timeout():
    with pytest.raises(ValueError):
        ProfileCapture(launch_count=0)
    with pytest.raises(ValueError):
        ProfileCapture(timeout_seconds=0.0)
    with pytest.raises(ValueError):
        ProfileCapture(launch_skip=-1)


def test_ncu_arguments_pin_launch_cap_and_metric_mode():
    set_args = ProfileCapture(
        set_name="basic", metrics=None, launch_count=7, launch_skip=2
    ).ncu_arguments()
    assert "--set" in set_args and "basic" in set_args
    assert "--launch-count" in set_args and "7" in set_args
    assert "--launch-skip" in set_args and "2" in set_args
    assert "--metrics" not in set_args
    metric = "gpu__time_duration.sum"
    metrics_args = ProfileCapture(set_name=None, metrics=(metric,)).ncu_arguments()
    assert metrics_args[metrics_args.index("--metrics") + 1] == metric
    assert "--set" not in metrics_args
    # The launch cap bounds profiling cost: no unbounded kernel-instance runs.
    assert set_args[set_args.index("--launch-count") + 1] == "7"


def test_capture_default_is_the_tier_a_metric_list():
    payload = ProfileCapture().to_dict()
    assert payload["mode"] == "metrics"
    assert payload["set_name"] is None
    assert tuple(payload["metrics"]) == tuple(CORE_TIER_A_METRICS)
    assert "--metrics" in ProfileCapture().ncu_arguments()


def test_capture_to_dict_records_every_knob():
    payload = ProfileCapture(set_name="basic", metrics=None, launch_count=5).to_dict()
    assert payload["mode"] == "set" and payload["set_name"] == "basic"
    assert payload["launch_count"] == 5 and payload["launch_skip"] == 0
    assert payload["timeout_seconds"] == 300.0


# ---------------------------------------------------------------------------
# evidence_view + summary + classification on the fixture (no ncu run).
# ---------------------------------------------------------------------------


def test_fixture_view_covers_full_tier_a_catalog():
    view = _fixture_view(_RAW_CSV_FULL)
    assert view["source"] == "ncu_profile"
    assert view["launch_count"] == 1
    assert view["catalog_metrics_missing_in_all_launches"] == []
    missing = set(CORE_TIER_A_METRICS) - {
        name for launch in view["launches"] for name in launch["metrics"]
    }
    assert not missing, f"fixture must cover the Tier-A set; missing {missing}"


def test_summary_reports_missing_metric_as_none_never_zero():
    partial = summarize_evidence(_fixture_view(_RAW_CSV_PARTIAL))
    assert partial["dram_throughput_pct_max"] is None
    assert partial["sm_throughput_pct_max"] is None
    assert partial["warps_active_pct_max"] is None
    assert partial["gpu_time_us_max"] == 85.2
    assert partial["launch_count"] == 1
    full = summarize_evidence(_fixture_view(_RAW_CSV_FULL))
    assert full["dram_throughput_pct_max"] == 72.5
    assert full["warps_active_pct_max"] == 58.6
    assert full["kernels"] == ["vectorized_layer_norm_kernel"]


def test_classification_upgrades_from_static_only_to_ncu_full():
    from kernelagent.generators.classify import classify_bottleneck

    static = classify_bottleneck(_planner_input(None))
    assert static.coverage == "static_only"

    full = classify_bottleneck(_planner_input(_fixture_view(_RAW_CSV_FULL)))
    assert full.coverage == "ncu_full"
    assert full.label == "memory_bound"  # dram 72.5% >= 60% saturation signal
    assert any("dram__throughput" in signal for signal in full.signals_fired)
    assert full.signals_missing == ()

    partial = classify_bottleneck(_planner_input(_fixture_view(_RAW_CSV_PARTIAL)))
    assert partial.coverage == "ncu_partial"
    assert any("metrics MISSING" in entry for entry in partial.signals_missing)


# ---------------------------------------------------------------------------
# profile_baseline: honest degradation paths (host ncu, timeout, blocker).
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    problem_dir = root / "KernelBench" / "level1"
    problem_dir.mkdir(parents=True)
    (problem_dir / "40_LayerNorm.py").write_text(LAYERNORM_SOURCE, encoding="utf-8")
    helper_dir = root / "src" / "kernelbench"
    helper_dir.mkdir(parents=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (helper_dir / harness).write_text("# pinned helper\n", encoding="utf-8")
    return root


def _profile(tmp_path: Path, snapshot: Path, capture: ProfileCapture) -> dict:
    return profile_baseline(
        profile_dir=tmp_path / "run" / "profile",
        workspace_root=tmp_path / "run" / "workspace",
        snapshot_root=snapshot,
        level=1,
        problem_name="40_LayerNorm.py",
        problem_source=LAYERNORM_SOURCE,
        gpu_device="nvidia.com/gpu=GPU-fake",
        capture=capture,
        problem_sha256="abc123",
    )


def test_missing_host_ncu_degrades_to_not_run_without_files(tmp_path: Path, mini_snapshot: Path):
    capture = ProfileCapture(ncu_bin=str(tmp_path / "no-such-ncu"))
    outcome = _profile(tmp_path, mini_snapshot, capture)
    assert outcome["status"] == "not_run"
    assert outcome["reason"] == "ncu_unavailable_on_host"
    assert outcome["evidence_view"] is None
    assert not (tmp_path / "run" / "profile" / BASELINE_EVIDENCE_NAME).exists()


def test_timeout_is_protected_and_container_cleaned_up(
    tmp_path: Path, mini_snapshot: Path, monkeypatch
):
    monkeypatch.setattr(pipeline, "host_ncu_bin", lambda preferred=None: "/fake/ncu")
    removed = []
    monkeypatch.setattr(
        pipeline, "_force_remove_container", lambda docker, name: removed.append(name)
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1.0)

    monkeypatch.setattr(pipeline.subprocess, "run", raise_timeout)
    outcome = _profile(
        tmp_path, mini_snapshot, ProfileCapture(ncu_bin="/fake/ncu", timeout_seconds=1.0)
    )
    assert outcome["status"] == "not_run"
    assert outcome["reason"] == "timeout"
    assert outcome["gpu_wall_seconds"] is not None
    assert len(removed) == 1 and removed[0].startswith("kernelagent-prof-")
    assert not (tmp_path / "run" / "profile" / BASELINE_EVIDENCE_NAME).exists()


def test_docker_unavailable_degrades_to_not_run(tmp_path: Path, mini_snapshot: Path, monkeypatch):
    monkeypatch.setattr(pipeline, "host_ncu_bin", lambda preferred=None: "/fake/ncu")

    def raise_oserror(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(pipeline.subprocess, "run", raise_oserror)
    outcome = _profile(tmp_path, mini_snapshot, ProfileCapture(ncu_bin="/fake/ncu"))
    assert outcome["status"] == "not_run"
    assert outcome["reason"] == "docker_unavailable"


def test_counter_permission_blocker_is_mapped_not_swallowed(
    tmp_path: Path, mini_snapshot: Path, monkeypatch
):
    monkeypatch.setattr(pipeline, "host_ncu_bin", lambda preferred=None: "/fake/ncu")
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="==ERROR== ERR_NVGPUCTRPERM denied"
    )
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: completed)
    outcome = _profile(tmp_path, mini_snapshot, ProfileCapture(ncu_bin="/fake/ncu"))
    assert outcome["status"] == "not_run"
    assert outcome["reason"] == "gpu_counter_permission_denied"
    assert "ERR_NVGPUCTRPERM" in outcome["stderr_tail"]


def test_staged_inputs_carry_driver_problem_and_helpers(tmp_path: Path, mini_snapshot: Path):
    inputs = stage_profile_inputs(
        tmp_path / "ws",
        snapshot_root=mini_snapshot,
        level=1,
        problem_name="40_LayerNorm.py",
        problem_source=LAYERNORM_SOURCE,
        capture=ProfileCapture(),
    )
    assert (inputs / "src" / "kernelbench" / "eval.py").is_file()
    assert (inputs / "KernelBench" / "level1" / "40_LayerNorm.py").read_text(
        encoding="utf-8"
    ) == LAYERNORM_SOURCE
    case = json.loads((inputs / "case.json").read_text(encoding="utf-8"))
    assert case["warmup_iters"] == 3 and case["measured_iters"] == 5
    driver = (inputs / "profile_driver.py").read_text(encoding="utf-8")
    assert "load_original_model_and_inputs" in driver
    # The driver is parent-authored and content-hashed (identity in journal).
    assert len(pipeline.driver_sha256()) == 64


# ---------------------------------------------------------------------------
# Loop wiring: fake profiler injected into optimize().
# ---------------------------------------------------------------------------

GOOD_CANDIDATE = json.dumps(
    {"code": "class ModelNew:\n    def forward(self, x):\n        return x"}
)


class ScriptedResponder:
    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []

    def complete(self, request):
        content = self._contents.pop(0)
        self.requests.append(request)
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=content,
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20),
        )


class RecordingLedger:
    def __init__(self):
        self._total = 0

    def record(self, response):
        self._total += response.usage.total_tokens

    def total_tokens(self):
        return self._total


def _passing_evaluate(candidate_source: str, candidate_name: str) -> dict:
    passed = "return x" in candidate_source
    return {
        "adapter_pass": passed,
        "compiled": passed,
        "correct": passed,
        "outcome_status": "completed",
        "stderr_tail": "" if passed else "assert torch.allclose failed",
        "upstream_metadata": "" if passed else "max abs diff 1.0",
    }


def _fake_timing(candidate_source: str, role: str) -> dict:
    batches = [10.0] * 12 if role == "eager" else [5.0] * 12
    return {"source_ok": True, "precondition": True, "async_leak": False, "batches": batches}


def _stage_problem(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    problem_dir = snapshot / "KernelBench" / "level1"
    problem_dir.mkdir(parents=True)
    (problem_dir / "40_LayerNorm.py").write_text(LAYERNORM_SOURCE, encoding="utf-8")
    return snapshot


def _config(tmp_path: Path, snapshot: Path, **overrides) -> OptimizationConfig:
    defaults = dict(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=1,
        max_repair_rounds=0,
        gpu_budget_seconds=3600.0,
        token_budget=100000,
        output=tmp_path / "run",
        resume=False,
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _journal_entries(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fake_profiler_outcome(view: dict, wall: float = 42.5) -> dict:
    return {
        "status": "collected",
        "reason": None,
        "detail": "",
        "capture": ProfileCapture().to_dict(),
        "driver_sha256": pipeline.driver_sha256(),
        "gpu_wall_seconds": wall,
        "stderr_tail": "",
        "report_path": "baseline.ncu-rep",
        "report_sha256": "f" * 64,
        "evidence_path": BASELINE_EVIDENCE_NAME,
        "evidence_view": view,
        "summary": summarize_evidence(view),
        "problem_sha256": "abc123",
        "ncu_bin": "/fake/ncu",
        "source": "ncu_profile",
    }


def _run_optimize(config, responder, profiler=_PROFILER_UNSET):
    return optimize(
        config,
        generator=CandidateGenerator(responder, ledger=RecordingLedger()),
        evaluate=_passing_evaluate,
        timing=_fake_timing,
        correctness_pro=None,
        profiler=profiler,
    )


def test_loop_journals_profile_and_upgrades_classification(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)
    view = _fixture_view(_RAW_CSV_FULL)
    profiler_calls = []

    def profiler():
        profiler_calls.append(1)
        return _fake_profiler_outcome(view)

    result = _run_optimize(
        _config(tmp_path, snapshot),
        ScriptedResponder([GOOD_CANDIDATE]),
        profiler=profiler,
    )
    assert result.state == "completed"
    assert len(profiler_calls) == 1, "profile exactly once, between baseline and candidates"

    entries = _journal_entries(tmp_path / "run")
    kinds = [entry["kind"] for entry in entries]
    assert "profile_started" in kinds and "profile_collected" in kinds
    collected = next(entry for entry in entries if entry["kind"] == "profile_collected")
    assert collected["action_id"] == "profile-baseline"
    assert collected["source"] == "ncu_profile"
    assert collected["capture"]["mode"] == "metrics"
    assert collected["capture"]["metrics"] == list(CORE_TIER_A_METRICS)
    assert collected["billed_gpu_seconds"] == 42.5
    assert collected["summary"]["dram_throughput_pct_max"] == 72.5

    # Profiling counts into the durable gpu budget (its own settled event).
    settled = [
        entry
        for entry in entries
        if entry["kind"] == "budget_settled" and entry.get("action_id") == "profile-baseline"
    ]
    assert len(settled) == 1 and settled[0]["gpu_seconds"] == 42.5
    assert settled[0].get("attempt_id"), "profile billing is keyed by a fresh attempt id"

    # The planner consumed the NCU view: coverage is ncu_full, not static_only.
    plan = next(entry for entry in entries if entry["kind"] == "method_plan")
    assert plan["evidence_coverage"] == "ncu_full"
    assert plan["classification"] == "memory_bound"

    record = json.loads(
        (tmp_path / "run" / "records" / "baseline-eager.json").read_text(encoding="utf-8")
    )
    assert record["profile"]["status"] == "collected"
    assert record["profile"]["billed_against_gpu_budget"] is True
    assert record["profile"]["evidence"] == "profile/baseline-evidence.json"
    assert record["batches_ms"] == [10.0] * 12, "formal timing evidence unchanged by profiling"
    assert record["profile"]["summary"]["sm_throughput_pct_max"] == 38.4

    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert report["profile"]["status"] == "collected"
    assert report["profile"]["billed_against_gpu_budget"] is True
    durable = report["budget"]["settled_gpu_seconds"]
    assert durable >= 42.5, "profile GPU wall seconds are part of the settled budget"


def test_loop_profiler_not_run_degrades_without_breaking(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)
    outcome = {
        "status": "not_run",
        "reason": "ncu_unavailable_on_host",
        "detail": "no ncu binary",
        "capture": ProfileCapture().to_dict(),
    }
    result = _run_optimize(
        _config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]), profiler=lambda: outcome
    )
    assert result.state == "completed", "profiling failure must not break the run"

    entries = _journal_entries(tmp_path / "run")
    not_run = [entry for entry in entries if entry["kind"] == "profile_not_run"]
    assert len(not_run) == 1 and not_run[0]["reason"] == "ncu_unavailable_on_host"
    assert not [entry for entry in entries if entry["kind"] == "profile_collected"]
    plan = next(entry for entry in entries if entry["kind"] == "method_plan")
    assert plan["evidence_coverage"] == "static_only", "honest degradation of coverage"

    record = json.loads(
        (tmp_path / "run" / "records" / "baseline-eager.json").read_text(encoding="utf-8")
    )
    assert record["profile"]["status"] == "not_run"
    assert record["profile"]["reason"] == "ncu_unavailable_on_host"


def test_loop_profiler_exception_is_contained(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)

    def explode():
        raise RuntimeError("ncu exploded")

    result = _run_optimize(
        _config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]), profiler=explode
    )
    assert result.state == "completed"
    entries = _journal_entries(tmp_path / "run")
    not_run = [entry for entry in entries if entry["kind"] == "profile_not_run"]
    assert len(not_run) == 1 and not_run[0]["reason"] == "profiler_error"
    assert "RuntimeError" in not_run[0]["detail"]


def test_loop_resume_reuses_collected_evidence_without_rerun(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)
    view = _fixture_view(_RAW_CSV_FULL)
    bad_candidate = json.dumps(
        {"code": "class ModelNew:\n    def forward(self, x):\n        return 0"}
    )
    result = _run_optimize(
        _config(tmp_path, snapshot, max_candidates=1),
        ScriptedResponder([bad_candidate]),
        profiler=lambda: _fake_profiler_outcome(view),
    )
    assert result.state == "no_improvement"
    # Materialize the evidence file the real pipeline would have written.
    profile_dir = tmp_path / "run" / "profile"
    profile_dir.mkdir(exist_ok=True)
    (profile_dir / BASELINE_EVIDENCE_NAME).write_text(
        json.dumps(view, indent=2, sort_keys=True), encoding="utf-8"
    )

    def forbidden_profiler():
        raise AssertionError("resume must reuse collected evidence, not re-profile")

    resumed = optimize(
        _config(tmp_path, snapshot, resume=True, max_candidates=2, max_repair_rounds=1),
        generator=CandidateGenerator(ScriptedResponder([GOOD_CANDIDATE]), ledger=RecordingLedger()),
        evaluate=_passing_evaluate,
        timing=_fake_timing,
        correctness_pro=None,
        profiler=forbidden_profiler,
    )
    assert resumed.state == "completed"
    entries = _journal_entries(tmp_path / "run")
    assert len([e for e in entries if e["kind"] == "profile_started"]) == 1, "no re-profile"
    plans = [e for e in entries if e["kind"] == "method_plan"]
    assert len(plans) == 2, "resume planned the new candidate from the reused evidence"
    assert all(p["evidence_coverage"] == "ncu_full" for p in plans)


def test_loop_without_profiler_has_no_profile_events(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)
    result = _run_optimize(_config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]))
    assert result.state == "completed"
    entries = _journal_entries(tmp_path / "run")
    assert not [e for e in entries if e["kind"].startswith("profile_")]
    plan = next(e for e in entries if e["kind"] == "method_plan")
    assert plan["evidence_coverage"] == "static_only"
    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert report["profile"] is None


def test_default_planner_still_works_when_view_is_fixture_empty(tmp_path: Path):
    """A launches-but-no-metrics view must not break the planner either."""
    from kernelagent.generators.classify import classify_bottleneck

    view = _fixture_view(_RAW_CSV_PARTIAL)
    view["launches"][0]["metrics"] = {}
    view["catalog_metrics_missing_in_all_launches"] = list(CORE_TIER_A_METRICS)
    planner = MethodPlanner(load_method_catalog())
    plan = planner.plan(_planner_input(view))
    assert plan.classification.coverage == "static_only"  # profiled, nothing measurable
    assert classify_bottleneck(_planner_input(view)).coverage == "static_only"


# ---------------------------------------------------------------------------
# CLI: kernelagent profile --output <run-dir>
# ---------------------------------------------------------------------------


def test_cli_profile_collects_then_reuses_idempotently(tmp_path: Path, monkeypatch):
    snapshot = _stage_problem(tmp_path)
    view = _fixture_view(_RAW_CSV_FULL)
    # Seed a run WITHOUT profile evidence (no profiler), then profile via CLI.
    result = _run_optimize(_config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]))
    assert result.state == "completed"

    fake_outcome = _fake_profiler_outcome(view)
    monkeypatch.setattr(
        pipeline,
        "build_baseline_profiler",
        lambda **kwargs: lambda: fake_outcome,
    )
    code = cli_main(["profile", "--output", str(tmp_path / "run")])
    assert code == 0
    entries = _journal_entries(tmp_path / "run")
    assert [e for e in entries if e["kind"] == "profile_collected"]
    record = json.loads(
        (tmp_path / "run" / "records" / "baseline-eager.json").read_text(encoding="utf-8")
    )
    assert record["profile"]["status"] == "collected"
    # The real pipeline writes the evidence file next to the report; the
    # fake outcome stands in for it, so materialize it for the reuse path.
    profile_dir = tmp_path / "run" / "profile"
    profile_dir.mkdir(exist_ok=True)
    (profile_dir / BASELINE_EVIDENCE_NAME).write_text(
        json.dumps(view, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Second invocation: reuse without ever CALLING the profiler (building
    # the closure is harmless; collection must not rerun).
    def forbidden_builder(**kwargs):
        def forbidden_call():
            raise AssertionError("second profile run must reuse the collected evidence")

        return forbidden_call

    monkeypatch.setattr(pipeline, "build_baseline_profiler", forbidden_builder)
    assert cli_main(["profile", "--output", str(tmp_path / "run")]) == 0


def test_cli_profile_not_run_exits_1(tmp_path: Path, monkeypatch):
    snapshot = _stage_problem(tmp_path)
    result = _run_optimize(_config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]))
    assert result.state == "completed"
    outcome = {"status": "not_run", "reason": "timeout", "detail": "over budget"}
    monkeypatch.setattr(pipeline, "build_baseline_profiler", lambda **kwargs: lambda: outcome)
    assert cli_main(["profile", "--output", str(tmp_path / "run")]) == 1


def test_cli_profile_config_error_on_unknown_run(tmp_path: Path):
    assert cli_main(["profile", "--output", str(tmp_path / "missing")]) == 3


def test_cli_profile_flags_reach_capture(tmp_path: Path, monkeypatch):
    snapshot = _stage_problem(tmp_path)
    result = _run_optimize(_config(tmp_path, snapshot), ScriptedResponder([GOOD_CANDIDATE]))
    assert result.state == "completed"
    seen = {}

    def recording_builder(**kwargs):
        seen["capture"] = kwargs.get("capture")
        return lambda: {
            "status": "collected",
            "reason": None,
            "detail": "",
            "capture": seen["capture"].to_dict(),
            "driver_sha256": pipeline.driver_sha256(),
            "gpu_wall_seconds": 1.0,
            "stderr_tail": "",
            "report_path": "baseline.ncu-rep",
            "report_sha256": "f" * 64,
            "evidence_path": BASELINE_EVIDENCE_NAME,
            "evidence_view": _fixture_view(_RAW_CSV_FULL),
            "summary": {},
            "problem_sha256": "abc123",
            "ncu_bin": "/fake/ncu",
            "source": "ncu_profile",
        }

    monkeypatch.setattr(pipeline, "build_baseline_profiler", recording_builder)
    code = cli_main(
        [
            "profile",
            "--output",
            str(tmp_path / "run"),
            "--metrics",
            "gpu__time_duration.sum,launch__grid_size",
            "--launch-count",
            "4",
            "--timeout-seconds",
            "90",
        ]
    )
    assert code == 0
    capture = seen["capture"]
    assert capture.metrics == ("gpu__time_duration.sum", "launch__grid_size")
    assert capture.set_name is None and capture.launch_count == 4
    assert capture.timeout_seconds == 90.0
