"""RV04 acceptance: GPU-second metering covers the real stages
(REVIEW.md "RV04: 预算覆盖实际阶段，非法配置提前失败", metering half).

Offline and fully deterministic: a controllable fake clock replaces
``time.monotonic`` as the metering clock, the GPU stage ports are fakes
that advance the clock by their simulated lease time, and no docker,
GPU, or model network is touched.

What is pinned here:

- GPU-second unit: the measured wall-clock span of each GPU stage's
  lease (staging/build + container execution + teardown happen inside
  the stage call), summed per attempt and settled as ``gpu_metering=
  "actual"``;
- attempts whose path never entered a GPU stage settle exactly 0 actual
  GPU seconds (tokens still billed) - no fictional GPU spend;
- per-stage reservation floors (evaluate 900 s, correctness_pro 900 s,
  timing 1200 s) folded into the attempt's reservation: a budget that
  cannot cover them refuses the action (``budget_refused`` ->
  ``budget_exhausted``) before any GPU or model call;
- deadline pass-through: ports that accept ``deadline_seconds`` receive
  ``min(stage floor, remaining budget)``; ports without the keyword are
  called unchanged; the real closures cap their container timeout at
  ``min(own timeout, deadline)``;
- interrupted attempts keep their reservation estimate held and their
  journal event says ``estimated=true`` with the held amounts;
  unlabeled/unknown settlements count as estimated, never as measured;
- B3 profile billing (its own measured-wall ``budget_settled`` under a
  fresh attempt id) stays compatible and is labeled actual.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelagent.adapters.models.client import ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.optimization import (
    CORRECTNESS_PRO_TIMEOUT_SECONDS,
    EVALUATE_TIMEOUT_SECONDS,
    EXIT_BUDGET_EXHAUSTED,
    TIMING_TIMEOUT_SECONDS,
    OptimizationConfig,
    _stage_deadline_seconds,
    optimize,
    parse_exit_code,
)
from kernelagent.orchestrator import (
    Action,
    Budget,
    Journal,
    Orchestrator,
    reconstruct_budget,
    settled_metering_split,
)

BASELINE_LEASE = 3.0
EVAL_LEASE = 5.0
TIMING_LEASE = 7.0


class FakeClock:
    """Controllable monotonic clock: metering reads it, ports advance it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedResponder:
    """Order-based model double that builds each response against the
    actual request (so hashes validate) and records every request seen."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=self._contents.pop(0),
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20),
        )


class RecordingLedger:
    def __init__(self):
        self._total_tokens = 0

    def record(self, response):
        self._total_tokens += response.usage.total_tokens

    def total_tokens(self) -> int:
        return self._total_tokens


def good_candidate() -> str:
    return json.dumps({"code": "class ModelNew:\n    def forward(self, x):\n        return x"})


def parse_failure() -> str:
    return json.dumps({"note": "no code key here"})


def policy_violation() -> str:
    return json.dumps(
        {
            "code": "import torch\nimport os\n"
            "class ModelNew:\n"
            "    def forward(self, x):\n"
            "        return x"
        }
    )


def _stage_problem(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text(
        "# pinned problem\n", encoding="utf-8"
    )
    return snapshot


def _make_ports(clock: FakeClock, *, calls: dict | None = None):
    """Offline GPU stage fakes with the LEGACY plain signature (no
    ``deadline_seconds``): the loop must call them unchanged. They consume
    their lease time from the fake clock."""

    def evaluate(candidate_source: str, candidate_name: str) -> dict:
        if calls is not None:
            calls["evaluate"] = calls.get("evaluate", 0) + 1
        clock.advance(EVAL_LEASE)
        return {
            "adapter_pass": True,
            "compiled": True,
            "correct": True,
            "outcome_status": "completed",
            "stderr_tail": "",
            "upstream_metadata": "",
        }

    def timing(candidate_source: str, role: str) -> dict:
        if calls is not None:
            calls[f"timing:{role}"] = calls.get(f"timing:{role}", 0) + 1
        clock.advance(TIMING_LEASE if role == "candidate" else BASELINE_LEASE)
        return {
            "source_ok": True,
            "precondition": True,
            "async_leak": False,
            # candidate batches faster than the eager baseline -> promoted
            "batches": [5.0] * 12 if role == "candidate" else [10.0] * 12,
        }

    return evaluate, timing


def _recording_ports(clock: FakeClock, deadlines_seen: dict, calls: dict | None = None):
    """Ports with the real adapters' signature shape: keyword-only
    ``deadline_seconds`` (the RV04 opt-in)."""

    def evaluate(candidate_source: str, candidate_name: str, *, deadline_seconds=None) -> dict:
        if calls is not None:
            calls["evaluate"] = calls.get("evaluate", 0) + 1
        deadlines_seen.setdefault(("candidate", "evaluate"), []).append(deadline_seconds)
        clock.advance(EVAL_LEASE)
        return {
            "adapter_pass": True,
            "compiled": True,
            "correct": True,
            "outcome_status": "completed",
            "stderr_tail": "",
            "upstream_metadata": "",
        }

    def timing(candidate_source: str, role: str, *, deadline_seconds=None) -> dict:
        if calls is not None:
            calls[f"timing:{role}"] = calls.get(f"timing:{role}", 0) + 1
        deadlines_seen.setdefault((role, "timing"), []).append(deadline_seconds)
        clock.advance(TIMING_LEASE if role == "candidate" else BASELINE_LEASE)
        return {
            "source_ok": True,
            "precondition": True,
            "async_leak": False,
            # candidate batches faster than the eager baseline -> promoted
            "batches": [5.0] * 12 if role == "candidate" else [10.0] * 12,
        }

    return evaluate, timing


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


def _run(config, contents_or_responder, *, evaluate, timing, clock, **kwargs):
    responder = (
        contents_or_responder
        if isinstance(contents_or_responder, ScriptedResponder)
        else ScriptedResponder(contents_or_responder)
    )
    result = optimize(
        config,
        generator=CandidateGenerator(responder, ledger=RecordingLedger()),
        evaluate=evaluate,
        timing=timing,
        correctness_pro=None,
        clock=clock,
        **kwargs,
    )
    return result, responder


def _journal_entries(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- measured settlement: the sum of the stage leases ----------------------


def test_multi_stage_measured_settlement_sums_under_fake_clock(tmp_path):
    clock = FakeClock()
    evaluate, timing = _make_ports(clock)
    config = _config(tmp_path, _stage_problem(tmp_path))
    result, _ = _run(config, [good_candidate()], evaluate=evaluate, timing=timing, clock=clock)

    assert result.state == "completed"
    settled = [
        entry for entry in _journal_entries(config.output) if entry["kind"] == "budget_settled"
    ]
    by_action = {entry["action_id"]: entry for entry in settled}
    # Each attempt's settlement is the SUM of its measured stage leases.
    assert by_action["baseline-eager"]["gpu_seconds"] == pytest.approx(BASELINE_LEASE)
    assert by_action["candidate-000"]["gpu_seconds"] == pytest.approx(EVAL_LEASE + TIMING_LEASE)
    assert all(entry["gpu_metering"] == "actual" for entry in settled)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    total = BASELINE_LEASE + EVAL_LEASE + TIMING_LEASE
    assert report["budget"]["settled_gpu_seconds"] == pytest.approx(total)
    assert report["budget"]["settled_gpu_seconds_actual"] == pytest.approx(total)
    assert report["budget"]["settled_gpu_seconds_estimated"] == 0.0
    assert report["budget"]["reserved_gpu_seconds"] == 0.0, (
        "settled attempts release their reservation"
    )
    assert "monotonic" in report["budget"]["gpu_seconds_definition"]


def test_no_gpu_work_settles_zero_gpu_seconds_but_tokens(tmp_path):
    """Generation-only and policy-rejected paths never enter a GPU stage:
    they settle 0 actual GPU seconds - the retired fixed quota billed 300
    for exactly these paths."""
    clock = FakeClock()
    calls: dict = {}
    evaluate, timing = _make_ports(clock, calls=calls)
    config = _config(tmp_path, _stage_problem(tmp_path), max_candidates=2, max_repair_rounds=1)
    result, responder = _run(
        config,
        [parse_failure(), policy_violation()],
        evaluate=evaluate,
        timing=timing,
        clock=clock,
    )
    assert result.state == "no_improvement"  # both candidates failed as planned
    assert calls.get("evaluate", 0) == 0, "neither failing candidate may reach the GPU"
    settled = [
        entry for entry in _journal_entries(config.output) if entry["kind"] == "budget_settled"
    ]
    by_action = {entry["action_id"]: entry for entry in settled}
    assert by_action["candidate-000"]["gpu_seconds"] == 0.0
    assert by_action["candidate-001"]["gpu_seconds"] == 0.0
    assert all(entry["gpu_metering"] == "actual" for entry in settled)
    assert by_action["candidate-000"]["tokens"] == 30, "token billing is unchanged"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["budget"]["settled_gpu_seconds"] == pytest.approx(BASELINE_LEASE), (
        "only the baseline lease was ever metered"
    )


# --- reservation floors: refuse before any GPU or model call ---------------


def test_budget_below_baseline_floor_refuses_before_any_call(tmp_path):
    clock = FakeClock()
    calls: dict = {}
    evaluate, timing = _make_ports(clock, calls=calls)
    config = _config(tmp_path, _stage_problem(tmp_path), gpu_budget_seconds=1000.0)
    result, responder = _run(
        config, [good_candidate()], evaluate=evaluate, timing=timing, clock=clock
    )

    assert result.state == "budget_exhausted"
    assert parse_exit_code(result) == EXIT_BUDGET_EXHAUSTED
    assert responder.calls == 0, "no model call without budget"
    assert calls.get("timing:eager", 0) == 0, "the baseline timing lease may not start"
    refused = [
        entry for entry in _journal_entries(config.output) if entry["kind"] == "budget_refused"
    ]
    assert len(refused) == 1 and refused[0]["action_id"] == "baseline-eager"


def test_budget_below_candidate_floors_refuses_candidate_start(tmp_path):
    """1200s covers the baseline but not a candidate's evaluate+timing
    floors (900+1200): the candidate is refused before generation."""
    clock = FakeClock()
    calls: dict = {}
    evaluate, timing = _make_ports(clock, calls=calls)
    config = _config(tmp_path, _stage_problem(tmp_path), gpu_budget_seconds=1500.0)
    result, responder = _run(
        config, [good_candidate()], evaluate=evaluate, timing=timing, clock=clock
    )

    assert result.state == "budget_exhausted"
    assert responder.calls == 0, "a refused candidate must not generate"
    assert calls.get("evaluate", 0) == 0
    assert calls.get("timing:eager", 0) == 1, "the funded baseline still ran"
    refused = [
        entry for entry in _journal_entries(config.output) if entry["kind"] == "budget_refused"
    ]
    assert len(refused) == 1 and refused[0]["action_id"] == "candidate-000"


# --- deadline pass-through to stage ports ----------------------------------


def test_deadline_reaches_ports_that_accept_it(tmp_path):
    """Ports with the real signature (keyword-only ``deadline_seconds``)
    receive min(stage floor, remaining budget). In every admissible state
    the attempt's own reservation funds at least its floors, so the
    deadline equals the stage floor here; ports WITHOUT the keyword (all
    other tests in this file) are called unchanged."""
    clock = FakeClock()
    deadlines_seen: dict = {}
    evaluate, timing = _recording_ports(clock, deadlines_seen)
    config = _config(tmp_path, _stage_problem(tmp_path))
    result, _ = _run(config, [good_candidate()], evaluate=evaluate, timing=timing, clock=clock)

    assert result.state == "completed"
    assert deadlines_seen[("eager", "timing")] == [TIMING_TIMEOUT_SECONDS]
    assert deadlines_seen[("candidate", "evaluate")] == [EVALUATE_TIMEOUT_SECONDS]
    assert deadlines_seen[("candidate", "timing")] == [TIMING_TIMEOUT_SECONDS]


def test_stage_deadline_subtracts_reservations_held_by_others():
    """Unit: the deadline is min(floor, limit - settled - held by OTHER
    attempts); a crashed attempt's held estimate tightens later
    attempts' deadlines, and the value clamps at 0."""
    budget = Budget(gpu_seconds_limit=10000.0, tokens_limit=100000)
    budget.settled_gpu_seconds = 3.0
    budget.reserved_gpu_seconds = 2100.0  # only this attempt's own floors
    assert _stage_deadline_seconds(budget, 2100.0, 900.0) == 900.0

    budget.reserved_gpu_seconds = 4200.0  # a crashed attempt holds 2100 more
    assert _stage_deadline_seconds(budget, 2100.0, 900.0) == pytest.approx(
        min(900.0, 10000.0 - 3.0 - 2100.0)
    )

    tight = Budget(gpu_seconds_limit=2000.0, tokens_limit=100000)
    tight.settled_gpu_seconds = 3.0
    tight.reserved_gpu_seconds = 4200.0  # own 2100 + a crashed attempt's 2100
    assert _stage_deadline_seconds(tight, 2100.0, 900.0) == 0.0, "never negative"


def test_real_stage_ports_cap_timeout_by_deadline(tmp_path, monkeypatch):
    """The real closures (config.build_stage_ports) apply min(own timeout,
    deadline) to the container timeout they pass to the drivers; no
    deadline keeps the configured timeout. No docker is touched: the
    driver calls are faked at the config-module boundary."""
    from kernelagent import config as config_module

    captured: dict[str, float | None] = {}

    def fake_evaluate_case(case, **kwargs):
        captured["evaluate"] = kwargs.get("timeout_seconds")
        return SimpleNamespace(
            adapter_pass=True,
            upstream_compiled=True,
            upstream_correctness=True,
            outcome_status="completed",
            detail={},
        )

    def fake_run_pro_case(**kwargs):
        captured["correctness_pro"] = kwargs.get("timeout_seconds")
        return {}  # overall_verdict({}) is a fail; only the timeout matters here

    def fake_run_timing_case(case, protocol, **kwargs):
        captured["timing"] = kwargs.get("timeout_seconds")
        return SimpleNamespace(
            source_ok=True,
            correctness_precondition=True,
            async_leak=False,
            batch_samples_ms=[1.0] * 12,
        )

    monkeypatch.setattr(config_module, "evaluate_case", fake_evaluate_case)
    monkeypatch.setattr(config_module, "run_pro_case", fake_run_pro_case)
    monkeypatch.setattr(config_module, "run_timing_case", fake_run_timing_case)

    from kernelagent.optimization import parse_problem_spec

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    problem_path = snapshot / "40_LayerNorm.py"
    problem_path.write_text("# pinned problem\n", encoding="utf-8")
    ports = config_module.build_stage_ports(
        _config(tmp_path, snapshot, output=tmp_path / "run"),
        parse_problem_spec("kernelbench:l1:40"),
        problem_path,
        snapshot,
        "nvidia.com/gpu=GPU-fake",
    )

    assert ports.evaluate("src", "c") is not None
    assert captured["evaluate"] == EVALUATE_TIMEOUT_SECONDS, "no deadline: configured timeout"
    ports.evaluate("src", "c", deadline_seconds=250.0)
    assert captured["evaluate"] == 250.0, "deadline below the floor caps the timeout"
    ports.evaluate("src", "c", deadline_seconds=99999.0)
    assert captured["evaluate"] == EVALUATE_TIMEOUT_SECONDS, "deadline above the floor: floor wins"
    ports.evaluate("src", "c", deadline_seconds=0.0)
    assert captured["evaluate"] == 0.0

    assert ports.timing("src", "eager") is not None
    assert captured["timing"] == TIMING_TIMEOUT_SECONDS
    ports.timing("src", "eager", deadline_seconds=100.0)
    assert captured["timing"] == 100.0

    ports.correctness_pro("src", "c")
    assert captured["correctness_pro"] == CORRECTNESS_PRO_TIMEOUT_SECONDS
    ports.correctness_pro("src", "c", deadline_seconds=42.5)
    assert captured["correctness_pro"] == 42.5


# --- interrupted / unknown: conservative, labeled estimates ----------------


def test_interrupted_attempt_holds_estimate_and_is_labeled_estimated(tmp_path):
    """A candidate whose GPU stage dies mid-lease: the measured seconds are
    lost WITH the process, so only the reservation estimate stays held
    (conservative), the journal event says estimated=true with the held
    amounts, and a resumed run re-executes under a fresh attempt."""
    clock = FakeClock()
    calls = {"evaluate": 0}

    def evaluate(candidate_source: str, candidate_name: str) -> dict:
        calls["evaluate"] += 1
        clock.advance(EVAL_LEASE)  # the lease ran, then the process died
        raise RuntimeError("simulated SIGKILL during the evaluation lease")

    plain_timing = _make_ports(clock)[1]
    snapshot = _stage_problem(tmp_path)
    config = _config(tmp_path, snapshot, gpu_budget_seconds=10000.0)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        _run(config, [good_candidate()], evaluate=evaluate, timing=plain_timing, clock=clock)

    entries = _journal_entries(config.output)
    assert not [
        entry
        for entry in entries
        if entry["kind"] == "budget_settled" and entry["action_id"] == "candidate-000"
    ], "no settlement without a finish"
    state = reconstruct_budget(Journal(config.output / "journal.jsonl").entries)
    assert state.reserved_gpu_seconds == pytest.approx(2100.0), (
        "the interrupted attempt keeps holding its floor estimate (900+1200)"
    )

    # Recovery: a fresh process reconciles, labels the interruption, re-runs.
    resumed_clock = clock
    recording: dict = {}
    rec_evaluate, rec_timing = _recording_ports(resumed_clock, recording)
    result, responder = _run(
        _config(tmp_path, snapshot, gpu_budget_seconds=10000.0, resume=True),
        [good_candidate()],
        evaluate=rec_evaluate,
        timing=rec_timing,
        clock=resumed_clock,
    )
    assert result.state == "completed"
    assert responder.calls == 1, "the re-execution generates under a fresh attempt"
    entries = _journal_entries(config.output)
    interrupted = [entry for entry in entries if entry["kind"] == "action_interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["estimated"] is True
    assert interrupted[0]["held_gpu_estimate"] == pytest.approx(2100.0)
    assert interrupted[0]["held_tokens_estimate"] == 2048

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    # Measured: baseline 3.0 + the resumed evaluate 5.0 + timing 7.0. The
    # lost crashed lease is covered by the still-HELD estimate, never
    # double-billed as a measurement.
    assert report["budget"]["settled_gpu_seconds_actual"] == pytest.approx(15.0)
    assert report["budget"]["settled_gpu_seconds_estimated"] == 0.0
    assert report["budget"]["reserved_gpu_seconds"] == pytest.approx(2100.0), (
        "the crashed attempt's estimate stays held - reported as estimate, never as measured usage"
    )
    # The deadline seen by the resumed attempt is tightened by the held estimate.
    assert recording[("candidate", "evaluate")] == [900.0], (
        "still funded: limit 10000 - settled - held 2100 >= 900"
    )


def test_unknown_settlement_fallback_is_labeled_estimated(tmp_path):
    """Orchestrator honesty rule: a result without a measurement settles at
    the reservation estimate with gpu_metering="estimated"; unlabeled
    settlements (legacy journals) count as estimated, never actual."""
    journal_path = tmp_path / "journal.jsonl"
    legacy = Journal(journal_path)
    legacy.append("budget_settled", action_id="legacy", gpu_seconds=300.0, tokens=1)

    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=1000))
    orchestrator.run(
        [Action("legacy-style", "input", 30.0, 5, lambda: {"result_ref": "r"})], resume=True
    )
    settled = [entry for entry in orchestrator.journal.entries if entry["kind"] == "budget_settled"]
    assert settled[-1]["gpu_metering"] == "estimated"
    assert settled[-1]["gpu_seconds"] == 30.0

    actual, estimated = settled_metering_split(orchestrator.journal.entries)
    assert actual == 0.0
    assert estimated == pytest.approx(330.0), "unlabeled legacy + fallback are estimates"


# --- B3 profile billing stays compatible -----------------------------------


def test_profile_billing_stays_compatible_and_labeled_actual(tmp_path):
    clock = FakeClock()
    evaluate, timing = _make_ports(clock)
    config = _config(tmp_path, _stage_problem(tmp_path))
    profiler_outcome = {
        "status": "collected",
        "reason": None,
        "detail": "",
        "capture": {},
        "driver_sha256": "d" * 64,
        "gpu_wall_seconds": 42.5,
        "report_path": "baseline.ncu-rep",
        "report_sha256": "f" * 64,
        "evidence_path": "baseline-evidence.json",
        "evidence_view": None,
        "summary": {},
        "source": "ncu_profile",
    }
    result, _ = _run(
        config,
        [good_candidate()],
        evaluate=evaluate,
        timing=timing,
        clock=clock,
        profiler=lambda: profiler_outcome,
    )
    assert result.state == "completed"
    entries = _journal_entries(config.output)
    profile_settled = [
        entry
        for entry in entries
        if entry["kind"] == "budget_settled" and entry["action_id"] == "profile-baseline"
    ]
    assert len(profile_settled) == 1
    assert profile_settled[0]["gpu_seconds"] == pytest.approx(42.5)
    assert profile_settled[0]["gpu_metering"] == "actual", (
        "the B3 precedent bills its measured wall time as an actual"
    )
    assert profile_settled[0].get("attempt_id"), "profile billing keeps its fresh attempt id"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    expected_total = 42.5 + BASELINE_LEASE + EVAL_LEASE + TIMING_LEASE
    assert report["budget"]["settled_gpu_seconds"] == pytest.approx(expected_total)
    assert report["budget"]["settled_gpu_seconds_actual"] == pytest.approx(expected_total)
    assert report["budget"]["settled_gpu_seconds_estimated"] == 0.0
    assert report["profile"]["billed_against_gpu_budget"] is True


def test_candidate_records_carry_no_metering_fields(tmp_path):
    """RV03 record parity: candidate records must stay byte-identical
    between an uninterrupted and a resumed path, so measured per-attempt
    GPU seconds live only in the journal, never in the records."""
    clock = FakeClock()
    evaluate, timing = _make_ports(clock)
    config = _config(tmp_path, _stage_problem(tmp_path))
    _run(config, [good_candidate()], evaluate=evaluate, timing=timing, clock=clock)
    for name in ("baseline-eager", "candidate-000"):
        record = json.loads(
            (config.output / "records" / f"{name}.json").read_text(encoding="utf-8")
        )
        assert "gpu_seconds" not in record
        assert "gpu_metering" not in record
