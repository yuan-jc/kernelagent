"""Dispatch aggregation and guard semantics (T21): pure CPU pins. The
real GPU wrapper execution runs through examples/dispatch_smoke.py."""

import pytest

from kernelagent.dispatch import (
    WorkloadOutcome,
    WorkloadSpec,
    check_guards,
    select_implementation,
    weighted_aggregate,
)


def test_weight_change_changes_aggregate():
    timings = {"small": 1.0, "large": 10.0}
    even = weighted_aggregate(timings, {"small": 1.0, "large": 1.0})
    skewed = weighted_aggregate(timings, {"small": 9.0, "large": 1.0})
    assert even["weighted_ms"] == 5.5
    assert skewed["weighted_ms"] < even["weighted_ms"]


def test_aggregate_refuses_uncovered_workloads():
    """Summarizing only the benefiting shapes must be an explicit error."""
    with pytest.raises(ValueError, match="weights missing"):
        weighted_aggregate({"small": 1.0, "large": 10.0}, {"small": 1.0})


def test_check_guards_enforce_per_dim_caps():
    spec = WorkloadSpec(workload_id="huge", weight=1.0, max_shape=(2048, 2048))
    ok, _ = check_guards(spec, [(1024, 1024)])
    assert ok
    violated, reason = check_guards(spec, [(4096, 512)])
    assert not violated and "input 0" in reason and "exceeds" in reason


def _outcome(wid, correct=True, degraded=False, guard=False, fallback=False):
    return WorkloadOutcome(
        workload_id=wid,
        correct=correct,
        degraded=degraded,
        guard_violation=guard,
        fallback_used=fallback,
    )


def test_conservative_retreat_on_any_degradation():
    outcomes = [
        _outcome("w-small"),  # fast and correct
        _outcome("w-large", degraded=True),  # candidate lost here badly
    ]
    usable, _, reason = select_implementation(outcomes, required_ids=["w-small", "w-large"])
    assert usable is False
    assert "conservative retreat" in reason


def test_guard_violation_rejects_candidate():
    outcomes = [_outcome("w1", guard=True)]
    usable, _, reason = select_implementation(outcomes, required_ids=["w1"])
    assert usable is False and "guard" in reason


def test_incorrect_workload_rejects_candidate():
    outcomes = [_outcome("w1", correct=False)]
    usable, _, reason = select_implementation(outcomes, required_ids=["w1"])
    assert usable is False and "incorrect" in reason


def test_missing_required_workload_refuses():
    outcomes = [_outcome("w1")]
    usable, _, reason = select_implementation(outcomes, required_ids=["w1", "w2"])
    assert usable is False and "unmeasured" in reason


def test_all_good_is_usable():
    outcomes = [_outcome("w1"), _outcome("w2")]
    usable, _, reason = select_implementation(outcomes, required_ids=["w1", "w2"])
    assert usable and reason == "all required workloads verified"


def test_spec_guard_weight_validation():
    spec = WorkloadSpec(workload_id="w", weight=0.5)
    assert spec.required is True


# --- Launch-plan Task 4 (P2-1): guards are per-input, per-dimension. The
# old implementation multiplied every input's element count together, so
# two legal (16,16) inputs were rejected under max_shape=(16,16). ---


def test_two_legal_inputs_within_max_shape_pass():
    spec = WorkloadSpec(workload_id="w", weight=1.0, max_shape=(16, 16))
    ok, reason = check_guards(spec, [(16, 16), (16, 16)])
    assert ok, f"legal inputs must pass: {reason}"


def test_single_input_overflow_rejects_with_index():
    spec = WorkloadSpec(workload_id="w", weight=1.0, max_shape=(16, 16))
    ok, reason = check_guards(spec, [(16, 16), (17, 16)])
    assert not ok
    assert "input 1" in reason, "the rejection must name the offending input"


def test_rank_mismatch_rejects_with_index():
    spec = WorkloadSpec(workload_id="w", weight=1.0, max_shape=(16, 16))
    ok, reason = check_guards(spec, [(8, 8, 8)])
    assert not ok
    assert "input 0" in reason
