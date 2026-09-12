"""Promotion and aggregation rules (T09, design §9.4-9.5).

Pure control-plane statistics over recorded evidence: hard constraints
first, batch-level bootstrap confirmation, honest denominators, and a
legal retain/no-speedup terminal. Real GPU evidence is produced by T05/T07
runs; these unit tests pin the decision semantics."""

import pytest

from kernelagent.promotion import (
    REJECT_GUARD,
    REJECT_INCORRECT,
    REJECT_RESOURCE,
    TERMINAL_PROMOTE,
    TERMINAL_RETAIN,
    CandidateFacts,
    aggregate_coverage,
    confirm_promotion,
    fast_p,
)

INCUMBENT = [10.0 + (i % 4) * 0.05 for i in range(12)]  # ~10.02 ms mean
FASTER = [5.0 + (i % 4) * 0.02 for i in range(12)]  # ~5.03 ms mean
SIMILAR = [10.0 + (i % 4) * 0.05 for i in range(12)]  # same as incumbent


def _facts(**overrides) -> CandidateFacts:
    values = {"compiled": True, "correct": True}
    values.update(overrides)
    return CandidateFacts(**values)


def test_incorrect_or_uncompiled_never_promotes():
    for facts in (_facts(correct=False), _facts(compiled=False)):
        decision = confirm_promotion(
            facts=facts, incumbent_batches_ms=INCUMBENT, candidate_batches_ms=FASTER
        )
        assert decision.outcome == REJECT_INCORRECT
        assert decision.ratio_ci_95 is None, "no comparison may run for an incorrect candidate"


def test_resource_exceeded_rejected_even_when_faster():
    decision = confirm_promotion(
        facts=_facts(resource_ok=False, failure_kind="oom"),
        incumbent_batches_ms=INCUMBENT,
        candidate_batches_ms=FASTER,
    )
    assert decision.outcome == REJECT_RESOURCE
    assert "oom" in decision.reason


def test_guard_failure_is_its_own_terminal():
    decision = confirm_promotion(
        facts=_facts(guard_ok=False, failure_kind="shape_guard"),
        incumbent_batches_ms=INCUMBENT,
        candidate_batches_ms=FASTER,
    )
    assert decision.outcome == REJECT_GUARD


def test_clear_speedup_promotes():
    decision = confirm_promotion(
        facts=_facts(),
        incumbent_batches_ms=INCUMBENT,
        candidate_batches_ms=FASTER,
    )
    assert decision.outcome == TERMINAL_PROMOTE
    assert decision.ratio_ci_95 is not None
    assert decision.ratio_ci_95[0] > 1.0 + decision.delta
    assert decision.decision_input_sha256


def test_uncertain_interval_keeps_incumbent():
    """A/A-ish candidate: CI covers 1+delta, incumbent kept - a legal
    outcome, not an error, and the CI is recorded in the decision."""
    decision = confirm_promotion(
        facts=_facts(),
        incumbent_batches_ms=SIMILAR,
        candidate_batches_ms=[x * 1.001 for x in SIMILAR],
    )
    assert decision.outcome == TERMINAL_RETAIN
    assert decision.ratio_ci_95 is not None
    assert decision.ratio_ci_95[0] <= 1.0 + decision.delta


def test_correct_but_not_faster_is_legal_retain():
    decision = confirm_promotion(
        facts=_facts(),
        incumbent_batches_ms=INCUMBENT,
        candidate_batches_ms=[x * 1.0005 for x in INCUMBENT],
    )
    assert decision.outcome == TERMINAL_RETAIN
    assert "incumbent kept" in decision.reason


def test_insufficient_batches_never_promotes():
    decision = confirm_promotion(
        facts=_facts(),
        incumbent_batches_ms=INCUMBENT[:3],
        candidate_batches_ms=FASTER,
    )
    assert decision.outcome == TERMINAL_RETAIN
    assert "insufficient" in decision.reason


def test_decision_records_identity_fields():
    decision = confirm_promotion(
        facts=_facts(),
        incumbent_batches_ms=INCUMBENT,
        candidate_batches_ms=FASTER,
        delta=0.05,
        seed=9,
        protocol_sha256="abc",
    )
    payload = decision.__dict__ if hasattr(decision, "__dict__") else {}
    assert decision.delta == 0.05
    assert decision.seed == 9
    assert decision.decision_input_sha256
    assert payload is not None or decision.detail is not None


def test_fast_p_strictly_greater_and_denominators_honest():
    report = fast_p(
        [
            {"correct": True, "reference_ms": 10.0, "candidate_ms": 5.0},  # 2x > 2? no, == 2 not >
            {"correct": True, "reference_ms": 10.0, "candidate_ms": 4.0},  # 2.5 > 2 yes
            {"correct": False, "reference_ms": 10.0, "candidate_ms": 1.0},  # fast but wrong
            {"correct": True, "reference_ms": 10.0, "candidate_ms": None},  # unmeasured
        ],
        threshold=2.0,
    )
    assert report["successes"] == 1, "boundary case ref/cand == threshold must not count"
    assert report["fast_p"] == 0.25, "denominator stays at attempted tasks"
    assert report["failures_in_denominator"] == 1
    assert report["unsupported_unmeasured"] == 1
    assert report["eligible_denominator"] == 3


def test_fast_p_empty_list_rejected():
    with pytest.raises(ValueError):
        fast_p([], threshold=1.5)


def test_coverage_reports_attempted_vs_measured():
    coverage = aggregate_coverage(
        [{"candidate_ms": 1.0}, {"candidate_ms": None}, {"candidate_ms": 2.0}],
        attempted=4,
    )
    assert coverage["attempted"] == 4
    assert coverage["with_candidate_measurement"] == 2
    assert coverage["attempted_coverage"] == 0.5
    assert coverage["without_measurement"] == 2
