"""Promotion and aggregation rules (T09, design §9.4-9.5).

Trusted-parent decision logic over evidence that already exists: the
batch samples produced by the T07 formal timing protocol and the
correctness verdict produced by the pinned evaluator (T05). This module
never times anything and never judges correctness itself - it derives
one auditable decision object from recorded facts.

Promotion rules v1 (frozen in docs/work-packages/T09.md before
implementation):
- Hard constraints first: a candidate that is not compiled+correct, or
  exceeded resources, or failed a guard, is rejected with the matching
  terminal state no matter how fast it looked.
- The confirm comparison is the bootstrap CI of S = mean(incumbent
  batches) / mean(candidate batches); the resampling unit is the
  independent batch. Promote only when the CI lower bound exceeds
  ``1 + delta``; an interval that cannot establish that keeps the
  incumbent - uncertainty is a legal, recorded outcome, never a failure
  to compute.
- Aggregation keeps denominators honest: failed and unsupported tasks
  are reported separately and stay in the attempted denominator;
  ``fast_p`` counts strictly greater-than ratios and only correct tasks
  as successes."""

import hashlib
import json
import math
from dataclasses import dataclass, field

from kernelagent.adapters.evals.timing import bootstrap_ratio_ci

TERMINAL_PROMOTE = "promote"
TERMINAL_RETAIN = "retain_incumbent"
REJECT_INCORRECT = "reject_incorrect"
REJECT_RESOURCE = "reject_resource"
REJECT_GUARD = "reject_guard"


@dataclass(frozen=True, slots=True)
class CandidateFacts:
    """Recorded facts about one candidate, as observed by the trusted
    evaluator and worker boundary."""

    compiled: bool
    correct: bool
    resource_ok: bool = True
    guard_ok: bool = True
    failure_kind: str | None = None  # e.g. "oom", "timeout", "compile_error"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    outcome: str
    reason: str
    ratio_ci_95: tuple[float, float] | None
    delta: float
    confidence: float
    seed: int
    decision_input_sha256: str
    detail: dict = field(default_factory=dict)


def _canonical_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def confirm_promotion(
    *,
    facts: CandidateFacts,
    incumbent_batches_ms: list[float],
    candidate_batches_ms: list[float],
    delta: float = 0.02,
    confidence: float = 0.95,
    seed: int = 0,
    protocol_sha256: str | None = None,
    min_batches: int = 5,
) -> PromotionDecision:
    """Apply promotion rules v1 to recorded facts and batch samples."""
    if not math.isfinite(delta) or delta < 0:
        raise ValueError(f"delta must be a finite non-negative number; got {delta!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")

    outcome = TERMINAL_RETAIN
    reason = ""
    if not facts.compiled or not facts.correct:
        outcome = REJECT_INCORRECT
        reason = "candidate did not compile or was not correct under the pinned evaluator"
    elif not facts.resource_ok:
        outcome = REJECT_RESOURCE
        reason = f"resource limit exceeded ({facts.failure_kind or 'unspecified'})"
    elif not facts.guard_ok:
        outcome = REJECT_GUARD
        reason = f"guard rejected the candidate ({facts.failure_kind or 'unspecified'})"

    ratio_ci = None
    input_enough = (
        len(incumbent_batches_ms) >= min_batches and len(candidate_batches_ms) >= min_batches
    )
    if outcome == TERMINAL_RETAIN and not reason:
        if not input_enough:
            reason = (
                f"insufficient independent batches for a confirm comparison "
                f"(incumbent={len(incumbent_batches_ms)}, candidate={len(candidate_batches_ms)}, "
                f"required={min_batches})"
            )
        else:
            ratio_ci = bootstrap_ratio_ci(
                incumbent_batches_ms, candidate_batches_ms, seed=seed, confidence=confidence
            )
            if math.isnan(ratio_ci[0]):
                reason = "bootstrap produced no usable samples"
            elif ratio_ci[0] > 1.0 + delta:
                outcome = TERMINAL_PROMOTE
                reason = f"CI lower bound {ratio_ci[0]:.4f} exceeds 1 + delta ({1.0 + delta:.4f})"
            else:
                reason = (
                    f"CI [{ratio_ci[0]:.4f}, {ratio_ci[1]:.4f}] does not establish speedup "
                    f"beyond 1 + delta ({1.0 + delta:.4f}); incumbent kept"
                )
    decision_input = {
        "facts": {
            "compiled": facts.compiled,
            "correct": facts.correct,
            "resource_ok": facts.resource_ok,
            "guard_ok": facts.guard_ok,
            "failure_kind": facts.failure_kind,
        },
        "incumbent_batches_ms": incumbent_batches_ms,
        "candidate_batches_ms": candidate_batches_ms,
        "delta": delta,
        "confidence": confidence,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "min_batches": min_batches,
    }
    return PromotionDecision(
        outcome=outcome,
        reason=reason,
        ratio_ci_95=ratio_ci,
        delta=delta,
        confidence=confidence,
        seed=seed,
        decision_input_sha256=_canonical_hash(decision_input),
        detail={
            "incumbent_batches": len(incumbent_batches_ms),
            "candidate_batches": len(candidate_batches_ms),
        },
    )


def fast_p(
    results: list[dict],
    *,
    threshold: float,
    ref_key: str = "reference_ms",
    candidate_key: str = "candidate_ms",
    correct_key: str = "correct",
) -> dict:
    """Upstream-style fast_p over one frozen task list (§9.5).

    Each entry carries the reference/candidate timing and the correctness
    verdict for one task. Denominators stay honest: every entry remains
    counted, and failed/unsupported entries are reported separately - they
    can never become successes, and removing them from N requires an
    explicit separate report (``eligible_denominator``)."""
    if not results:
        raise ValueError("fast_p requires a non-empty task list")
    successes = 0
    failures = 0
    unsupported = 0
    for entry in results:
        correct = entry.get(correct_key)
        ref = entry.get(ref_key)
        cand = entry.get(candidate_key)
        if correct is not True:
            failures += 1
            continue
        if ref is None or cand is None or cand <= 0:
            unsupported += 1
            continue
        if ref / cand > threshold:  # strictly greater, per implementation note
            successes += 1
    n = len(results)
    return {
        "fast_p": successes / n,
        "successes": successes,
        "attempted_denominator": n,
        "failures_in_denominator": failures,
        "unsupported_unmeasured": unsupported,
        "eligible_denominator": n - unsupported,
        "threshold": threshold,
        "rule": "success requires correct AND reference_ms / candidate_ms > threshold (strict)",
    }


def aggregate_coverage(results: list[dict], *, attempted: int) -> dict:
    """Coverage report distinguishing attempted from eligible tasks; a
    shrinking denominator must be visible, never implicit."""
    if attempted <= 0:
        raise ValueError("attempted must be a positive count")
    measured = sum(1 for entry in results if entry.get("candidate_ms") is not None)
    return {
        "attempted": attempted,
        "with_candidate_measurement": measured,
        "attempted_coverage": measured / attempted,
        "without_measurement": attempted - measured,
    }
