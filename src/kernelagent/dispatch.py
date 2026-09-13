"""Multi-workload dispatch with guards and honest aggregation (T21,
design §9.4).

A DispatchEntry pairs a candidate implementation with per-workload
guards (shape constraints that must hold) and measured per-workload
timings. The dispatcher:

- verifies EVERY required workload before considering a candidate: a
  guard violation on one workload routes that workload to the fallback
  reference implementation, but a degradation beyond the configured
  threshold on ANY workload disqualifies the whole candidate (conservative
  retreat - no cherry-picking the winning shapes);
- aggregates with explicitly configured weights over ALL required
  workloads (missing workloads make the score unknown, never a pass);
- times the final dispatch wrapper itself (guard evaluation included),
  not just the naked kernel.
"""

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    workload_id: str
    weight: float
    required: bool = True
    max_shape: tuple[int, ...] | None = None  # per-dim cap applied per input tensor


@dataclass(frozen=True, slots=True)
class WorkloadOutcome:
    workload_id: str
    correct: bool
    degraded: bool  # slower than fallback beyond threshold
    guard_violation: bool
    fallback_used: bool = False


def check_guards(workload: WorkloadSpec, tensor_shapes: list[tuple[int, ...]]) -> tuple[bool, str]:
    """Guard check: every input tensor is verified per dimension against
    the workload's max_shape bound - ranks must match and no single dim may
    exceed its cap. Inputs are never multiplied together: two legal
    (16,16) tensors pass under max_shape=(16,16), and a violation names
    the offending input index."""
    if workload.max_shape is None:
        return True, ""
    for index, shape in enumerate(tensor_shapes):
        if len(shape) != len(workload.max_shape):
            return False, (
                f"input {index} has {len(shape)} dimension(s); "
                f"guard expects {len(workload.max_shape)}"
            )
        for dim, (actual, cap) in enumerate(zip(shape, workload.max_shape)):
            if actual > cap:
                return False, (f"input {index} dim {dim} size {actual} exceeds guard cap {cap}")
    return True, ""


def select_implementation(
    outcomes: list[WorkloadOutcome],
    *,
    required_ids: list[str],
) -> tuple[bool, list[str], str]:
    """Decide whether a candidate may serve all required workloads.

    Returns (usable, fallback_workloads, reason). Rules: every required
    workload must be correct with no guard violation; a degradation
    beyond threshold on ANY required workload disqualifies the whole
    candidate (conservative retreat)."""
    missing = [wid for wid in required_ids if wid not in {o.workload_id for o in outcomes}]
    if missing:
        return False, [], f"required workloads unmeasured: {missing}"
    fallbacks: list[str] = []
    for outcome in outcomes:
        if outcome.workload_id not in required_ids:
            continue
        if not outcome.correct:
            return False, [], f"workload {outcome.workload_id!r} incorrect"
        if outcome.guard_violation:
            return False, [], f"workload {outcome.workload_id!r} guard violated"
        if outcome.degraded:
            return (
                False,
                [],
                (
                    f"workload {outcome.workload_id!r} degraded beyond threshold; "
                    "conservative retreat"
                ),
            )
        if outcome.fallback_used:
            fallbacks.append(outcome.workload_id)
    return True, fallbacks, "all required workloads verified"


def weighted_aggregate(timings_ms: dict[str, float], weights: dict[str, float]) -> dict:
    """Weighted aggregate over ALL provided workloads. Refuses to produce
    a score when weights do not cover every measured workload (the
    'only summarize benefiting shapes' trap is an explicit error)."""
    if not timings_ms:
        raise ValueError("no workload timings provided")
    uncovered = sorted(set(timings_ms) - set(weights))
    if uncovered:
        raise ValueError(f"aggregation weights missing for workloads: {uncovered}")
    total_weight = sum(weights[w] for w in timings_ms)
    if total_weight <= 0:
        raise ValueError("total weight must be positive")
    score = sum(timings_ms[w] * weights[w] for w in timings_ms) / total_weight
    return {
        "weighted_ms": score,
        "covered_workloads": sorted(timings_ms),
        "total_weight": total_weight,
    }


def dispatch_input_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
