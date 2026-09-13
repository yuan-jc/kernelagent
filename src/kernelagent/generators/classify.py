"""Bottleneck classification (generator-design §2).

A pure function over :class:`~kernelagent.generators.input.GeneratorInput`.
Four MVP bottleneck classes are decided by priority short-circuit
(numerical -> launch_bound -> parallel_deficit -> memory_bound); two
reserved classes (compute_bound, latency_bound) follow the design table.
When nothing fires the label is ``uncertain`` - the planner answers with a
low-risk baseline plan instead of pretending to know.

Honesty rules (design §2.3, adapters/profiling/ncu.py semantics):

- a Tier-A signal whose metrics are MISSING is neither fired nor treated as
  counter-evidence; it is recorded in ``signals_missing`` verbatim;
- the coverage tag (ncu_full | ncu_partial | static_only | history_only)
  travels with every plan item so downstream readers can see how much
  evidence the hypothesis rests on;
- NCU evidence characterizes the BASELINE (ADR-0003: candidates are never
  profiled) - a plan based on it is a hypothesis about the candidate that
  formal timing must confirm.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kernelagent.generators.input import GeneratorInput

# Mirrors adapters/profiling/ncu.py MetricCatalog.default(). Kept in sync by
# tests/test_method_generators.py (the generators package must stay free of
# adapter imports so it stays offline-pure).
CORE_TIER_A_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)

BOTTLENECK_LABELS: tuple[str, ...] = (
    "numerical",
    "launch_bound",
    "parallel_deficit",
    "memory_bound",
    "compute_bound",
    "latency_bound",
    "uncertain",
)
COVERAGE_LEVELS: tuple[str, ...] = (
    "ncu_full",
    "ncu_partial",
    "static_only",
    "history_only",
)

# Thresholds are the design §2.1 table's heuristics, not calibrated constants.
SHORT_KERNEL_US = 20.0
MULTI_KERNEL_LAUNCH_COUNT = 4
MULTI_KERNEL_FORWARD_CALLS = 3
SATURATED_PCT = 60.0
LOW_THROUGHPUT_PCT = 40.0
LOW_WARPS_PCT = 30.0
GRID_PER_SM_FACTOR = 2.0
MID_THROUGHPUT_RANGE = (40.0, 70.0)

# Streaming normalization/attention families: the static prior says their
# naive forms materialize large intermediates (design §2.2 prior table).
_STREAMING_CLASSES = frozenset({"softmax", "attention", "layernorm", "rmsnorm"})

_NAN_NOTE = re.compile(
    r"\bnan\b|\binf\b|\binfinity\b|overflow|out of bounds|index out of range|"
    r"illegal memory|越界|间歇|intermittent",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BottleneckClass:
    """Honest classification result: label, why, and what could not be seen."""

    label: str
    signals_fired: tuple[str, ...]
    signals_missing: tuple[str, ...]
    coverage: str

    def __post_init__(self) -> None:
        if self.label not in BOTTLENECK_LABELS:
            raise ValueError(f"label must be one of {BOTTLENECK_LABELS}; got {self.label!r}")
        if self.coverage not in COVERAGE_LEVELS:
            raise ValueError(f"coverage must be one of {COVERAGE_LEVELS}; got {self.coverage!r}")


class _NCUView:
    """Read access to an ``evidence_view()`` payload with MISSING semantics:
    a metric absent from all launches is MISSING (None), never 0."""

    def __init__(self, view: Mapping[str, Any] | None):
        self._view = view
        launches = view.get("launches") if isinstance(view, Mapping) else None
        self._launches: tuple[Mapping[str, Any], ...] = (
            tuple(launch for launch in launches if isinstance(launch, Mapping))
            if isinstance(launches, list)
            else ()
        )
        missing = (
            view.get("catalog_metrics_missing_in_all_launches")
            if isinstance(view, Mapping)
            else None
        )
        self.missing_in_all: frozenset[str] = (
            frozenset(metric for metric in missing if isinstance(metric, str))
            if isinstance(missing, list)
            else frozenset()
        )

    @property
    def has_launches(self) -> bool:
        return bool(self._launches)

    @property
    def launch_count(self) -> int:
        return len(self._launches)

    def value(self, metric: str) -> float | None:
        """Max across launches (bottleneck = any launch saturates), or None
        when the metric is MISSING in every launch."""
        values: list[float] = []
        for launch in self._launches:
            metrics = launch.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            entry = metrics.get(metric)
            if isinstance(entry, Mapping) and isinstance(entry.get("value"), int | float):
                values.append(float(entry["value"]))
        return max(values) if values else None

    def any_metric_present(self) -> bool:
        return any(self.value(metric) is not None for metric in CORE_TIER_A_METRICS)


def _has_numerical_failure(input_data: GeneratorInput) -> tuple[bool, tuple[str, ...]]:
    """Tier-C numerical signal: a correctness-class failure whose note carries
    inf/NaN/out-of-bounds/intermittent evidence (design §2.1 numerical row)."""
    fired: list[str] = []
    for attempt in input_data.attempts:
        if attempt.failure_class != "correctness" or attempt.outcome != "failed":
            continue
        if attempt.stage not in ("evaluate", "correctness_pro", "timing"):
            continue
        if _NAN_NOTE.search(attempt.note):
            fired.append(
                f"tier_c: {attempt.stage} failure note matches numerical-instability "
                f"pattern ({attempt.stage}: {attempt.note[:120]})"
            )
    return bool(fired), tuple(fired)


def _grid_size(ncu: _NCUView) -> float | None:
    return ncu.value("launch__grid_size")


def classify_bottleneck(input_data: GeneratorInput) -> BottleneckClass:
    """Classify the bottleneck with priority short-circuit (design §2.1/2.2)."""
    ncu = _NCUView(input_data.ncu_view)
    fired: list[tuple[int, str]] = []
    missing: list[str] = []
    op_class = _operator_class(input_data)
    features = input_data.code_features

    def tier_a(name: str, metrics: tuple[str, ...], priority: int, condition, text) -> None:
        values = {metric: ncu.value(metric) for metric in metrics}
        unavailable = sorted(metric for metric, value in values.items() if value is None)
        if unavailable:
            missing.append(f"tier_a:{name}: metrics MISSING, signal not evaluated: {unavailable}")
            return
        if condition(values):
            fired.append((priority, f"tier_a:{name}: {text(values)}"))

    numerical, numerical_signals = _has_numerical_failure(input_data)
    fired.extend((0, signal) for signal in numerical_signals)

    # launch_bound: extremely short kernels or a multi-launch/multi-call chain.
    tier_a(
        "short_kernel",
        ("gpu__time_duration.sum",),
        1,
        lambda v: float(v["gpu__time_duration.sum"]) <= SHORT_KERNEL_US,
        lambda v: (
            f"max gpu__time_duration.sum={v['gpu__time_duration.sum']}us <= {SHORT_KERNEL_US}us"
        ),
    )
    if ncu.launch_count >= MULTI_KERNEL_LAUNCH_COUNT:
        fired.append(
            (
                1,
                f"tier_a:multi_launch_sequence: launch_count={ncu.launch_count} >= "
                f"{MULTI_KERNEL_LAUNCH_COUNT} (multi-kernel chain)",
            )
        )
    if features.forward_call_count >= MULTI_KERNEL_FORWARD_CALLS:
        fired.append(
            (
                1,
                f"tier_b:multi_kernel_chain: forward issues {features.forward_call_count} "
                f"calls (multi-kernel chain suspected)",
            )
        )

    # parallel_deficit: too few CTAs for the device, or measured under-occupancy
    # with both main throughputs low.
    if input_data.hardware.sm_count is not None:
        tier_a(
            "small_grid",
            ("launch__grid_size",),
            2,
            lambda v: (
                float(v["launch__grid_size"]) < GRID_PER_SM_FACTOR * input_data.hardware.sm_count
            ),
            lambda v: (
                f"launch__grid_size={int(v['launch__grid_size'])} < "
                f"{GRID_PER_SM_FACTOR}*sm_count={GRID_PER_SM_FACTOR * input_data.hardware.sm_count}"
            ),
        )
    else:
        missing.append(
            "tier_a:small_grid: hardware.sm_count MISSING, grid-based signal not evaluated"
        )
    tier_a(
        "under_occupancy",
        (
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        2,
        lambda v: (
            float(v["sm__warps_active.avg.pct_of_peak_sustained_active"]) < LOW_WARPS_PCT
            and float(v["sm__throughput.avg.pct_of_peak_sustained_elapsed"]) < LOW_THROUGHPUT_PCT
            and float(v["dram__throughput.avg.pct_of_peak_sustained_elapsed"]) < LOW_THROUGHPUT_PCT
        ),
        lambda v: (
            "warps_active="
            f"{v['sm__warps_active.avg.pct_of_peak_sustained_active']}% < {LOW_WARPS_PCT}% "
            f"with sm/dram throughput both < {LOW_THROUGHPUT_PCT}%"
        ),
    )

    # memory_bound: DRAM saturated, or the static streaming-family prior.
    tier_a(
        "dram_saturated",
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",),
        3,
        lambda v: float(v["dram__throughput.avg.pct_of_peak_sustained_elapsed"]) >= SATURATED_PCT,
        lambda v: (
            f"dram__throughput.avg.pct_of_peak_sustained_elapsed="
            f"{v['dram__throughput.avg.pct_of_peak_sustained_elapsed']}% >= {SATURATED_PCT}%"
        ),
    )
    if op_class in _STREAMING_CLASSES:
        fired.append(
            (
                3,
                f"tier_b:streaming_family_prior: operator_class {op_class!r} typically "
                "materializes large intermediates in its naive form",
            )
        )

    # compute_bound (reserved): SM saturated, or matmul-family source without tl.dot.
    tier_a(
        "sm_saturated",
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
        4,
        lambda v: float(v["sm__throughput.avg.pct_of_peak_sustained_elapsed"]) >= SATURATED_PCT,
        lambda v: (
            f"sm__throughput.avg.pct_of_peak_sustained_elapsed="
            f"{v['sm__throughput.avg.pct_of_peak_sustained_elapsed']}% >= {SATURATED_PCT}%"
        ),
    )
    if op_class in ("matmul", "conv") and not features.has_tl_dot:
        fired.append(
            (
                4,
                f"tier_b:matmul_without_tensor_core: operator_class {op_class!r} source shows "
                "no tl.dot (scalar accumulation suspected)",
            )
        )

    # latency_bound (reserved): both main throughputs mid-range (stall profile).
    tier_a(
        "mid_throughput_stall",
        (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        5,
        lambda v: (
            MID_THROUGHPUT_RANGE[0]
            <= float(v["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
            <= MID_THROUGHPUT_RANGE[1]
            and MID_THROUGHPUT_RANGE[0]
            <= float(v["dram__throughput.avg.pct_of_peak_sustained_elapsed"])
            <= MID_THROUGHPUT_RANGE[1]
        ),
        lambda v: f"sm and dram throughput both in {MID_THROUGHPUT_RANGE} (stall profile)",
    )

    label = "uncertain"
    ordered = sorted(fired, key=lambda item: (item[0], item[1]))
    priority_to_label = {
        0: "numerical",
        1: "launch_bound",
        2: "parallel_deficit",
        3: "memory_bound",
        4: "compute_bound",
        5: "latency_bound",
    }
    for priority, _signal in ordered:
        if priority in priority_to_label:
            label = priority_to_label[priority]
            break

    signals_fired = tuple(signal for _priority, signal in ordered)
    coverage = _coverage(ncu, signals_fired)
    return BottleneckClass(
        label=label,
        signals_fired=signals_fired,
        signals_missing=tuple(missing),
        coverage=coverage,
    )


def _operator_class(input_data: GeneratorInput) -> str:
    from kernelagent.generators.features import infer_operator_class

    return infer_operator_class(input_data.code_features, input_data.task)


def _coverage(ncu: _NCUView, signals_fired: tuple[str, ...]) -> str:
    if ncu.has_launches and ncu.any_metric_present():
        return "ncu_full" if not ncu.missing_in_all else "ncu_partial"
    if ncu.has_launches and not ncu.any_metric_present():
        return "static_only"  # profiled but nothing measurable: no NCU signal may fire
    has_history = any(signal.startswith("tier_c:") for signal in signals_fired)
    has_static = any(signal.startswith(("tier_a:", "tier_b:")) for signal in signals_fired)
    if has_history and not has_static:
        return "history_only"
    return "static_only"
