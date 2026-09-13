"""Method matching, ranking and the MethodPlan (generator-design §3).

The planner is a pure, deterministic function GeneratorInput -> MethodPlan:
no hidden state, no LLM call, no GPU. Three filters run per catalog entry
(design §3.1):

1. hard capability guards (cc>=9.0 methods on SM89 are refused with the
   explicit reason, never silently skipped);
2. signal matching - a wired method with zero fired signals does not match
   (explicit refusal reason either way);
3. history filtering - a method already promoted on this problem is not
   re-proposed (MVP does not yet generate parameter neighborhoods), and a
   method that failed is demoted by the success prior instead of dropped,
   so honest retries stay possible.

Ranking follows design §3.2: ``score = prior_gain * success_p / est_cost +
exploration`` - a heuristic over methods.yaml ``expected_gain`` (an
UNCALIBRATED prior, mostly from A100/H100 numbers), not a probability.
Hard rules that are not part of the score: numerics-category methods come
first (correctness before performance), and each candidate receives exactly
one plan item (single-factor discipline is enforced at injection time).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from kernelagent.domain.method import Hypothesis, MethodProposal
from kernelagent.generators.catalog import MethodCatalog, MethodEntry
from kernelagent.generators.classify import BottleneckClass, _NCUView, classify_bottleneck
from kernelagent.generators.features import infer_operator_class
from kernelagent.generators.input import AttemptRecord, GeneratorInput, HardwareFacts
from kernelagent.generators.prompt_fragments import render_plan_fragment

TOP_K = 3
_COST_FACTOR_CATEGORIES = frozenset({"tuning_infra", "launch_overhead"})
_GAMMA_EXPLORATION = 0.25
_FAILURE_DEMOTION = 0.3
_BASELINE_GPU_SECONDS = 300.0  # one generate+evaluate round trip (matches the loop's estimate)
_STREAMING_CLASSES = frozenset({"softmax", "attention", "layernorm", "rmsnorm"})
_TILE_CLASSES = frozenset({"matmul", "conv"})

# design §2.2: low-risk default plan when nothing fired (classification=uncertain).
BASELINE_PLAN_METHODS: tuple[str, ...] = (
    "memory.coalesce_contiguous_access",
    "tuning.autotune_space_design",
    "launch.autotune_key_pruning",
)

# Hard capability guards (design §3.1 filter #1). The catalog carries these
# as prose ("SM89 上恒拒绝"); the planner needs them executable. Keys must
# exist in the catalog (pinned by tests).
REQUIRED_COMPUTE_CAPABILITY: dict[str, float] = {
    "pipeline.tma_tensor_descriptors": 9.0,
    "pipeline.warp_specialization": 9.0,
}

# Catalog entries deliberately NOT wired as plan items in the MVP. Every
# catalog method ends up either wired or here - nothing is silently
# skipped (pinned by tests).
UNWIRED_METHOD_REASONS: dict[str, str] = {
    "tuning.do_bench_internal_protocol": (
        "universal timing-boundary constraint: carried as a mandatory GUARDS line in "
        "every prompt fragment, not a separate single-factor round"
    ),
    "tuning.shape_bucket_cache": (
        "requires persisted cross-workload tuning memory; MVP attempts are per-run only"
    ),
    "hardware.sm89_adaptive_tile": (
        "applied as the shared parameter-space/config filter for tile-family methods, "
        "not a standalone single-factor round"
    ),
    "meta.one_factor_per_round": (
        "enforced structurally: each candidate receives exactly one TARGET CHANGE fragment"
    ),
    "meta.failure_feedback_loop": (
        "enforced structurally: AttemptRecord feedback is re-injected into every seed_note"
    ),
}

# Config-space templates (tuning.autotune_space_design / tutorials 03+09
# tables as recorded in methods.yaml). Data only - never executed here.
_GEMM_CONFIGS: tuple[dict, ...] = (
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_stages": 3, "num_warps": 8},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "num_stages": 5, "num_warps": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "num_stages": 5, "num_warps": 2},
)
_ELEMENTWISE_CONFIGS: tuple[dict, ...] = (
    {"BLOCK": 256, "num_warps": 2},
    {"BLOCK": 1024, "num_warps": 4},
    {"BLOCK": 4096, "num_warps": 8},
)
_REDUCTION_CONFIGS: tuple[dict, ...] = (
    {"BLOCK": 512, "num_warps": 2},
    {"BLOCK": 1024, "num_warps": 4},
    {"BLOCK": 2048, "num_warps": 8},
)
_AUTOTUNE_FAMILY_METHODS = frozenset(
    {
        "parallel.occupancy_tuning",
        "parallel.grid_expansion_splitk",
        "pipeline.num_stages_software_pipelining",
        "pipeline.persistent_kernel",
        "launch.autotune_key_pruning",
        "tuning.autotune_space_design",
    }
)


@dataclass(frozen=True, slots=True)
class PlanRefusal:
    """Every non-selected catalog method gets an explicit, readable reason."""

    method_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class MethodPlanItem:
    """One planned single-factor application (design §3.3)."""

    method_id: str
    category: str
    hypothesis: Hypothesis
    proposal: MethodProposal
    prompt_fragment: str
    parameter_space: tuple[dict, ...]
    estimated_gpu_seconds: float
    target_signals: tuple[str, ...]
    evidence_coverage: str
    bottleneck_label: str


@dataclass(frozen=True, slots=True)
class MethodPlan:
    """Plan + honest bookkeeping: classification, every refusal, empty reason."""

    items: tuple[MethodPlanItem, ...]
    classification: BottleneckClass
    refusals: tuple[PlanRefusal, ...]
    empty_reason: str | None = None

    def method_ids(self) -> tuple[str, ...]:
        return tuple(item.method_id for item in self.items)

    def first_actionable(self, attempts: tuple[AttemptRecord, ...]) -> MethodPlanItem | None:
        """Top item not already promoted on this problem (single application
        per problem; failed items are merely demoted, so they stay eligible)."""
        promoted = {a.method_id for a in attempts if a.outcome == "promoted"}
        return next((item for item in self.items if item.method_id not in promoted), None)

    def journal_fields(self) -> dict:
        """JSON-serializable fields for the loop's ``method_plan`` event."""
        return {
            "classification": self.classification.label,
            "evidence_coverage": self.classification.coverage,
            "signals_fired": list(self.classification.signals_fired),
            "signals_missing": list(self.classification.signals_missing),
            "planned_methods": list(self.method_ids()),
            "refusal_count": len(self.refusals),
            "empty_reason": self.empty_reason,
        }


def _prior_gain(expected_gain: str) -> float:
    """low/medium/high -> 1/2/3. Conservative reading of compound priors:
    'unknown' caps at medium (2); a not-applicable entry scores 0 and is in
    any case behind the hardware guard."""
    text = expected_gain.lower()
    if text.startswith("not_applicable"):
        return 0.0
    if "unknown" in text:
        return 2.0
    scale = {"low": 1.0, "medium": 2.0, "high": 3.0}
    values = [value for token, value in scale.items() if token in text]
    return max(values) if values else 1.0


def _success_prior(method_id: str, attempts: tuple[AttemptRecord, ...]) -> float:
    """Compounding demotion per non-infra failure on this problem (§3.2).
    Infrastructure failures are the environment's, not the method's."""
    prior = 1.0
    for attempt in attempts:
        if attempt.method_id == method_id and attempt.outcome == "failed":
            if attempt.failure_class != "infra":
                prior *= _FAILURE_DEMOTION
    return prior


def _parameter_space_for(method_id: str, operator_class: str) -> tuple[dict, ...]:
    if method_id not in _AUTOTUNE_FAMILY_METHODS:
        return ()
    if operator_class in ("matmul", "conv"):
        return _GEMM_CONFIGS
    if operator_class in ("reduction", "softmax", "attention", "layernorm", "rmsnorm"):
        return _REDUCTION_CONFIGS
    return _ELEMENTWISE_CONFIGS


def _fired_tier_a(ncu: _NCUView, metric: str, threshold: float, *, above: bool) -> str | None:
    value = ncu.value(metric)
    if value is None:
        return None
    fired = value >= threshold if above else value < threshold
    if not fired:
        return None
    comparison = ">=" if above else "<"
    return f"tier_a: {metric}={value} {comparison} {threshold}"


def _signal_rules() -> dict[str, Callable[[GeneratorInput], tuple[str, ...]]]:
    """Wired signal predicates, one per matchable catalog method. Each returns
    the fired-signal descriptions (only ever referencing evidence that is
    actually present in the input - the honesty pin in tests)."""

    def op_class(inp: GeneratorInput) -> str:
        return infer_operator_class(inp.code_features, inp.task)

    def ncu_of(inp: GeneratorInput) -> _NCUView:
        return _NCUView(inp.ncu_view)

    def static(*signals: str) -> Callable[[GeneratorInput], tuple[str, ...]]:
        return lambda _inp: signals

    def combine(*parts: Callable[[GeneratorInput], str | None]):
        def rule(inp: GeneratorInput) -> tuple[str, ...]:
            fired = [part(inp) for part in parts]
            return tuple(signal for signal in fired if signal)

        return rule

    def tier_a_metric(metric: str, threshold: float, *, above: bool):
        return lambda inp: _fired_tier_a(ncu_of(inp), metric, threshold, above=above)

    def has_numerical_failure_signal(inp: GeneratorInput) -> str | None:
        from kernelagent.generators.classify import _has_numerical_failure

        fired, signals = _has_numerical_failure(inp)
        return signals[0] if fired else None

    def gated(operator_classes: frozenset[str], text: str):
        """A static family-prior signal that fires only for the given classes."""
        return lambda inp: (
            text.format(oc=op_class(inp)) if op_class(inp) in operator_classes else ()
        )

    def small_grid_for(operator_classes: frozenset[str], label: str):
        """Tier-A small-grid signal gated on a tile/reduction family."""

        def rule(inp: GeneratorInput) -> str | None:
            value = _NCUView(inp.ncu_view).value("launch__grid_size")
            sm_count = inp.hardware.sm_count
            if value is None or sm_count is None or op_class(inp) not in operator_classes:
                return None
            if value < 2.0 * sm_count:
                return (
                    f"tier_a: launch__grid_size={int(value)} below 2.0*sm_count={2.0 * sm_count} "
                    f"on a {label} workload - parallelism deficit"
                )
            return None

        return rule

    def fuse_chain_rule(inp: GeneratorInput) -> str | None:
        chain_classes = _STREAMING_CLASSES | {"elementwise", "reduction"}
        if inp.code_features.forward_call_count >= 2 and op_class(inp) in chain_classes:
            return (
                f"tier_b: forward issues {inp.code_features.forward_call_count} calls in the "
                f"{op_class(inp)!r} family - intermediate tensors round-trip DRAM"
            )
        return None

    def boundary_masking_rule(inp: GeneratorInput) -> str | None:
        markers = ("out of bounds", "memcheck", "越界", "illegal memory")
        for attempt in inp.attempts:
            if attempt.failure_class == "correctness" and any(
                marker in attempt.note.lower() for marker in markers
            ):
                return (
                    "tier_c: out-of-bounds/memcheck-style failure evidence - boundary "
                    "masking is the first suspect"
                )
        return None

    def stable_softmax_rule(inp: GeneratorInput) -> tuple[str, ...]:
        if op_class(inp) in ("softmax", "attention"):
            return (
                "tier_b: softmax-family operator - exp stability is a hard prerequisite, "
                "not an optional optimization",
            )
        signal = has_numerical_failure_signal(inp)
        return (signal,) if signal else ()

    rules: dict[str, Callable[[GeneratorInput], tuple[str, ...]]] = {
        # ---- memory_layout -------------------------------------------------
        "memory.coalesce_contiguous_access": combine(
            lambda inp: (
                (
                    f"tier_b: operator_class {op_class(inp)!r} benefits from coalesced "
                    "inner-dimension access"
                )
                if op_class(inp) in ("elementwise", "reduction", "embedding")
                else None
            ),
            tier_a_metric(
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                60.0,
                above=True,
            ),
        ),
        "memory.vectorize_wide_loads": lambda inp: (
            (
                f"tier_b: dtypes {inp.code_features.dtypes} with operator_class "
                f"{op_class(inp)!r} allow 128-bit vectorized accesses"
            )
            if ({"float16", "bfloat16"} & set(inp.code_features.dtypes))
            and op_class(inp) in ("elementwise", "matmul", "conv")
            else ()
        ),
        "memory.l2_tile_swizzle": gated(
            _TILE_CLASSES,
            "tier_b: operator_class {oc!r} - grouped program ordering lifts L2 tile reuse",
        ),
        # ---- fusion ----------------------------------------------------------
        "memory.fuse_elementwise_chain": fuse_chain_rule,
        "memory.gemm_epilogue_fusion": lambda inp: (
            "tier_b: matmul followed by pointwise work - epilogue fusion applies"
            if op_class(inp) == "matmul" and inp.code_features.forward_call_count >= 2
            else ()
        ),
        "memory.eliminate_intermediate_materialization": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} materializes a product-shaped "
            "intermediate in its naive form"
            if op_class(inp) in ("softmax", "attention")
            else ()
        ),
        # ---- resource_tuning -------------------------------------------------
        "memory.register_pressure_reduction": combine(
            tier_a_metric("launch__occupancy_limit_registers", 60.0, above=True),
            tier_a_metric("sm__warps_active.avg.pct_of_peak_sustained_active", 30.0, above=False),
        ),
        # ---- parallelism -----------------------------------------------------
        "parallel.occupancy_tuning": lambda inp: (
            f"tier_b: no autotune in the reference and operator_class {op_class(inp)!r} - "
            "fixed launch shape suspected"
            if not inp.code_features.has_autotune
            and op_class(inp) in ("elementwise", "matmul", "conv", "reduction", "embedding")
            else ()
        ),
        "parallel.grid_expansion_splitk": combine(
            small_grid_for(_TILE_CLASSES | {"reduction"}, "tile/reduction"),
        ),
        "parallel.grid_stride_elementwise": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - grid-stride flattening applies"
            if op_class(inp) == "elementwise"
            else ()
        ),
        "parallel.batched_grouped_fusion": lambda inp: (
            "tier_b: batched/grouped family hint in the source"
            if "batched" in inp.task.problem_name.lower()
            or "grouped" in inp.task.problem_name.lower()
            or "multi_head" in inp.task.problem_name.lower()
            else ()
        ),
        # ---- compute_precision ---------------------------------------------
        "compute.tensor_core_dot": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} without tl.dot - scalar FMA "
            "accumulation suspected"
            if op_class(inp) in _TILE_CLASSES and not inp.code_features.has_tl_dot
            else ()
        ),
        "compute.tf32_bf16_within_contract": lambda inp: (
            "tier_b: fp32 linear-algebra workload - in-contract TF32 path may apply"
            if op_class(inp) in _TILE_CLASSES and "float32" in inp.code_features.dtypes
            else ()
        ),
        "compute.fp8_experimental": lambda inp: (
            f"tier_b: fp8 dtypes {inp.code_features.dtypes} present"
            if any(dtype.startswith("float8") for dtype in inp.code_features.dtypes)
            else ()
        ),
        "compute.algebraic_strength_reduction": lambda inp: (
            "tier_b: transcendental calls in the hot path (exp/log/pow/sqrt)"
            if inp.code_features.uses_transcendental
            else ()
        ),
        # ---- pipelining ------------------------------------------------------
        "pipeline.num_stages_software_pipelining": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} has an inner loop and no autotuned "
            "stage depth"
            if op_class(inp) in ("matmul", "conv", "reduction")
            and not inp.code_features.has_autotune
            else ()
        ),
        "pipeline.persistent_kernel": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - tail-wave/launch-count hypothesis "
            "applies (shape-dependent, not a universal upgrade)"
            if op_class(inp) in _TILE_CLASSES
            else ()
        ),
        "pipeline.tma_tensor_descriptors": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} could use descriptor batched accesses"
            if op_class(inp) in _TILE_CLASSES
            else ()
        ),
        "pipeline.warp_specialization": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} has producer/consumer overlap potential"
            if op_class(inp) in _TILE_CLASSES
            else ()
        ),
        # ---- launch_overhead -------------------------------------------------
        "launch.multi_kernel_merge": lambda inp: (
            f"tier_b: forward issues {inp.code_features.forward_call_count} kernel-like "
            "calls - launch-gap hypothesis"
            if inp.code_features.forward_call_count >= 3
            else ()
        ),
        "launch.host_overhead_cleanup": lambda inp: (
            f"tier_b: {inp.code_features.host_allocations} host-side allocation(s) and/or "
            "device sync calls in forward"
            if inp.code_features.host_allocations >= 1 or inp.code_features.has_sync_call
            else ()
        ),
        "launch.autotune_key_pruning": lambda inp: (
            "tier_b: @triton.autotune present - key/prune budget control applies"
            if inp.code_features.has_autotune
            else ()
        ),
        # ---- reduction ---------------------------------------------------------
        "reduction.two_pass_deterministic": lambda inp: (
            (
                f"tier_b: operator_class {op_class(inp)!r} - parallel deterministic "
                "reduction structure applies"
            )
            if op_class(inp) in ("reduction", "layernorm", "rmsnorm", "softmax")
            or inp.code_features.uses_atomic
            else ()
        ),
        "reduction.warp_block_staged": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - block-shape/reduction-axis layout applies"
            if op_class(inp) in ("layernorm", "rmsnorm", "softmax")
            else ()
        ),
        "reduction.online_streaming": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - online max/sum removes the "
            "pre-pass over the reduction axis"
            if op_class(inp) in _STREAMING_CLASSES
            else ()
        ),
        "reduction.split_large_axis": combine(
            small_grid_for(frozenset({"reduction"}), "reduction"),
        ),
        # ---- algorithmic -------------------------------------------------------
        "algorithm.flash_tile_streaming": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - flash tile streaming hypothesis"
            if op_class(inp) == "attention"
            else ()
        ),
        "algorithm.masked_early_exit": lambda inp: (
            "tier_b: triangular/structured mask detected in the source - block-level skip applies"
            if inp.code_features.has_triangular_mask
            else ()
        ),
        "algorithm.gather_scatter_optimization": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} - index-access family"
            if op_class(inp) == "embedding"
            else ()
        ),
        # ---- tuning_infra ------------------------------------------------------
        "tuning.autotune_space_design": lambda inp: (
            f"tier_b: operator_class {op_class(inp)!r} has a prior config-space template"
            if inp.code_features.has_autotune or op_class(inp) != "unknown"
            else ()
        ),
        # ---- numerics ----------------------------------------------------------
        "numerics.stable_softmax_logsumexp": stable_softmax_rule,
        "numerics.fp32_accumulators": lambda inp: (
            (
                f"tier_b: dtypes {inp.code_features.dtypes} with reduction/matmul work - "
                "fp32 accumulation is a hard prerequisite"
            )
            if ({"float16", "bfloat16"} & set(inp.code_features.dtypes))
            and op_class(inp) in _TILE_CLASSES | {"reduction"} | _STREAMING_CLASSES
            else ()
        ),
        "numerics.boundary_masking": boundary_masking_rule,
    }
    return rules


_WIRED_RULES = _signal_rules()


def _capability_guard(method_id: str, hardware: HardwareFacts) -> str | None:
    """Refusal reason when a hard capability guard blocks the method, else None."""
    required = REQUIRED_COMPUTE_CAPABILITY.get(method_id)
    if required is None:
        return None
    cc = hardware.compute_capability
    if cc is None:
        return (
            f"device compute capability MISSING (cannot verify >= {required}); "
            "refusing conservatively - capability is never assumed"
        )
    if cc < required:
        return f"device cc {cc} < required {required} (SM90+ feature)"
    return None


def _canonical_space_sha256(parameter_space: tuple[dict, ...]) -> str:
    payload = json.dumps(list(parameter_space), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MethodPlanner:
    """Deterministic GeneratorInput -> MethodPlan (no LLM, no GPU, no state)."""

    catalog: MethodCatalog
    top_k: int = TOP_K

    def plan(self, input_data: GeneratorInput) -> MethodPlan:
        classification = classify_bottleneck(input_data)
        operator_class = infer_operator_class(input_data.code_features, input_data.task)
        refusals: list[PlanRefusal] = []
        candidates: list[tuple[float, int, MethodEntry, tuple[str, ...]]] = []

        for index, entry in enumerate(self.catalog.methods):
            guard = _capability_guard(entry.id, input_data.hardware)
            if guard is not None:
                refusals.append(PlanRefusal(entry.id, guard))
                continue
            if entry.id in UNWIRED_METHOD_REASONS:
                refusals.append(PlanRefusal(entry.id, UNWIRED_METHOD_REASONS[entry.id]))
                continue
            rule = _WIRED_RULES.get(entry.id)
            if rule is None:  # wired table and catalog drifted apart - fail loud
                raise ValueError(
                    f"catalog method {entry.id!r} has no signal rule and no unwired reason; "
                    "extend generators.plan"
                )
            fired = rule(input_data)
            # Rules may return a bare signal string or None; normalize so
            # every downstream field (target_signals, hypothesis observations)
            # is a tuple of strings.
            if fired is None:
                fired = ()
            elif isinstance(fired, str):
                fired = (fired,)
            else:
                fired = tuple(fired)
            if not fired:
                refusals.append(
                    PlanRefusal(entry.id, "no applicability signal matched the current evidence")
                )
                continue
            promoted = any(
                attempt.method_id == entry.id and attempt.outcome == "promoted"
                for attempt in input_data.attempts
            )
            if promoted:
                refusals.append(
                    PlanRefusal(
                        entry.id,
                        "already confirmed promoted on this problem; MVP does not propose "
                        "the parameter-space neighborhood",
                    )
                )
                continue
            base = (
                _prior_gain(entry.expected_gain)
                * _success_prior(entry.id, input_data.attempts)
                / (1.5 if entry.category in _COST_FACTOR_CATEGORIES else 1.0)
            )
            candidates.append((base, index, entry, fired))

        # Exploration: the first-seen family keeps its bonus (design §3.2).
        candidates.sort(key=lambda item: (-item[0], item[1]))
        seen: set[str] = set()
        scored: list[tuple[float, int, MethodEntry, tuple[str, ...], bool]] = []
        for base, index, entry, fired_signal in candidates:
            score = base + (_GAMMA_EXPLORATION if entry.category not in seen else 0.0)
            seen.add(entry.category)
            demoted = _success_prior(entry.id, input_data.attempts) < 1.0
            scored.append((score, index, entry, fired_signal, demoted))

        # Correctness before performance (hard rule, not part of the score) -
        # but a numerics hypothesis that already failed on this problem yields
        # its front slot: the feedback loop outranks the family prior.
        scored.sort(
            key=lambda item: (
                0 if item[2].category == "numerics" and not item[4] else 1,
                -item[0],
                item[1],
            )
        )
        chosen = scored[: self.top_k]
        # Matched but ranked out of the plan: still gets an explicit, readable
        # reason - nothing about a method's disposition is left implicit.
        for score, _index, entry, _fired_signal, _demoted in scored[self.top_k :]:
            refusals.append(
                PlanRefusal(
                    entry.id,
                    f"matched but ranked below top-{self.top_k} (score {score:.3f}); "
                    "it will resurface when higher-ranked hypotheses fail",
                )
            )
        items = tuple(
            self._build_item(entry, fired_signal, input_data, classification, operator_class)
            for _score, _index, entry, fired_signal, _demoted in chosen
        )

        items, refusals, empty_reason = self._fallback_baseline(
            items, refusals, classification, input_data, operator_class
        )
        if not items and empty_reason is None:
            head = (
                f"{refusals[0].method_id}: {refusals[0].reason}"
                if refusals
                else "no candidate methods"
            )
            empty_reason = f"no method could be planned ({len(refusals)} refusals; first: {head})"
        return MethodPlan(
            items=items,
            classification=classification,
            refusals=tuple(refusals),
            empty_reason=empty_reason,
        )

    def _fallback_baseline(
        self,
        items: tuple[MethodPlanItem, ...],
        refusals: list[PlanRefusal],
        classification: BottleneckClass,
        input_data: GeneratorInput,
        operator_class: str,
    ) -> tuple[tuple[MethodPlanItem, ...], list[PlanRefusal], str | None]:
        """design §2.2 (3): when nothing fired (uncertain), answer with the
        low-risk baseline trio instead of an empty plan; if even those are
        exhausted, say so explicitly - never a silent empty list."""
        if items or classification.label != "uncertain":
            return items, refusals, None
        catalog_by_id = self.catalog.by_id()
        baseline_items: list[MethodPlanItem] = []
        for method_id in BASELINE_PLAN_METHODS:
            entry = catalog_by_id[method_id]
            guard = _capability_guard(method_id, input_data.hardware)
            if guard is not None:
                refusals.append(PlanRefusal(method_id, guard))
                continue
            promoted = any(
                attempt.method_id == method_id and attempt.outcome == "promoted"
                for attempt in input_data.attempts
            )
            if promoted:
                refusals.append(
                    PlanRefusal(
                        method_id,
                        "baseline-plan method already promoted on this problem",
                    )
                )
                continue
            signal = (
                "tier_b: baseline low-risk plan "
                "(classification uncertain: no bottleneck signal fired)"
            )
            baseline_items.append(
                self._build_item(entry, (signal,), input_data, classification, operator_class)
            )
        if baseline_items:
            return tuple(baseline_items), refusals, None
        return (
            (),
            refusals,
            "classification uncertain and every baseline-plan method was already promoted "
            "or refused - no honest plan remains",
        )

    def _build_item(
        self,
        entry: MethodEntry,
        fired_signals: tuple[str, ...],
        input_data: GeneratorInput,
        classification: BottleneckClass,
        operator_class: str,
    ) -> MethodPlanItem:
        parameter_space = _parameter_space_for(entry.id, operator_class)
        falsification = tuple(entry.risks_preconditions) + (
            "trusted evaluator formal timing shows no improvement beyond noise; "
            "the hypothesis is then falsified for this problem",
        )
        hypothesis = Hypothesis(
            statement=(
                f"{entry.name}: {entry.mechanism} Expected gain {entry.expected_gain} "
                "(uncalibrated prior; only the trusted evaluator's formal results confirm it)"
            ),
            predicted_observations=fired_signals,
            falsification_conditions=falsification,
        )
        proposal = MethodProposal(
            method_id=entry.id,
            method_version=self.catalog.version,
            hypothesis=hypothesis,
            parameter_space_sha256=_canonical_space_sha256(parameter_space),
            estimated_gpu_seconds=(
                _BASELINE_GPU_SECONDS * 1.5
                if entry.category in _COST_FACTOR_CATEGORIES
                else _BASELINE_GPU_SECONDS
            ),
            target_workload_ids=(input_data.task.task_id,),
        )
        fragment = render_plan_fragment(
            method_id=entry.id,
            statement=hypothesis.statement,
            requirements=entry.triton_howto,
            parameter_space=parameter_space,
            evidence_coverage=classification.coverage,
            bottleneck_label=classification.label,
        )
        return MethodPlanItem(
            method_id=entry.id,
            category=entry.category,
            hypothesis=hypothesis,
            proposal=proposal,
            prompt_fragment=fragment,
            parameter_space=parameter_space,
            estimated_gpu_seconds=proposal.estimated_gpu_seconds,
            target_signals=fired_signals,
            evidence_coverage=classification.coverage,
            bottleneck_label=classification.label,
        )
