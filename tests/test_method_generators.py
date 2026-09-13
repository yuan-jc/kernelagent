"""Method generator MVP (generator-design §6.1/6.2 offline regression set).

Everything here runs offline with fixed inputs: bottleneck classification,
three-tier signals, hard capability guards, prompt injection with stable
request identity, honest degradation on missing evidence, strict catalog
loading, and single-factor discipline. No GPU and no model call is made or
implied; live validation is a separate, NOT_RUN-until-executed step.
"""

import hashlib
import json
from pathlib import Path

import pytest

from kernelagent.adapters.models.client import ModelRequest, ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator, build_request
from kernelagent.generators.catalog import DATA_PATH, CatalogError, load_catalog, load_default
from kernelagent.generators.classify import (
    CORE_TIER_A_METRICS,
    BottleneckClass,
    classify_bottleneck,
)
from kernelagent.generators.features import (
    extract_code_features,
    infer_operator_class,
    task_profile_from_problem,
)
from kernelagent.generators.input import AttemptRecord, GeneratorInput, HardwareFacts
from kernelagent.generators.plan import (
    BASELINE_PLAN_METHODS,
    REQUIRED_COMPUTE_CAPABILITY,
    UNWIRED_METHOD_REASONS,
    MethodPlanner,
)
from kernelagent.generators.prompt_fragments import compose_seed_note
from kernelagent.methods import TaskProfile
from kernelagent.optimization import optimize

RESEARCH_YAML = (
    Path(__file__).resolve().parents[1] / "research" / "optimization-methods" / "methods.yaml"
)


def _hw(**overrides) -> HardwareFacts:
    facts = HardwareFacts(compute_capability=8.9, sm_count=24)
    fields = {"compute_capability": facts.compute_capability, "sm_count": facts.sm_count}
    return HardwareFacts(**{**fields, **overrides})


def _view(launch_metrics: list[dict[str, float]], missing: list[str] | None = None) -> dict:
    launches = [
        {"metrics": {name: {"value": value, "unit": None} for name, value in m.items()}}
        for m in launch_metrics
    ]
    return {
        "source": "ncu_profile",
        "launch_count": len(launches),
        "launches": launches,
        "catalog_metrics_missing_in_all_launches": list(missing or ()),
        "note": "missing metrics are absent, never zero-filled",
    }


SOFTMAX_TASK = task_profile_from_problem("48_Softmax.py", 1, 48)
MATMUL_TASK = task_profile_from_problem("03_Matmul.py", 1, 3)
ELEMENTWISE_TASK = task_profile_from_problem("07_FusedChain.py", 2, 7)
UNKNOWN_TASK = task_profile_from_problem("99_Mystery.py", 3, 99)

SOFTMAX_SOURCE = (
    "import torch\n"
    "class Model:\n"
    "    def forward(self, x):\n"
    "        return torch.softmax(x, dim=1)\n"
)
MATMUL_SOURCE = (
    "import torch\n"
    "class Model:\n"
    "    def forward(self, a, b):\n"
    "        return torch.matmul(a, b).to(torch.float16)\n"
)


def _input(task=UNKNOWN_TASK, source="# nothing\n", ncu=None, attempts=(), hardware=None):
    return GeneratorInput(
        task=task,
        code_features=extract_code_features(source),
        attempts=tuple(attempts),
        ncu_view=ncu,
        hardware=hardware if hardware is not None else _hw(),
    )


# --- catalog -----------------------------------------------------------------


def test_catalog_loads_38_methods_strictly():
    catalog = load_default()
    assert len(catalog) == 38, "the frozen A2 catalog must ship all 38 methods"
    assert catalog.version == "0.1"
    assert len({entry.id for entry in catalog.methods}) == 38
    for entry in catalog.methods:
        assert entry.mechanism and entry.expected_gain
        assert any(
            entry.signals_for_tier(tier) for tier in ("tier_a_ncu", "tier_b_task", "tier_c_history")
        )
        assert entry.triton_howto


def test_catalog_freeze_provenance_matches_research_source():
    catalog = load_default()
    assert catalog.frozen_from == "research/optimization-methods/methods.yaml"
    if not RESEARCH_YAML.exists():  # pragma: no cover - sdist-only environments
        pytest.skip("research methods.yaml not present in this environment")
    digest = hashlib.sha256(RESEARCH_YAML.read_bytes()).hexdigest()
    assert catalog.frozen_source_sha256 == digest, (
        "frozen JSON copy is stale relative to research/optimization-methods/methods.yaml; "
        "re-freeze it (see generators/catalog.py docstring)"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["methods"][3].pop("mechanism"),
        lambda p: p["methods"][1]["applicability_signals"].update({"tier_x": ["nope"]}),
        lambda p: p["methods"][2].update({"id": p["methods"][0]["id"]}),
        lambda p: p["meta"].pop("version"),
        lambda p: p.update({"methods": []}),
        lambda p: p["methods"][5].update({"triton_howto": "not-a-list"}),
    ],
)
def test_catalog_rejects_corrupt_entries(tmp_path, mutate):
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(corrupt)


def test_tier_a_metric_names_mirror_ncu_metric_catalog():
    from kernelagent.adapters.profiling.ncu import MetricCatalog

    assert set(CORE_TIER_A_METRICS) == set(MetricCatalog.default().entries), (
        "generator tier-A signals must reference exactly the NCU MetricCatalog fields"
    )


# --- features and operator class ---------------------------------------------


def test_task_profile_infers_category_from_problem_name():
    assert task_profile_from_problem("40_LayerNorm.py", 1, 40).category == "layernorm"
    assert task_profile_from_problem("03_Matmul.py", 1, 3).category == "matmul"
    assert task_profile_from_problem("06_Conv2d.py", 1, 6).category == "conv"
    unknown = task_profile_from_problem("99_Mystery.py", 3, 99)
    assert unknown.category == "unknown", "'unknown' is the honest outcome, never a guess"


def test_code_features_extraction_signals():
    source = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.autotune(configs=[], key=['N'])\n"
        "def k(x):\n"
        "    tl.dot(a, b)\n"
        "class Model:\n"
        "    def forward(self, x):\n"
        "        y = torch.zeros_like(x)\n"
        "        z = torch.triu(x)\n"
        "        return y.to(torch.float16), z.item()\n"
    )
    features = extract_code_features(source)
    assert features.parses
    assert features.uses_triton and features.has_autotune and features.has_tl_dot
    assert features.host_allocations == 1
    assert features.has_triangular_mask
    assert features.has_sync_call
    assert "float16" in features.dtypes


def test_unparseable_source_degrades_honestly():
    features = extract_code_features("def broken(:\n")
    assert not features.parses
    assert features.forward_call_count == 0
    assert infer_operator_class(features, UNKNOWN_TASK) == "unknown"


# --- bottleneck classification ------------------------------------------------


def test_classification_priority_and_missing_semantics():
    # numerical outranks everything else (correctness first).
    numerical = classify_bottleneck(
        _input(
            task=SOFTMAX_TASK,
            source=SOFTMAX_SOURCE,
            ncu=_view([{"dram__throughput.avg.pct_of_peak_sustained_elapsed": 90.0}]),
            attempts=(
                AttemptRecord(None, "correctness_pro", "correctness", "output contains nan"),
            ),
        )
    )
    assert numerical.label == "numerical"
    # DRAM saturated alone classifies memory_bound.
    memory = classify_bottleneck(
        _input(
            task=UNKNOWN_TASK,
            ncu=_view([{"dram__throughput.avg.pct_of_peak_sustained_elapsed": 78.0}]),
        )
    )
    assert memory.label == "memory_bound"
    # launch_bound outranks memory: short kernels + a multi-call chain.
    launch = classify_bottleneck(
        _input(
            task=ELEMENTWISE_TASK,
            source="import torch\nclass Model:\n    def forward(self, x):\n"
            "        return torch.relu(torch.sigmoid(x)), torch.tanh(x)\n",
            ncu=_view(
                [
                    {
                        "gpu__time_duration.sum": 5.0,
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 70.0,
                    }
                ]
                * 4
            ),
        )
    )
    assert launch.label == "launch_bound"


@pytest.mark.parametrize(
    ("case", "expected_label"),
    [
        # design §6.1 row: elementwise, many kernels, short durations.
        ("elementwise_multi_kernel_short", "launch_bound"),
        # design §6.1 row: matmul without tl.dot (static only).
        ("matmul_without_tl_dot", "compute_bound"),
        # design §6.1 row: matmul with grid 16 on a 24-SM device.
        ("matmul_small_grid", "parallel_deficit"),
        # design §6.1 row: softmax with a NaN failure in history.
        ("softmax_nan_history", "numerical"),
        # design §5 walkthrough: dram 78%, single launch.
        ("dram_saturated", "memory_bound"),
        # reserved class: both throughputs mid-range (stall profile).
        ("mid_throughput_stall", "latency_bound"),
        # nothing fires at all.
        ("no_signal", "uncertain"),
    ],
)
def test_bottleneck_classification_table(case, expected_label):
    if case == "elementwise_multi_kernel_short":
        inp = _input(
            task=ELEMENTWISE_TASK,
            source="import torch\nclass Model:\n    def forward(self, x):\n"
            "        return torch.relu(x), torch.sigmoid(x), torch.tanh(x)\n",
            ncu=_view([{"gpu__time_duration.sum": 6.0}] * 4),
        )
    elif case == "matmul_without_tl_dot":
        inp = _input(task=MATMUL_TASK, source=MATMUL_SOURCE)
    elif case == "matmul_small_grid":
        inp = _input(task=MATMUL_TASK, source=MATMUL_SOURCE, ncu=_view([{"launch__grid_size": 16}]))
    elif case == "softmax_nan_history":
        inp = _input(
            task=SOFTMAX_TASK,
            source=SOFTMAX_SOURCE,
            attempts=(AttemptRecord(None, "correctness_pro", "correctness", "nan in output"),),
        )
    elif case == "dram_saturated":
        inp = _input(
            task=UNKNOWN_TASK,
            ncu=_view(
                [
                    {
                        "launch__grid_size": 4096,
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 78.0,
                    }
                ]
            ),
        )
    elif case == "mid_throughput_stall":
        inp = _input(
            task=UNKNOWN_TASK,
            ncu=_view(
                [
                    {
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 55.0,
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
                    }
                ]
            ),
        )
    else:
        inp = _input()
    assert classify_bottleneck(inp).label == expected_label


def test_missing_metrics_recorded_never_fired():
    all_missing = _view([{}], missing=list(CORE_TIER_A_METRICS))
    result = classify_bottleneck(_input(task=SOFTMAX_TASK, source=SOFTMAX_SOURCE, ncu=all_missing))
    assert result.coverage == "static_only", "profiled-but-unmeasurable must not claim ncu coverage"
    assert result.signals_missing, "MISSING metrics must be recorded, not dropped"
    assert not any(signal.startswith("tier_a:") for signal in result.signals_fired), (
        "no tier-A signal may fire from MISSING metrics"
    )


def test_coverage_levels():
    full = _view([{metric: 1.0 for metric in CORE_TIER_A_METRICS}], missing=[])
    assert classify_bottleneck(_input(ncu=full)).coverage == "ncu_full"
    partial = _view(
        [{"launch__grid_size": 4096}],
        missing=["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    )
    assert classify_bottleneck(_input(ncu=partial)).coverage == "ncu_partial"
    assert classify_bottleneck(_input(ncu=None)).coverage == "static_only"
    history_only = classify_bottleneck(
        _input(
            attempts=(AttemptRecord(None, "correctness_pro", "correctness", "nan"),),
            ncu=None,
            task=UNKNOWN_TASK,
        )
    )
    assert history_only.coverage == "history_only"


# --- hardware guards ----------------------------------------------------------


@pytest.mark.parametrize("method_id", sorted(REQUIRED_COMPUTE_CAPABILITY))
def test_sm90_methods_refused_on_sm89_with_explicit_reason(method_id):
    planner = MethodPlanner(load_default())
    plan = planner.plan(_input(task=MATMUL_TASK, source=MATMUL_SOURCE))
    refusals = {refusal.method_id: refusal.reason for refusal in plan.refusals}
    assert method_id in refusals, "SM90-only methods must never be planned on SM89"
    assert "8.9" in refusals[method_id] and "9.0" in refusals[method_id]
    assert method_id not in plan.method_ids()


def test_unknown_compute_capability_refuses_conservatively():
    planner = MethodPlanner(load_default())
    plan = planner.plan(
        _input(task=MATMUL_TASK, source=MATMUL_SOURCE, hardware=_hw(compute_capability=None))
    )
    refusals = {refusal.method_id: refusal.reason for refusal in plan.refusals}
    assert "MISSING" in refusals["pipeline.tma_tensor_descriptors"]


# --- matching, ranking, discipline ---------------------------------------------


def test_wired_and_unwired_cover_whole_catalog():
    from kernelagent.generators.plan import _WIRED_RULES

    catalog_ids = {entry.id for entry in load_default().methods}
    wired = set(_WIRED_RULES)
    unwired = set(UNWIRED_METHOD_REASONS)
    assert not wired - catalog_ids, "wired rules must reference existing catalog ids"
    assert not (catalog_ids - wired - unwired), (
        "every catalog method is wired or has an explicit reason - nothing silently skipped"
    )


def test_mvp_priority_categories_are_selectable():
    """A2 conclusion: fusion/memory_layout, numerics and tuning_infra methods
    must be reachable and injectable, not just loaded."""
    catalog = load_default()
    from kernelagent.generators.plan import _WIRED_RULES

    wired_categories = {}
    for entry in catalog.methods:
        if entry.id in _WIRED_RULES:
            wired_categories.setdefault(entry.category, []).append(entry.id)
    for category in ("fusion", "memory_layout", "numerics", "tuning_infra"):
        assert wired_categories.get(category), f"{category} must have at least one wired method"

    # fusion/memory_layout reachable from an elementwise multi-kernel problem,
    # and injectable as exactly one single-factor fragment.
    fusion_plan = MethodPlanner(catalog).plan(
        _input(
            task=ELEMENTWISE_TASK,
            source="import torch\nclass Model:\n    def forward(self, x):\n"
            "        return torch.relu(x), torch.sigmoid(x), torch.tanh(x)\n",
            ncu=_view([{"gpu__time_duration.sum": 6.0}] * 4),
        )
    )
    assert "memory.fuse_elementwise_chain" in fusion_plan.method_ids()
    assert compose_seed_note("", fusion_plan.items[0].prompt_fragment).count("TARGET CHANGE") == 1

    # tuning_infra reachable from the uncertain baseline plan (design §2.2's
    # low-risk trio), and injectable as single-factor fragments.
    baseline_plan = MethodPlanner(catalog).plan(_input())
    assert baseline_plan.classification.label == "uncertain"
    baseline_categories = {item.category for item in baseline_plan.items}
    assert "tuning_infra" in baseline_categories
    assert "memory_layout" in baseline_categories
    assert all(item.prompt_fragment.count("TARGET CHANGE") == 1 for item in baseline_plan.items)

    # numerics reachable and ranked first on correctness evidence.
    numerics_plan = MethodPlanner(catalog).plan(
        _input(
            task=SOFTMAX_TASK,
            source=SOFTMAX_SOURCE,
            attempts=(AttemptRecord(None, "correctness_pro", "correctness", "nan detected"),),
        )
    )
    assert numerics_plan.items[0].method_id == "numerics.stable_softmax_logsumexp"


def test_failed_method_demoted_promoted_method_refused():
    planner = MethodPlanner(load_default())
    inp = _input(task=MATMUL_TASK, source=MATMUL_SOURCE)
    baseline = planner.plan(inp)
    top = baseline.method_ids()[0]

    failed_history = (AttemptRecord(top, "evaluate", "correctness", "assert allclose failed"),)
    demoted = planner.plan(
        GeneratorInput(
            task=inp.task,
            code_features=inp.code_features,
            attempts=failed_history,
            hardware=inp.hardware,
        )
    )
    assert demoted.method_ids()[0] != top, "the failed method must lose the top slot"
    refusals = {refusal.method_id: refusal.reason for refusal in demoted.refusals}
    if top in demoted.method_ids():
        assert demoted.method_ids().index(top) > 0
    else:
        assert "ranked below" in refusals[top], "a ranked-out method keeps an explicit reason"

    promoted_history = (AttemptRecord(top, "confirm", "none", "promoted", outcome="promoted"),)
    refused_plan = planner.plan(
        GeneratorInput(
            task=inp.task,
            code_features=inp.code_features,
            attempts=promoted_history,
            hardware=inp.hardware,
        )
    )
    assert top not in refused_plan.method_ids()
    reason = {r.method_id: r.reason for r in refused_plan.refusals}[top]
    assert "promoted" in reason


def test_plan_is_deterministic_and_bounded():
    planner = MethodPlanner(load_default(), top_k=3)
    inp = _input(task=MATMUL_TASK, source=MATMUL_SOURCE)
    first = planner.plan(inp)
    assert first == planner.plan(inp)
    assert len(first.items) == 3
    assert first.method_ids() == tuple(dict.fromkeys(first.method_ids()))
    for item in first.items:
        assert item.evidence_coverage == first.classification.coverage
        assert item.proposal.method_version == load_default().version
        assert item.proposal.parameter_space_sha256
        assert item.estimated_gpu_seconds > 0
        assert item.hypothesis.predicted_observations == item.target_signals


def test_uncertain_classification_yields_baseline_plan():
    plan = MethodPlanner(load_default()).plan(_input())
    assert plan.classification.label == "uncertain"
    assert plan.method_ids() == BASELINE_PLAN_METHODS
    assert all("uncertain" in item.target_signals[0] for item in plan.items)


def test_empty_plan_carries_explicit_reason():
    planner = MethodPlanner(load_default(), top_k=3)
    inp = _input()  # uncertain -> baseline trio
    attempts = tuple(
        AttemptRecord(method_id, "confirm", "none", "promoted", outcome="promoted")
        for method_id in BASELINE_PLAN_METHODS
    )
    exhausted = GeneratorInput(
        task=inp.task, code_features=inp.code_features, attempts=attempts, hardware=inp.hardware
    )
    plan = planner.plan(exhausted)
    assert not plan.items
    assert plan.empty_reason, "an empty plan must say why, never pass silently"
    assert "promoted" in plan.empty_reason


# --- honesty of signals ---------------------------------------------------------


def test_target_signals_never_reference_missing_metrics():
    planner = MethodPlanner(load_default())
    ncu = _view(
        [{"launch__grid_size": 16}],
        missing=["gpu__time_duration.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    )
    plan = planner.plan(_input(task=MATMUL_TASK, source=MATMUL_SOURCE, ncu=ncu))
    for item in plan.items:
        for signal in item.target_signals:
            for metric in ("gpu__time_duration.sum", "dram__throughput"):
                assert metric not in signal, (
                    f"{item.method_id} target signal references a MISSING metric: {signal}"
                )


# --- prompt fragments and request identity ----------------------------------------


def test_fragment_single_factor_and_mandatory_guards():
    planner = MethodPlanner(load_default())
    plan = planner.plan(_input(task=ELEMENTWISE_TASK, source=MATMUL_SOURCE))
    fragment = plan.items[0].prompt_fragment
    assert fragment.count("TARGET CHANGE") == 1, "exactly one authorized change per round"
    assert "apply ONLY this change" in fragment
    assert "numerics.boundary_masking" in fragment
    assert "numerics.fp32_accumulators" in fragment
    assert "tuning.do_bench_internal_protocol" in fragment
    assert "Do not measure or report performance yourself" in fragment


def test_autotune_space_rendered_only_when_present():
    plan = MethodPlanner(load_default()).plan(_input())  # uncertain -> baseline trio
    without_space = plan.items[0]  # memory.coalesce: not a config-search method
    assert without_space.parameter_space == ()
    assert "not a config search" in without_space.prompt_fragment
    with_space = next(item for item in plan.items if item.parameter_space)
    assert "AUTOTUNE SPACE" in with_space.prompt_fragment
    assert "config:" in with_space.prompt_fragment
    # GEMM-family methods draw from the tutorials/03 config table.
    matmul_plan = MethodPlanner(load_default()).plan(
        _input(
            task=MATMUL_TASK,
            source=MATMUL_SOURCE,
            ncu=_view([{"launch__grid_size": 16}]),
        )
    )
    gemm_item = next(
        item for item in matmul_plan.items if item.method_id == "parallel.grid_expansion_splitk"
    )
    assert gemm_item.parameter_space and "BLOCK_M" in gemm_item.prompt_fragment


def test_seed_note_composition_and_request_identity():
    planner = MethodPlanner(load_default())
    plan = planner.plan(_input(task=ELEMENTWISE_TASK, source=MATMUL_SOURCE))
    fragment = plan.items[0].prompt_fragment

    assert compose_seed_note("", None) == ""
    assert compose_seed_note("feedback", None) == "feedback"
    composed = compose_seed_note("feedback", fragment)
    assert composed == "feedback\n\n" + fragment

    base = build_request("problem", "model")
    with_plan = build_request("problem", "model", seed_note=fragment)
    assert isinstance(with_plan, ModelRequest)
    assert with_plan.request_sha256 != base.request_sha256, "plan injection is part of identity"
    repeat = build_request("problem", "model", seed_note=fragment)
    assert repeat.request_sha256 == with_plan.request_sha256, (
        "same inputs must reproduce the same request identity"
    )
    via_compose = build_request("problem", "model", seed_note=compose_seed_note("", fragment))
    assert via_compose.request_sha256 == with_plan.request_sha256


def test_attempt_record_validation():
    AttemptRecord(None, "evaluate", "correctness", "note")
    AttemptRecord("m", "confirm", "none", "", outcome="promoted")
    with pytest.raises(ValueError):
        AttemptRecord(None, "evaluate", "none", "note")  # 'none' requires promoted
    with pytest.raises(ValueError):
        AttemptRecord(None, "evaluate", "not-a-class", "note")


# --- loop integration (offline, scripted responder) --------------------------------


class ScriptedResponder:
    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        content = self._contents.pop(0)
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

    def total_tokens(self) -> int:
        return self._total


def candidate_module(body: str) -> str:
    return json.dumps({"code": f"class ModelNew:\n    {body}\n"})


BAD_CANDIDATE = candidate_module("def forward(self, x):\n        return 0")
GOOD_CANDIDATE = candidate_module("def forward(self, x):\n        return x")


def _stage_problem(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text(
        "# pinned problem\n", encoding="utf-8"
    )
    return snapshot


def _config(tmp_path: Path, snapshot: Path, **overrides):
    from kernelagent.optimization import OptimizationConfig

    defaults = dict(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=3,
        max_repair_rounds=2,
        gpu_budget_seconds=3600.0,
        token_budget=100000,
        output=tmp_path / "run",
        resume=False,
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


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


def _fast_candidate_timing(candidate_source: str, role: str) -> dict:
    batches = [10.0] * 12 if role == "eager" else [5.0] * 12
    return {"source_ok": True, "precondition": True, "async_leak": False, "batches": batches}


def _run(config, responder):
    return optimize(
        config,
        generator=CandidateGenerator(responder, ledger=RecordingLedger()),
        evaluate=_passing_evaluate,
        timing=_fast_candidate_timing,
        correctness_pro=None,
    )


def _journal_entries(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_loop_injects_plan_and_journals_method_plan(tmp_path):
    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([BAD_CANDIDATE, GOOD_CANDIDATE])
    result = _run(_config(tmp_path, snapshot), responder)
    assert result.state == "completed"

    entries = _journal_entries(tmp_path / "run")
    plan_events = [entry for entry in entries if entry["kind"] == "method_plan"]
    assert len(plan_events) == 2, "one method_plan event per candidate attempt"
    first = plan_events[0]
    assert first["action_id"] == "candidate-000"
    assert first["classification"] == "memory_bound"  # layernorm static prior
    assert first["evidence_coverage"] == "static_only"
    assert first["selected_method"] == "reduction.online_streaming"
    assert first["planned_methods"] and first["refusal_count"] >= 1

    # the first request carries exactly one single-factor fragment
    first_request_text = responder.requests[0].messages[-1][1]
    assert "TARGET CHANGE" in first_request_text
    assert "reduction.online_streaming" in first_request_text
    assert first_request_text.count("TARGET CHANGE") == 1
    # the second request keeps the failure feedback alongside the rotated plan
    second_request_text = responder.requests[1].messages[-1][1]
    assert "failed" in second_request_text
    assert second_request_text.count("TARGET CHANGE") == 1

    # failures attribute the method on the durable record (resume input)
    record = json.loads(
        (tmp_path / "run" / "records" / "candidate-000.json").read_text(encoding="utf-8")
    )
    assert record["method_id"] == "reduction.online_streaming"


def test_loop_without_planner_is_legacy(tmp_path):
    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([BAD_CANDIDATE, GOOD_CANDIDATE])
    result = optimize(
        _config(tmp_path, snapshot),
        generator=CandidateGenerator(responder, ledger=RecordingLedger()),
        evaluate=_passing_evaluate,
        timing=_fast_candidate_timing,
        correctness_pro=None,
        planner=None,
    )
    assert result.state == "completed"
    assert not [
        entry for entry in _journal_entries(tmp_path / "run") if entry["kind"] == "method_plan"
    ]
    for request in responder.requests:
        assert "TARGET CHANGE" not in request.messages[-1][1]
    assert "failed" in responder.requests[1].messages[-1][1], "feedback loop unchanged"


def test_loop_plan_is_deterministic_across_runs(tmp_path):
    hashes = []
    for index in (1, 2):
        snapshot = _stage_problem(tmp_path / f"case{index}")
        responder = ScriptedResponder([BAD_CANDIDATE, GOOD_CANDIDATE])
        result = _run(_config(tmp_path / f"case{index}", snapshot), responder)
        assert result.state == "completed"
        hashes.append([request.request_sha256 for request in responder.requests])
    assert hashes[0] == hashes[1], "same inputs must produce identical request identities"


def test_loop_resume_replays_attempts_and_rotates_the_plan(tmp_path):
    snapshot = _stage_problem(tmp_path)
    responder1 = ScriptedResponder([BAD_CANDIDATE])
    result1 = _run(
        _config(tmp_path, snapshot, max_candidates=1, max_repair_rounds=0, output=tmp_path / "run"),
        responder1,
    )
    assert result1.state == "no_improvement"
    first_selected = next(
        entry["selected_method"]
        for entry in _journal_entries(tmp_path / "run")
        if entry["kind"] == "method_plan"
    )
    assert first_selected is not None

    responder2 = ScriptedResponder([GOOD_CANDIDATE])
    result2 = _run(
        _config(tmp_path, snapshot, max_candidates=2, output=tmp_path / "run", resume=True),
        responder2,
    )
    assert result2.state == "completed"
    resumed_event = next(
        entry
        for entry in _journal_entries(tmp_path / "run")
        if entry["kind"] == "method_plan" and entry["action_id"] == "candidate-001"
    )
    assert resumed_event["selected_method"] != first_selected, (
        "a resumed run must see the persisted failure and rotate the plan"
    )
    resumed_request_text = responder2.requests[0].messages[-1][1]
    assert "failed" in resumed_request_text, "restored feedback is re-injected"
    assert resumed_request_text.count("TARGET CHANGE") == 1


def test_planner_input_contract_stays_pure():
    """The planner consumes only the declared evidence fields - no ambient
    state; two equal inputs from different construction paths plan equally."""
    planner = MethodPlanner(load_default())
    task = TaskProfile("t", 1, 40, "40_LayerNorm.py", "layernorm")
    features = extract_code_features("# pinned problem\n")
    direct = GeneratorInput(
        task=task, code_features=features, hardware=HardwareFacts.sm89_reference()
    )
    via_helper = _input(
        task=task, source="# pinned problem\n", hardware=HardwareFacts.sm89_reference()
    )
    assert planner.plan(direct) == planner.plan(via_helper)


def test_bottleneck_class_dataclass_validates():
    with pytest.raises(ValueError):
        BottleneckClass(label="vibes", signals_fired=(), signals_missing=(), coverage="static_only")
    with pytest.raises(ValueError):
        BottleneckClass(
            label="memory_bound", signals_fired=(), signals_missing=(), coverage="ncu_all"
        )
