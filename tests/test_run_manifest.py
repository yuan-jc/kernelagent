"""RunManifest: round-trip fidelity, validation, identity guard policy.

The manifest is the versioned identity card a resume must verify before it
may reuse any previous result (RV01 / review R1): identity fields refuse,
revisible fields require explicit journal events, and secrets never enter
the payload."""

import json

import pytest

from kernelagent.domain import (
    MANIFEST_SCHEMA_VERSION,
    RUN_TERMINAL_STATES,
    ContractError,
    DomainError,
    PipelineStage,
    RunManifest,
    RunState,
    content_sha256,
    dumps,
    loads,
)
from kernelagent.domain.run_manifest import (
    IDENTITY_FIELDS,
    MANIFEST_FILENAME,
    PIPELINE_STAGE_ORDER,
    REVISIBLE_FIELDS,
)

_SHA1 = "a" * 64
_SHA2 = "b" * 64
_SHA3 = "c" * 64
_SHA4 = "d" * 64


def make_manifest(**overrides) -> RunManifest:
    defaults = dict(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at="2026-09-14T03:00:00+00:00",
        run_protocol="optimize-v1",
        problem="kernelbench:l1:40",
        problem_name="40_LayerNorm.py",
        problem_sha256=_SHA1,
        problem_source="# pinned problem\n",
        benchmark="kernelbench",
        snapshot="ScalingIntelligence__KernelBench",
        snapshot_root="/data/sources/ScalingIntelligence__KernelBench",
        backend="triton",
        eval_driver_sha256=_SHA2,
        timing_driver_sha256=_SHA3,
        timing_protocol_sha256=_SHA4,
        image_repo="kernelagent-eval",
        image_id="sha256:" + "e" * 64,
        gpu_device="nvidia.com/gpu=GPU-fake",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="MODEL_PROVIDER_API_KEY",
        max_candidates=5,
        max_repair_rounds=2,
        gpu_budget_seconds=1800.0,
        token_budget=200000,
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


def test_manifest_json_round_trip_preserves_identity():
    manifest = make_manifest()
    restored = loads(dumps(manifest))
    assert restored == manifest
    assert type(restored) is type(manifest)
    assert content_sha256(restored) == content_sha256(manifest)
    envelope = json.loads(dumps(manifest))
    assert envelope["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert envelope["type"] == "RunManifest"


def test_manifest_rejects_invalid_field_values():
    with pytest.raises(ContractError, match="schema_version"):
        make_manifest(schema_version="0.9")
    with pytest.raises(ContractError, match="problem_sha256"):
        make_manifest(problem_sha256="not-a-hash")
    with pytest.raises(ContractError, match="max_candidates"):
        make_manifest(max_candidates=0)
    with pytest.raises(ContractError, match="max_candidates"):
        make_manifest(max_candidates=-3)
    with pytest.raises(ContractError, match="max_repair_rounds"):
        make_manifest(max_repair_rounds=-1)
    with pytest.raises(ContractError, match="finite"):
        make_manifest(gpu_budget_seconds=float("nan"))
    with pytest.raises(ContractError, match="finite"):
        make_manifest(gpu_budget_seconds=float("inf"))
    with pytest.raises(ContractError, match="gpu_budget_seconds"):
        make_manifest(gpu_budget_seconds=0.0)
    with pytest.raises(ContractError, match="token_budget"):
        make_manifest(token_budget=0)


def test_manifest_identity_mismatches_are_reported_per_field():
    stored = make_manifest()
    assert stored.identity_mismatches(make_manifest()) == {}
    changed = make_manifest(
        gpu_device="nvidia.com/gpu=GPU-other",
        image_id="sha256:" + "9" * 64,
        timing_protocol_sha256=_SHA2,
        problem_sha256=_SHA2,
        model_id="other-model",
    )
    mismatches = stored.identity_mismatches(changed)
    assert set(mismatches) == {
        "gpu_device",
        "image_id",
        "timing_protocol_sha256",
        "problem_sha256",
        "model_id",
    }
    assert mismatches["gpu_device"] == (
        "nvidia.com/gpu=GPU-fake",
        "nvidia.com/gpu=GPU-other",
    )


def test_manifest_revisible_fields_are_not_identity():
    stored = make_manifest()
    bigger = make_manifest(
        max_candidates=10,
        max_repair_rounds=4,
        gpu_budget_seconds=3600.0,
        token_budget=999999,
    )
    assert stored.identity_mismatches(bigger) == {}, (
        "budget/allowance changes must never look like an identity change"
    )
    assert set(stored.budget_revisions(bigger)) == set(REVISIBLE_FIELDS)
    assert stored.budget_revisions(make_manifest()) == {}


def test_manifest_identity_and_revisible_sets_are_disjoint():
    assert not set(IDENTITY_FIELDS) & set(REVISIBLE_FIELDS)


def test_unknown_manifest_payloads_rejected_not_defaulted():
    """A manifest from another schema version, or with missing/unknown
    fields, must fail loudly - never silently count as 'identity matches'."""
    payload = json.loads(dumps(make_manifest()))
    payload["schema_version"] = "9.9"
    with pytest.raises(DomainError, match=MANIFEST_SCHEMA_VERSION):
        loads(json.dumps(payload))
    legacy = payload["data"]
    legacy.pop("timing_protocol_sha256")
    with pytest.raises(DomainError, match="timing_protocol_sha256"):
        loads(json.dumps({**payload, "schema_version": MANIFEST_SCHEMA_VERSION, "data": legacy}))
    extra = dict(payload["data"], credential="should-not-exist")
    with pytest.raises(DomainError, match="unknown field"):
        loads(json.dumps({**payload, "schema_version": MANIFEST_SCHEMA_VERSION, "data": extra}))


def test_manifest_never_carries_the_credential_value(monkeypatch, tmp_path):
    """The manifest records the credential env var NAME only; the value
    stays in the environment and must not leak into the persisted JSON."""
    from kernelagent.optimization import (
        OptimizationConfig,
        build_run_manifest,
    )

    monkeypatch.setenv("MODEL_PROVIDER_API_KEY", "super-secret-value-123")
    config = OptimizationConfig(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=1,
        max_repair_rounds=0,
        gpu_budget_seconds=60.0,
        token_budget=1000,
        output=tmp_path / "run",
        snapshot_root=tmp_path,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    problem_file = tmp_path / "40_LayerNorm.py"
    problem_file.write_text("# pinned problem\n", encoding="utf-8")
    manifest = build_run_manifest(
        config,
        problem_path=problem_file,
        problem_source=problem_file.read_text(encoding="utf-8"),
        problem_sha256="a" * 64,
        snapshot_root=tmp_path,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    persisted = dumps(manifest)
    assert "super-secret-value-123" not in persisted
    assert manifest.api_key_env == "MODEL_PROVIDER_API_KEY"
    assert not hasattr(manifest, "api_key")


def test_pipeline_stage_order_matches_alpha_sequence():
    assert [stage.value for stage in PIPELINE_STAGE_ORDER] == [
        "generate",
        "policy",
        "evaluate",
        "correctness_pro",
        "timing",
        "confirm",
    ]
    assert PipelineStage("generate") is PipelineStage.GENERATE


def test_run_state_terminal_states_are_explicit():
    assert {state.value for state in RUN_TERMINAL_STATES} == {
        "completed",
        "no_improvement",
        "budget_exhausted",
        "infra_error",
    }
    assert RunState.RUNNING not in RUN_TERMINAL_STATES
    assert RunState.CANCELLED not in RUN_TERMINAL_STATES


def test_manifest_filename_is_stable():
    assert MANIFEST_FILENAME == "run_manifest.json"
