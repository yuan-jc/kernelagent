"""Object invariants: frozen values, contract validation, §11.1 identity."""

import dataclasses

import domain_samples as samples
import pytest

from kernelagent.domain import (
    BuildSpec,
    ContractError,
    implementation_id,
)


def test_domain_objects_are_frozen():
    ref = samples.make_evidence_ref()
    impl = samples.make_implementation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.kind = "ncu_report"
    with pytest.raises(dataclasses.FrozenInstanceError):
        impl.entry_point = "other:run"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_sha256", "not-hex"),
        ("artifact_sha256", "A" * 64),
        ("kind", ""),
        ("producer_version", ""),
    ],
)
def test_evidence_ref_rejects_invalid_fields(field, value):
    with pytest.raises(ContractError):
        samples.make_evidence_ref(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "message_part"),
    [
        ({"shape": (0, 8)}, "positive"),
        ({"shape": (-2, 8)}, "positive"),
        ({"shape": (1.5, 8)}, "integers"),
        ({"shape": [4, 4]}, "tuple"),
        ({"name": ""}, "non-empty"),
    ],
)
def test_io_port_rejects_invalid_shapes(overrides, message_part):
    with pytest.raises(ContractError, match=message_part):
        samples.make_io_port(**overrides)


def test_io_port_allows_dynamic_dimension_and_scalars():
    assert samples.make_io_port(shape=(-1, 64)).shape == (-1, 64)
    assert samples.make_io_port(shape=()).shape == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"granularity": "layer"},
        {"granularity": ""},
        {"operator_id": ""},
        {"outputs": ()},
        {"inputs": (samples.make_io_port(name="a"), samples.make_io_port(name="a"))},
        {"outputs": (samples.make_io_port(name="y"), samples.make_io_port(name="y"))},
    ],
)
def test_operator_spec_rejects_invalid_contracts(overrides):
    with pytest.raises(ContractError):
        samples.make_operator(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shapes": ((16, 16),)},
        {"dtypes": ("float32",)},
        {"shapes": ((0, 16), (16, 16))},
        {"shapes": ((-1, 16), (16, 16))},
        {"weight": 0.0},
        {"weight": -0.5},
        {"seed": -1},
        {"workload_id": ""},
        {"operator_id": ""},
    ],
)
def test_workload_rejects_invalid_contracts(overrides):
    with pytest.raises(ContractError):
        samples.make_workload(**overrides)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: samples.make_task(workloads=()),
        lambda: samples.make_task(workloads=(samples.make_workload(), samples.make_workload())),
        lambda: samples.make_task(workloads=(samples.make_workload(operator_id="other-op"),)),
        lambda: samples.make_task(
            workloads=(samples.make_workload(shapes=((16, 16),), dtypes=("float32",)),)
        ),
        lambda: samples.make_task(track="production"),
        lambda: samples.make_task(task_id=""),
    ],
    ids=["empty", "duplicate-ids", "operator-mismatch", "input-count-mismatch", "track", "id"],
)
def test_task_rejects_invalid_contracts(builder):
    with pytest.raises(ContractError):
        builder()


@pytest.mark.parametrize(
    "builder",
    [
        lambda: samples.make_implementation(source_files=()),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(), samples.make_source_file())
        ),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(path="../evil.py"),)
        ),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(path="/abs/kernel.py"),)
        ),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(path="C:/win/kernel.py"),)
        ),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(path="a\\b.py"),)
        ),
        lambda: samples.make_implementation(
            source_files=(samples.make_source_file(path="./kernel.py"),)
        ),
        lambda: samples.make_implementation(parent_ids=("not-hex",)),
        lambda: samples.make_implementation(entry_point=""),
        lambda: samples.make_implementation(backend=""),
    ],
    ids=[
        "empty",
        "duplicate-paths",
        "parent-traversal",
        "absolute",
        "windows-drive",
        "backslash",
        "dot-segment",
        "bad-parent-hash",
        "entry",
        "backend",
    ],
)
def test_implementation_rejects_invalid_contracts(builder):
    with pytest.raises(ContractError):
        builder()


def test_build_spec_rejects_duplicate_env_and_bad_pairs():
    with pytest.raises(ContractError):
        BuildSpec(toolchain="triton", env=(("A", "1"), ("A", "2")))
    with pytest.raises(ContractError):
        BuildSpec(toolchain="triton", env=(("A", "1"), ["B", "2"]))
    with pytest.raises(ContractError):
        BuildSpec(toolchain="")


@pytest.mark.parametrize(
    "builder",
    [
        lambda: samples.make_implementation(source_files=[samples.make_source_file()]),
        lambda: samples.make_implementation(source_files=("kernel.py",)),
        lambda: samples.make_implementation(build_spec={"toolchain": "triton"}),
        lambda: samples.make_implementation(applicability_guard=[]),
        lambda: samples.make_implementation(origin=[]),
        lambda: BuildSpec(toolchain="triton", flags=["-O3"]),
        lambda: BuildSpec(toolchain="triton", env=[["A", "1"]]),
        lambda: samples.make_workload(shapes=[(16, 16), (16, 16)]),
        lambda: samples.make_workload(dtypes=["float32", "float32"]),
        lambda: samples.make_operator(inputs=[samples.make_io_port()]),
        lambda: samples.make_operator(outputs=("output",)),
        lambda: samples.make_operator(notes=[]),
        lambda: samples.make_task(workloads=[samples.make_workload()]),
        lambda: samples.make_hypothesis(predicted_observations=["faster"]),
        lambda: samples.make_hypothesis(supporting_evidence=("artifact",)),
        lambda: samples.make_method_proposal(target_workload_ids=["w1"]),
        lambda: samples.make_evaluation_result(evidence=[samples.make_evidence_ref()]),
        lambda: samples.make_task_result(evidence=[samples.make_evidence_ref()]),
    ],
)
def test_frozen_contracts_reject_mutable_containers_and_wrong_child_types(builder):
    with pytest.raises(ContractError):
        builder()


@pytest.mark.parametrize(
    "builder",
    [
        lambda: samples.make_method_proposal(estimated_gpu_seconds=-1.0),
        lambda: samples.make_method_proposal(estimated_gpu_seconds=float("nan")),
        lambda: samples.make_method_proposal(parameter_space_sha256="short"),
        lambda: samples.make_method_proposal(method_id=""),
        lambda: samples.make_method_proposal(hypothesis=samples.make_hypothesis(statement="")),
    ],
    ids=["negative-gpu-seconds", "nan-gpu-seconds", "bad-hash", "method-id", "empty-hypothesis"],
)
def test_method_proposal_rejects_invalid_contracts(builder):
    with pytest.raises(ContractError):
        builder()


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "ok"},
        {"status": "PASS"},
        {"protocol_sha256": ""},
        {"candidate_id": ""},
        {"environment_sha256": "zz"},
    ],
)
def test_evaluation_result_rejects_invalid_status_and_hashes(overrides):
    with pytest.raises(ContractError):
        samples.make_evaluation_result(**overrides)


def test_task_result_requires_champion_when_improved():
    with pytest.raises(ContractError):
        samples.make_task_result(champion=None)


def test_task_result_allows_no_improvement_without_champion():
    result = samples.make_task_result(terminal_state="completed_no_improvement", champion=None)
    assert result.champion is None
    assert result.best_result is not None


def test_implementation_id_is_stable_across_equal_content():
    assert implementation_id(samples.make_implementation()) == implementation_id(
        samples.make_implementation()
    )


def test_implementation_id_ignores_file_order_but_not_content():
    base = samples.make_implementation()
    reordered = samples.make_implementation(
        source_files=tuple(reversed(base.source_files)),
    )
    changed = samples.make_implementation(
        source_files=(
            samples.make_source_file(content="def run(a, b):\n    return a * b\n"),
            base.source_files[1],
        )
    )
    assert implementation_id(base) == implementation_id(reordered)
    assert implementation_id(base) != implementation_id(changed)


def test_implementation_id_covers_entry_guard_build_and_file_set():
    base = samples.make_implementation()
    assert implementation_id(base) != implementation_id(
        samples.make_implementation(entry_point="kernel:run_v2")
    )
    assert implementation_id(base) != implementation_id(
        samples.make_implementation(applicability_guard="sm >= 9.0")
    )
    assert implementation_id(base) != implementation_id(
        samples.make_implementation(build_spec=samples.make_build_spec(flags=("-O2",)))
    )
    assert implementation_id(base) != implementation_id(
        samples.make_implementation(
            source_files=base.source_files + (samples.make_source_file(path="extra.py"),)
        )
    )
    assert implementation_id(base) != implementation_id(
        samples.make_implementation(build_spec=None)
    )


def test_implementation_id_excludes_usage_and_provenance_fields():
    base = samples.make_implementation()
    assert implementation_id(base) == implementation_id(
        samples.make_implementation(operator_id="other-op", backend="cuda", origin="x@1")
    )
