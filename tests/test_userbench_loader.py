"""user-bench suite loader: schema validation, freezing, domain mapping."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernelagent.adapters.benchmarks.userbench import (
    UserBenchError,
    generate_manifest,
    load_manifest,
    load_problem,
    load_suite,
    verify_manifest,
)

REPO = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO / "configs/userbench/suites/examples_v1"


@pytest.fixture()
def suite():
    return load_suite(SUITE_ROOT)


@pytest.fixture()
def writable_suite(tmp_path):
    target = tmp_path / "examples_v1"
    shutil.copytree(SUITE_ROOT, target)
    return target


def _refreeze(suite_root: Path) -> None:
    (suite_root / "manifest.json").write_text(
        json.dumps(generate_manifest(suite_root), indent=2) + "\n"
    )


def test_committed_suite_loads_and_maps_to_domain(suite):
    assert suite.name == "examples_v1"
    assert [problem.name for problem in suite.problems] == ["add_relu_fp32", "rmsnorm_fp32"]
    add_relu = suite.problems[0]
    task = add_relu.to_task()
    assert task.task_id == "userbench-examples_v1-add_relu_fp32"
    assert task.operator.operator_id == "userbench/examples_v1/add_relu_fp32"
    assert task.operator.granularity == "operator"
    assert [port.name for port in task.operator.inputs] == ["a", "b"]
    assert [port.dtype for port in task.operator.inputs] == ["float32", "float32"]
    assert task.operator.inputs[0].shape == (-1, -1)
    assert [(w.workload_id, w.shapes, w.seed) for w in task.workloads] == [
        ("w1_base", ((1024, 4096), (1024, 4096)), 0),
        ("w2_wide", ((4096, 8192), (4096, 8192)), 1),
    ]


def test_workload_specs_map_to_dispatch_guards(suite):
    add_relu = suite.problems[0]
    specs = add_relu.workload_specs()
    assert [(s.workload_id, s.weight, s.required) for s in specs] == [
        ("w1_base", 1.0, True),
        ("w2_wide", 2.0, True),
    ]
    assert specs[1].max_shape == (4096, 8192)
    assert specs[0].max_shape is None


def test_tolerance_is_bound_and_reference_hash_frozen(suite):
    rmsnorm = suite.problems[1]
    assert (rmsnorm.tolerance.rtol, rmsnorm.tolerance.atol) == (1e-5, 1e-6)
    assert rmsnorm.tolerance.policy == "allclose"
    recorded = suite.manifest["files"][f"{rmsnorm.directory}/reference.py"]
    assert rmsnorm.reference_sha256 == recorded
    assert len(rmsnorm.reference_sha256) == 64


def test_protocol_identity_is_named_and_content_bound(suite):
    identity = suite.protocol_identity()
    assert identity["correctness"] == "userbench-correctness-v1"
    assert identity["timing"] == "userbench-timing-v1"
    assert len(identity["identity_sha256"]) == 64


def test_tampered_reference_is_rejected(writable_suite):
    reference = writable_suite / "problems/rmsnorm_fp32/reference.py"
    reference.write_text("import torch\n\n\ndef run(x):\n    return x * 2.0\n", encoding="utf-8")
    with pytest.raises(UserBenchError) as excinfo:
        load_suite(writable_suite)
    assert "corrupted" in str(excinfo.value)


def test_tampered_tolerance_is_rejected_without_refreeze(writable_suite):
    problem_path = writable_suite / "problems/add_relu_fp32/problem.yaml"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8").replace("rtol: 1.0e-5", "rtol: 1.0e-2"),
        encoding="utf-8",
    )
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)


def test_tolerance_above_suite_ceiling_fails_even_after_refreeze(writable_suite):
    problem_path = writable_suite / "problems/add_relu_fp32/problem.yaml"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8").replace("rtol: 1.0e-5", "rtol: 1.0e-2"),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError) as excinfo:
        load_suite(writable_suite)
    assert "ceiling" in str(excinfo.value)


def test_duplicate_workload_id_fails(writable_suite):
    problem_path = writable_suite / "problems/rmsnorm_fp32/problem.yaml"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8").replace("id: w2_wide", "id: w1_base"),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError) as excinfo:
        load_suite(writable_suite)
    assert "duplicate workload id" in str(excinfo.value)


def test_workload_rank_mismatch_fails(writable_suite):
    problem_path = writable_suite / "problems/rmsnorm_fp32/problem.yaml"
    original = problem_path.read_text(encoding="utf-8")
    problem_path.write_text(
        original.replace("shapes: [[1024, 4096]]", "shapes: [[1024]]"), encoding="utf-8"
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)


def test_workload_input_count_mismatch_fails(writable_suite):
    problem_path = writable_suite / "problems/rmsnorm_fp32/problem.yaml"
    original = problem_path.read_text(encoding="utf-8")
    pattern = 'dtypes: ["float32"]\n      seed: 1'
    replacement = 'dtypes: ["float32", "float32"]\n      seed: 1'
    problem_path.write_text(original.replace(pattern, replacement), encoding="utf-8")
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)


def test_unknown_distribution_and_policy_fail(writable_suite):
    problem_path = writable_suite / "problems/rmsnorm_fp32/problem.yaml"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8").replace(
            "distribution: randn", "distribution: beta"
        ),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)

    problem_path.write_text(
        problem_path.read_text(encoding="utf-8")
        .replace("distribution: beta", "distribution: randn")
        .replace("policy: allclose", "policy: looser_allclose"),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)


def test_missing_reference_entry_fails(writable_suite):
    problem_path = writable_suite / "problems/rmsnorm_fp32/problem.yaml"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8").replace("entry: run", ""),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_problem(load_suite(writable_suite), "rmsnorm_fp32")


def test_wrong_api_version_fails(writable_suite):
    suite_path = writable_suite / "suite.yaml"
    suite_path.write_text(
        suite_path.read_text(encoding="utf-8").replace(
            "kernelagent/userbench/v1alpha1", "kernelagent/userbench/v9"
        ),
        encoding="utf-8",
    )
    _refreeze(writable_suite)
    with pytest.raises(UserBenchError):
        load_suite(writable_suite)


def test_manifest_verification_reports_missing_and_corrupted(writable_suite):
    (writable_suite / "problems/rmsnorm_fp32/reference.py").unlink()
    suite = load_manifest(writable_suite)
    report = verify_manifest(writable_suite, suite)
    assert report["missing"] == ["problems/rmsnorm_fp32/reference.py"]
    assert report["matched"] == len(suite["files"]) - 1


def test_loader_imports_no_torch():
    code = (
        "import sys; "
        "import kernelagent.adapters.benchmarks.userbench; "
        "assert 'torch' not in sys.modules, 'loader imported torch'"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_example_candidates_exist_for_smoke(suite):
    for problem in suite.problems:
        candidate = suite.root / problem.directory / "examples/candidate.py"
        assert candidate.is_file(), f"missing example candidate for {problem.name}"
        source = candidate.read_text(encoding="utf-8")
        assert f"def {problem.reference_entry}(" in source
