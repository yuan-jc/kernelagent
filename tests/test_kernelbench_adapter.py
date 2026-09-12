"""Snapshot reader: listing, retrieval, integrity failures, task ids."""

import subprocess
import sys

import kernelbench_fixtures as fixtures
import pytest

from kernelagent.adapters.benchmarks.kernelbench import (
    DevManifestError,
    KernelBenchSnapshot,
    Problem,
    ProblemRef,
    SnapshotIntegrityError,
    TaskNotFoundError,
    load_files_manifest,
    run_verify,
    task_id,
    verify_snapshot,
)


@pytest.fixture()
def mini(tmp_path):
    root = tmp_path / "snapshot"
    files = fixtures.build_mini_snapshot(root)
    manifest_path = fixtures.write_json(
        tmp_path / "files.manifest.json", fixtures.files_manifest(files)
    )
    return root, files, manifest_path


def test_task_id_contains_level_and_problem_id():
    assert task_id(1, 1) == "kernelbench-l1-p001"
    assert task_id(2, 100) == "kernelbench-l2-p100"
    assert "l1" in task_id(1, 7) and "p007" in task_id(1, 7)


def test_list_and_get_are_consistent_across_loads(mini):
    root, files, manifest_path = mini
    first = KernelBenchSnapshot(root, load_files_manifest(manifest_path))
    second = KernelBenchSnapshot(root, load_files_manifest(manifest_path))
    assert first.levels() == (1, 2, 3)
    refs = first.list_problems(1)
    assert [ref.problem_id for ref in refs] == [1, 3]
    assert all(isinstance(ref, ProblemRef) for ref in refs)
    assert first.list_problems(1) == second.list_problems(1)
    problem = first.get_problem(1, 3)
    again = second.get_problem(1, 3)
    assert problem == again
    assert isinstance(problem, Problem)
    assert problem.code == files["KernelBench/level1/3_Batched_matrix_multiplication.py"]
    assert problem.ref.task_id == "kernelbench-l1-p003"
    assert problem.ref.sha256


def test_missing_problem_id_raises_with_available_hint(mini):
    root, _, manifest_path = mini
    snapshot = KernelBenchSnapshot(root, load_files_manifest(manifest_path))
    with pytest.raises(TaskNotFoundError) as excinfo:
        snapshot.get_problem(1, 99)
    assert list(excinfo.value.available) == [1, 3]
    assert "99" in str(excinfo.value)


def test_missing_file_is_reported_and_blocks_construction(mini):
    root, _, manifest_path = mini
    (root / "KernelBench/level2/1_FusedOp.py").unlink()
    manifest = load_files_manifest(manifest_path)
    report = verify_snapshot(root, manifest)
    assert report["missing"] == ["KernelBench/level2/1_FusedOp.py"]
    assert report["matched"] == len(manifest.entries) - 1
    with pytest.raises(SnapshotIntegrityError):
        KernelBenchSnapshot(root, manifest)


def test_corrupted_bytes_are_reported_with_expected_and_actual(mini):
    root, _, manifest_path = mini
    target = root / "KernelBench/level1/1_Square_matrix_multiplication_.py"
    target.write_text("# tampered\n", encoding="utf-8")
    manifest = load_files_manifest(manifest_path)
    report = verify_snapshot(root, manifest)
    assert report["corrupted"][0]["path"].endswith("1_Square_matrix_multiplication_.py")
    assert report["corrupted"][0]["expected_sha256"] != report["corrupted"][0]["actual_sha256"]
    with pytest.raises(SnapshotIntegrityError):
        KernelBenchSnapshot(root, manifest)


def test_manifest_tamper_is_detected(mini):
    root, files, manifest_path = mini
    payload = fixtures.files_manifest(files)
    payload["files"][0]["sha256"] = "0" * 64
    tampered = fixtures.write_json(tmp_path := root.parent / "tampered.json", payload)
    assert tampered == tmp_path
    with pytest.raises(SnapshotIntegrityError):
        KernelBenchSnapshot(root, load_files_manifest(tampered))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(schema_version="9.9"),
        lambda payload: payload.update(commit="not-a-commit"),
        lambda payload: payload["files"].append(dict(payload["files"][0])),
        lambda payload: payload["files"][0].update(path="../escape.py"),
        lambda payload: payload["files"][0].update(path="C:/win.py"),
        lambda payload: payload["files"][0].update(sha256="short"),
        lambda payload: payload.update(files=[]),
    ],
    ids=[
        "schema",
        "commit",
        "duplicate-path",
        "traversal",
        "windows-absolute",
        "bad-hash",
        "empty",
    ],
)
def test_files_manifest_schema_rejections(mini, mutate):
    _, files, _ = mini
    payload = fixtures.files_manifest(files)
    mutate(payload)
    bad = fixtures.write_json(mini[2].parent / "bad.manifest.json", payload)
    with pytest.raises(DevManifestError):
        load_files_manifest(bad)


def test_extra_disk_files_outside_manifest_are_invisible(mini):
    root, _, manifest_path = mini
    fixtures.write_snapshot_file(root, "KernelBench/level1/99_Unlisted_problem.py", "x = 1\n")
    snapshot = KernelBenchSnapshot(root, load_files_manifest(manifest_path))
    assert 99 not in snapshot.problem_ids(1)


def test_run_verify_passes_on_healthy_snapshot(mini):
    root, files, manifest_path = mini
    dev = fixtures.dev_manifest(files, root.parent, manifest_path)
    dev_path = fixtures.write_json(root.parent / "dev.manifest.json", dev)
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 0
    assert summary["status"] == "PASS"
    assert summary["files"]["missing"] == []
    assert summary["dev_manifest_check"]["protocol_references"] == "verified"
    assert summary["dev_subset"]["1"]["missing_ids"] == []


def test_run_verify_fails_on_corrupted_snapshot(mini):
    root, files, manifest_path = mini
    dev = fixtures.dev_manifest(files, root.parent, manifest_path)
    dev_path = fixtures.write_json(root.parent / "dev.manifest.json", dev)
    target = root / "KernelBench/level3/1_TinyModel.py"
    target.write_text("# drift\n", encoding="utf-8")
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 1
    assert summary["status"] == "FAIL"
    assert summary["files"]["corrupted"]


def test_run_verify_reports_dev_problem_absent_from_manifest(mini):
    root, files, manifest_path = mini
    dev = fixtures.dev_manifest(files, root.parent, manifest_path)
    dev["problems"].append(
        {
            "level": 1,
            "problem_id": 42,
            "name": "42_Absent_problem.py",
            "sha256": "c" * 64,
            "task_id": task_id(1, 42),
        }
    )
    dev_path = fixtures.write_json(root.parent / "dev.manifest.json", dev)
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 1
    assert summary["status"] == "FAIL"
    assert "42_Absent_problem.py" in summary["error"]


def test_adapter_imports_no_gpu_stack():
    code = (
        "import sys; "
        "import kernelagent.adapters.benchmarks.kernelbench; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_real_committed_files_manifest_loads():
    from pathlib import Path

    manifest_path = (
        Path(__file__).resolve().parents[1] / "configs/kernelbench/snapshot-files.manifest.json"
    )
    manifest = load_files_manifest(manifest_path)
    assert len(manifest.entries) == 270
    members = manifest.problem_members()
    levels = sorted({level for level, _, _ in members})
    assert levels == [1, 2, 3, 4]
