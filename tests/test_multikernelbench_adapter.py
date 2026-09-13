"""MultiKernelBench snapshot reader, protocol identity and dev manifest."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernelagent.adapters.benchmarks.multikernelbench import (
    MkbManifestError,
    MkbSnapshot,
    SnapshotIntegrityError,
    TaskNotFoundError,
    canonical_content_sha256,
    load_dev_manifest,
    load_files_manifest,
    protocol_identity,
    run_verify,
    task_id,
    verify_snapshot,
)

COMMIT = "b" * 40

PROBLEM_CODE = (
    "import torch\nimport torch.nn as nn\n"
    "\n"
    "class Model(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return torch.relu(x)\n"
    "\n"
    "batch_size = 16\n"
    "dim = 128\n"
    "\n"
    "def get_inputs():\n"
    "    return [torch.rand(batch_size, dim)]\n"
    "\n"
    "def get_init_inputs():\n"
    "    return []\n"
)
PROTOCOL_CODE = {
    "config.py": "num_correct_trials = 5\nnum_perf_trials = 100\nnum_warmup = 3\nseed_num = 1024\n",
    "utils/correctness.py": "from config import num_correct_trials\n",
    "utils/performance.py": "from config import num_perf_trials\n",
}


def _write(root: Path, relative: str, content: str) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))
    return relative


def build_snapshot(root: Path) -> dict[str, str]:
    files = {
        "reference/activation/relu.py": PROBLEM_CODE,
        "reference/fuse/gemm_scale_batchnorm.py": PROBLEM_CODE,
        **PROTOCOL_CODE,
    }
    for relative, content in files.items():
        _write(root, relative, content)
    return files


def entry_for(relative: str, content: str) -> dict:
    data = content.encode("utf-8")
    return {
        "path": relative,
        "git_blob_sha1": hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def files_manifest(files: dict[str, str], commit: str = COMMIT) -> dict:
    return {
        "schema_version": "1.0",
        "repository": "wzzll123/MultiKernelBench",
        "commit": commit,
        "files": [entry_for(path, content) for path, content in sorted(files.items())],
    }


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def dev_manifest(files: dict[str, str], files_manifest_path: Path) -> dict:
    payload = files_manifest(files)
    binding = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    references = [
        {"path": path, "sha256": entry_for(path, files[path])["sha256"]} for path in PROTOCOL_CODE
    ]
    identity = protocol_identity(
        load_files_manifest(write_json(files_manifest_path.parent / "fm.json", payload)),
        tuple((ref["path"], ref["sha256"]) for ref in references),
    )
    return {
        "schema_version": "1.0",
        "suite": "multikernelbench",
        "upstream": {
            "repository": "wzzll123/MultiKernelBench",
            "commit": COMMIT,
            "files_manifest": "configs/multikernelbench/snapshot-files.manifest.json",
            "files_manifest_sha256": binding,
        },
        "problems": [
            {
                "category": "activation",
                "op": "relu",
                "path": "reference/activation/relu.py",
                "sha256": entry_for("reference/activation/relu.py", PROBLEM_CODE)["sha256"],
                "task_id": "mkb-activation-relu",
                "estimated_input_bytes": 16 * 128 * 4,
            }
        ],
        "protocol": {
            "name": identity["name"],
            "identity_sha256": identity["identity_sha256"],
            "description": "pinned upstream MKB NVIDIA protocol",
            "references": references,
        },
        "excluded_over_input_budget": [],
        "excluded_unresolved_input_estimate": [],
    }


@pytest.fixture()
def mini(tmp_path):
    root = tmp_path / "snapshot"
    files = build_snapshot(root)
    manifest_path = write_json(tmp_path / "files.manifest.json", files_manifest(files))
    dev = dev_manifest(files, manifest_path)
    dev_path = write_json(tmp_path / "dev.manifest.json", dev)
    return root, files, manifest_path, dev_path


def test_task_id_contains_category_and_op():
    assert task_id("activation", "relu") == "mkb-activation-relu"
    assert task_id("fuse", "gemm_scale_batchnorm") == "mkb-fuse-gemm_scale_batchnorm"


def test_list_and_get_are_consistent(mini):
    root, files, manifest_path, _ = mini
    snapshot = MkbSnapshot(root, load_files_manifest(manifest_path))
    assert snapshot.categories() == ("activation", "fuse")
    assert snapshot.ops("activation") == ("relu",)
    problem = snapshot.get_problem("activation", "relu")
    assert problem.code == files["reference/activation/relu.py"]
    assert problem.ref.task_id == "mkb-activation-relu"
    assert problem.ref.sha256 == entry_for("reference/activation/relu.py", PROBLEM_CODE)["sha256"]
    with pytest.raises(TaskNotFoundError) as excinfo:
        snapshot.get_problem("activation", "gelu")
    assert "relu" in str(excinfo.value.available)


def test_missing_and_corrupted_files_block_construction(mini):
    root, _, manifest_path, _ = mini
    (root / "reference/activation/relu.py").unlink()
    manifest = load_files_manifest(manifest_path)
    report = verify_snapshot(root, manifest)
    assert report["missing"] == ["reference/activation/relu.py"]
    with pytest.raises(SnapshotIntegrityError):
        MkbSnapshot(root, manifest)
    (root / "reference/activation/relu.py").write_bytes(b"# tampered\n")
    report = verify_snapshot(root, manifest)
    assert report["corrupted"][0]["path"] == "reference/activation/relu.py"
    with pytest.raises(SnapshotIntegrityError):
        MkbSnapshot(root, manifest)


def test_extra_disk_files_outside_manifest_are_invisible(mini):
    root, _, manifest_path, _ = mini
    _write(root, "reference/activation/gelu.py", "x = 1\n")
    snapshot = MkbSnapshot(root, load_files_manifest(manifest_path))
    assert "gelu" not in snapshot.ops("activation")


def test_protocol_source_is_rehashed(mini):
    root, _, manifest_path, _ = mini
    snapshot = MkbSnapshot(root, load_files_manifest(manifest_path))
    assert "seed_num" in snapshot.protocol_source("config.py")
    (root / "config.py").write_bytes(b"seed_num = 0\n")
    with pytest.raises(SnapshotIntegrityError):
        snapshot.protocol_source("config.py")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(schema_version="9.9"),
        lambda payload: payload.update(commit="not-a-commit"),
        lambda payload: payload.update(repository="ScalingIntelligence/KernelBench"),
        lambda payload: payload["files"].append(dict(payload["files"][0])),
        lambda payload: payload["files"][0].update(path="../escape.py"),
        lambda payload: payload["files"][0].update(path="backends/cuda_backend.py"),
        lambda payload: payload["files"][0].update(sha256="short"),
        lambda payload: payload.update(files=[]),
    ],
    ids=[
        "schema",
        "commit",
        "repository",
        "duplicate-path",
        "traversal",
        "out-of-scope",
        "bad-hash",
        "empty",
    ],
)
def test_files_manifest_schema_rejections(mini, mutate):
    _, files, manifest_path, _ = mini
    payload = files_manifest(files)
    mutate(payload)
    bad = write_json(manifest_path.parent / "bad.manifest.json", payload)
    with pytest.raises(MkbManifestError):
        load_files_manifest(bad)


def test_run_verify_passes_on_healthy_snapshot(mini):
    root, _, manifest_path, dev_path = mini
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 0, summary
    assert summary["status"] == "PASS"
    assert summary["files"]["missing"] == []
    assert summary["dev_manifest_check"]["protocol_references"] == "verified"
    assert summary["dev_subset"]["missing_ids"] == []
    assert summary["protocol"]["name"] == "mkb-upstream-cuda-v1"


def test_run_verify_fails_on_drifted_protocol_identity(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    drifted = dict(PROTOCOL_CODE)
    drifted["config.py"] = drifted["config.py"] + "num_warmup = 4\n"
    _write(root, "config.py", drifted["config.py"])
    entry = entry_for("config.py", drifted["config.py"])
    dev["protocol"]["references"][0]["sha256"] = entry["sha256"]
    tampered = write_json(dev_path.parent / "tampered.dev.json", dev)
    # files manifest still lists the original config.py bytes: the snapshot
    # now fails verification AND the protocol identity no longer matches.
    summary, code = run_verify(root, manifest_path, tampered)
    assert code == 1
    assert summary["status"] == "FAIL"


def test_dev_manifest_problem_sha_mismatch_is_fatal(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    dev["problems"][0]["sha256"] = "0" * 64
    bad = write_json(dev_path.parent / "bad.dev.json", dev)
    summary, code = run_verify(root, manifest_path, bad)
    assert code == 1
    assert "mismatch" in summary["error"]


def test_dev_manifest_files_binding_drift_is_fatal(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    dev["upstream"]["files_manifest_sha256"] = "1" * 64
    bad = write_json(dev_path.parent / "bad.dev.json", dev)
    summary, code = run_verify(root, manifest_path, bad)
    assert code == 1
    assert "drifted" in summary["error"]


def test_protocol_identity_changes_when_references_change(mini):
    root, files, manifest_path, _ = mini
    manifest = load_files_manifest(manifest_path)
    base = [
        (path, entry_for(path, content)["sha256"])
        for path, content in sorted(PROTOCOL_CODE.items())
    ]
    first = protocol_identity(manifest, tuple(base))
    changed = list(base)
    changed[0] = (changed[0][0], "0" * 64)
    second = protocol_identity(manifest, tuple(changed))
    assert first["identity_sha256"] != second["identity_sha256"]
    assert first["name"] == second["name"] == "mkb-upstream-cuda-v1"


def test_adapter_imports_no_gpu_stack():
    code = (
        "import sys; "
        "import kernelagent.adapters.benchmarks.multikernelbench; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_real_committed_manifests_load():
    repo = Path(__file__).resolve().parents[1]
    files = load_files_manifest(repo / "configs/multikernelbench/snapshot-files.manifest.json")
    assert len(files.entries) == 305
    members = files.problem_members()
    assert len(members) == 300
    assert all(not category.startswith("npukernelbench") for category, _, _ in members)
    dev = load_dev_manifest(repo / "configs/multikernelbench/dev-manifest.json")
    assert len(dev.problems) == 20
    assert dev.protocol["name"] == "mkb-upstream-cuda-v1"
    assert (
        canonical_content_sha256(repo / "configs/multikernelbench/snapshot-files.manifest.json")
        == dev.files_manifest_sha256
    )
    assert len(dev.excluded_over_input_budget) == 17
