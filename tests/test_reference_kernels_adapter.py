"""GPU-MODE reference-kernels adapter: parser, workloads, identity, verify."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernelagent.adapters.benchmarks import reference_kernels as rk
from kernelagent.adapters.benchmarks.reference_kernels import (
    RkManifestError,
    canonical_content_sha256,
    load_dev_manifest,
    load_files_manifest,
    parse_task_yml,
    protocol_identity,
    run_verify,
    task_id,
    workload_specs,
)

COMMIT = "c" * 40

TASK_YML = """# name: vectorsum-cuda-inline
files:
  - {"name": "submission.py", "source": "@SUBMISSION@"}
  - {"name": "task.py", "source": "task.py"}

lang: "py"

description: |
  Implement a vector sum reduction kernel.

  Input: A tensor of shape `(N,)`.

config:
  main: "eval.py"

templates:
  Python: "../template.py"

tests:
  - {"size": 1023, "seed": 4242}
  - {"size": 1024, "seed": 5236}

benchmarks:
  # - {"size": 8192, "seed": 54352}
  - {"size": 1638400, "seed": 93246} # fits on every listed GPU
  - {"size": 3276800, "seed": 6256}

test_timeout: 180
benchmark_timeout: 180
ranked_timeout: 420
"""

SET_YAML = """name: pmpp_v2 Practice Problems
deadline: ""
description: ""
problems:
  - directory: pmpp_v2/vectorsum_py
    name: vectorsum_v2
    deadline: "2100-12-31"
    gpus:
      - B200
      - H100
"""

UTILS_PY = "def clear_l2_cache():\n    pass\n"
EVAL_PY = "from utils import clear_l2_cache\n"
REFERENCE_PY = (
    "from utils import make_match_reference\n"
    "check_implementation = make_match_reference(lambda data: data)\n"
)
TASK_PY = "class TestSpec:\n    size: int\n    seed: int\n"


def _entry(relative: str, content: str) -> dict:
    data = content.encode("utf-8")
    return {
        "path": relative,
        "git_blob_sha1": hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def mini_files() -> dict[str, str]:
    return {
        "LICENSE": "June 9 Researcher Reciprocity License v1.0\n",
        "problems/pmpp_v2.yaml": SET_YAML,
        "problems/pmpp_v2/eval.py": EVAL_PY,
        "problems/pmpp_v2/utils.py": UTILS_PY,
        "problems/pmpp_v2/vectorsum_py/task.yml": TASK_YML,
        "problems/pmpp_v2/vectorsum_py/task.py": TASK_PY,
        "problems/pmpp_v2/vectorsum_py/reference.py": REFERENCE_PY,
    }


def files_manifest(files: dict[str, str], commit: str = COMMIT) -> dict:
    return {
        "schema_version": "1.0",
        "repository": "gpu-mode/reference-kernels",
        "commit": commit,
        "files": [_entry(path, content) for path, content in sorted(files.items())],
    }


def _write_all(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))


def dev_manifest_payload(files: dict[str, str], files_manifest_path: Path) -> dict:
    payload = files_manifest(files)
    binding = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    dev = {
        "schema_version": "1.0",
        "suite": "reference-kernels",
        "upstream": {
            "repository": "gpu-mode/reference-kernels",
            "commit": COMMIT,
            "files_manifest": "configs/reference-kernels/snapshot-files.manifest.json",
            "files_manifest_sha256": binding,
        },
        "problems": [
            {
                "set": "pmpp_v2",
                "name": "vectorsum_v2",
                "directory": "pmpp_v2/vectorsum_py",
                "members": sorted(
                    path for path in files if path.startswith("problems/pmpp_v2/vectorsum_py/")
                ),
                "member_sha256": [
                    _entry(path, files[path])["sha256"]
                    for path in sorted(
                        path for path in files if path.startswith("problems/pmpp_v2/vectorsum_py/")
                    )
                ],
                "task_yml_sha256": _entry("problems/pmpp_v2/vectorsum_py/task.yml", TASK_YML)[
                    "sha256"
                ],
                "task_yml_text": TASK_YML,
                "inputs": [
                    {"name": "data", "dtype": "float32", "dims": ["size"]},
                    {"name": "output", "dtype": "float32", "dims": [1]},
                ],
                "test_timeout": 180,
                "benchmark_timeout": 180,
                "ranked_timeout": 420,
                "official_gpus": ["B200", "H100"],
            }
        ],
        "protocol": {
            "name": "reference-kernels-local-replica-v1",
            "identity_sha256": "TO_COMPUTE",
            "description": "local replication; never comparable with official leaderboard",
        },
        "license_note": "June 9 Researcher Reciprocity License v1.0 - local evaluation only.",
    }
    return dev


@pytest.fixture()
def mini(tmp_path):
    root = tmp_path / "snapshot"
    files = mini_files()
    _write_all(root, files)
    manifest_path = tmp_path / "files.manifest.json"
    manifest_path.write_text(json.dumps(files_manifest(files), indent=2) + "\n")
    dev = dev_manifest_payload(files, manifest_path)
    dev_file = load_dev_manifest(_write_dev(tmp_path, dev))
    problems = (
        rk.problem_from_manifest(
            load_files_manifest(manifest_path), dev_file.problems[0], set_yamls={}
        ),
    )
    identity = protocol_identity(
        load_files_manifest(manifest_path), problems, name=dev["protocol"]["name"]
    )
    dev["protocol"]["identity_sha256"] = identity["identity_sha256"]
    dev_path = _write_dev(tmp_path, dev)
    return root, files, manifest_path, dev_path


def _write_dev(tmp_path: Path, dev: dict) -> Path:
    path = tmp_path / "dev.manifest.json"
    path.write_text(json.dumps(dev, indent=2) + "\n")
    return path


def test_task_id_format():
    assert task_id("pmpp_v2", "vectorsum_v2") == "rk-pmpp_v2-vectorsum_v2"


def test_parser_handles_comments_blocks_and_trailing_comments():
    parsed = parse_task_yml(TASK_YML)
    assert parsed["lang"] == "py"
    assert parsed["config"] == {"main": "eval.py"}
    assert parsed["templates"] == {"Python": "../template.py"}
    assert "Implement a vector sum reduction kernel." in parsed["description"]
    assert parsed["tests"] == [
        {"size": 1023, "seed": 4242},
        {"size": 1024, "seed": 5236},
    ]
    assert parsed["benchmarks"] == [
        {"size": 1638400, "seed": 93246},
        {"size": 3276800, "seed": 6256},
    ]
    assert parsed["test_timeout"] == 180
    assert parsed["ranked_timeout"] == 420


def test_parser_handles_list_of_mappings_and_plain_strings():
    parsed = parse_task_yml(SET_YAML)
    assert parsed["name"] == "pmpp_v2 Practice Problems"
    assert parsed["problems"][0] == {
        "directory": "pmpp_v2/vectorsum_py",
        "name": "vectorsum_v2",
        "deadline": "2100-12-31",
        "gpus": ["B200", "H100"],
    }


def test_parser_rejects_structure_outside_subset():
    from kernelagent.adapters.benchmarks._restricted_yaml import RestrictedYamlError

    with pytest.raises(RestrictedYamlError):
        parse_task_yml("key: &anchor value")
    with pytest.raises(RestrictedYamlError):
        parse_task_yml("key: {unterminated: true")


def test_workload_specs_resolve_shapes_dtypes_and_seeds(mini):
    root, files, manifest_path, dev_path = mini
    manifest = load_files_manifest(manifest_path)
    dev = load_dev_manifest(dev_path)
    problem = rk.problem_from_manifest(manifest, dev.problems[0], set_yamls={})
    specs = workload_specs(problem)
    tests = [spec for spec in specs if spec.kind == "test"]
    benchmarks = [spec for spec in specs if spec.kind == "benchmark"]
    assert [s.workload_id for s in tests] == [
        "rk-pmpp_v2-vectorsum_v2-test00",
        "rk-pmpp_v2-vectorsum_v2-test01",
    ]
    assert tests[0].shapes == ((1023,), (1,))
    assert tests[0].dtypes == ("float32", "float32")
    assert tests[0].seed == 4242
    assert benchmarks[0].shapes == ((1638400,), (1,))
    # declarative workloads convert into domain objects without executing code
    workload = tests[0].workload()
    assert workload.operator_id == "rk-pmpp_v2-vectorsum_v2"
    assert workload.weight == 1.0


def test_spec_missing_dim_key_is_a_load_time_error(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    dev["problems"][0]["inputs"][0]["dims"] = ["size", "missing_key"]
    loaded = load_dev_manifest(_write_dev(dev_path.parent, dev))
    with pytest.raises(RkManifestError):
        rk.problem_from_manifest(
            load_files_manifest(manifest_path), loaded.problems[0], set_yamls={}
        )


def test_embedded_task_yml_text_must_hash_to_binding(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    dev["problems"][0]["task_yml_text"] = TASK_YML + "# tampered\n"
    loaded = load_dev_manifest(_write_dev(dev_path.parent, dev))
    with pytest.raises(RkManifestError):
        rk.problem_from_manifest(
            load_files_manifest(manifest_path), loaded.problems[0], set_yamls={}
        )


def test_timeout_drift_between_manifest_and_task_yml_is_fatal(mini):
    root, files, manifest_path, dev_path = mini
    dev = json.loads(dev_path.read_text())
    dev["problems"][0]["test_timeout"] = 1
    loaded = load_dev_manifest(_write_dev(dev_path.parent, dev))
    with pytest.raises(RkManifestError):
        rk.problem_from_manifest(
            load_files_manifest(manifest_path), loaded.problems[0], set_yamls={}
        )


def test_files_manifest_rejects_out_of_scope_paths(tmp_path):
    files = mini_files()
    files["problems/helion/x/task.yml"] = "key: value\n"
    payload = files_manifest(files)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RkManifestError):
        load_files_manifest(path)


def test_run_verify_passes_and_detects_protocol_drift(mini):
    root, files, manifest_path, dev_path = mini
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 0, summary
    assert summary["status"] == "PASS"
    assert summary["workloads"] == {"test": 2, "benchmark": 2}
    assert summary["dev_manifest_check"]["protocol_references"] == "verified"

    tampered = root / "problems/pmpp_v2/utils.py"
    tampered.write_bytes(b"def clear_l2_cache():\n    return 1\n")
    summary, code = run_verify(root, manifest_path, dev_path)
    assert code == 1
    assert summary["status"] == "FAIL"
    assert "utils.py" in summary["error"]


def test_protocol_identity_covers_problem_sources(mini):
    root, files, manifest_path, dev_path = mini
    manifest = load_files_manifest(manifest_path)
    dev = load_dev_manifest(dev_path)
    problems = tuple(
        rk.problem_from_manifest(manifest, problem, set_yamls={}) for problem in dev.problems
    )
    identity = protocol_identity(manifest, problems, name=dev.protocol["name"])
    assert set(identity["files"]) == {
        "LICENSE",
        "problems/pmpp_v2/eval.py",
        "problems/pmpp_v2/utils.py",
        "problems/pmpp_v2/vectorsum_py/reference.py",
        "problems/pmpp_v2/vectorsum_py/task.py",
    }
    # identity is deterministic for the same inputs...
    other = protocol_identity(manifest, problems, name=dev.protocol["name"])
    assert other["identity_sha256"] == identity["identity_sha256"]
    # ...and changes when the protocol name changes
    renamed = protocol_identity(manifest, problems, name="reference-kernels-local-replica-v2")
    assert renamed["identity_sha256"] != identity["identity_sha256"]


def test_adapter_imports_no_gpu_stack():
    code = (
        "import sys; "
        "import kernelagent.adapters.benchmarks.reference_kernels; "
        "bad = {'torch', 'triton', 'numpy', 'openai'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_real_committed_manifests_load():
    repo = Path(__file__).resolve().parents[1]
    files = load_files_manifest(repo / "configs/reference-kernels/snapshot-files.manifest.json")
    assert len(files.entries) == 80
    assert files.set_names() == ("linalg", "pmpp_v2")
    assert len(files.problem_directories("pmpp_v2")) == 8
    assert len(files.problem_directories("linalg")) == 4
    dev = load_dev_manifest(repo / "configs/reference-kernels/dev-manifest.json")
    assert len(dev.problems) == 12
    assert dev.protocol["name"] == "reference-kernels-local-replica-v1"
    assert (
        canonical_content_sha256(repo / "configs/reference-kernels/snapshot-files.manifest.json")
        == dev.files_manifest_sha256
    )
