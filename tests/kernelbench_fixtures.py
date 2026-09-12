"""Build synthetic KernelBench-style snapshots and manifests for adapter tests.

These fixtures verify the reader only; they are never used as optimization
tasks and never leave the test's tmp directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

COMMIT = "a" * 40

LEVEL1_CODE = (
    "import torch\nimport torch.nn as nn\n"
    "\n"
    "class Model(nn.Module):\n"
    "    def forward(self, a, b):\n"
    "        return a @ b\n"
)
LEVEL1B_CODE = (
    "import torch\n"
    "\n"
    "class Model(torch.nn.Module):\n"
    "    def forward(self, a):\n"
    "        return a + 1\n"
)
LEVEL2_CODE = (
    "import torch\n"
    "\n"
    "class Model(torch.nn.Module):\n"
    "    def forward(self, x):\n"
    "        return torch.relu(x).sum()\n"
)
LEVEL3_CODE = (
    "import torch\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
)


def write_snapshot_file(root: Path, relative: str, content: str) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: newline translation would drift the recorded sha256.
    target.write_bytes(content.encode("utf-8"))
    return relative


def build_mini_snapshot(root: Path) -> dict[str, str]:
    """Create a small snapshot with problems on three levels plus upstream
    protocol source files; returns {relative path: content}."""
    files = {
        f"KernelBench/level1/{name}": code
        for name, code in [
            ("1_Square_matrix_multiplication_.py", LEVEL1_CODE),
            ("3_Batched_matrix_multiplication.py", LEVEL1B_CODE),
        ]
    }
    files["KernelBench/level2/1_FusedOp.py"] = LEVEL2_CODE
    files["KernelBench/level3/1_TinyModel.py"] = LEVEL3_CODE
    files["src/kernelbench/dataset.py"] = "# dataset selection rules\n"
    files["src/kernelbench/eval.py"] = "# evaluation entry points\n"
    files["src/kernelbench/timing.py"] = "# timing entry points\n"
    for relative, content in files.items():
        write_snapshot_file(root, relative, content)
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
        "repository": "ScalingIntelligence/KernelBench",
        "commit": commit,
        "files": [entry_for(path, content) for path, content in sorted(files.items())],
    }


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def problem_entries(files: dict[str, str], levels_prefixes: list[tuple[int, str]]) -> list[dict]:
    entries = []
    for level, name in levels_prefixes:
        relative = f"KernelBench/level{level}/{name}"
        content = files[relative]
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        pid = int(name.split("_")[0])
        entries.append(
            {
                "level": level,
                "problem_id": pid,
                "name": name,
                "sha256": sha,
                "task_id": f"kernelbench-l{level}-p{pid:03d}",
            }
        )
    return entries


def dev_manifest(
    files: dict[str, str],
    tmp_path: Path,
    manifest_path: Path,
    commit: str = COMMIT,
) -> dict:
    manifest_payload = files_manifest(files, commit=commit)
    write_json(manifest_path, manifest_payload)
    binding = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "schema_version": "1.0",
        "suite": "kernelbench",
        "upstream": {
            "repository": "ScalingIntelligence/KernelBench",
            "commit": commit,
            "dataset_module": "src/kernelbench/dataset.py",
            "dataset_module_sha256": hashlib.sha256(
                files["src/kernelbench/dataset.py"].encode("utf-8")
            ).hexdigest(),
            "selection_rule": "test fixture",
            "files_manifest": str(manifest_path),
            "files_manifest_sha256": binding,
        },
        "problems": problem_entries(
            files,
            [
                (1, "1_Square_matrix_multiplication_.py"),
                (1, "3_Batched_matrix_multiplication.py"),
                (2, "1_FusedOp.py"),
            ],
        ),
        "protocols": {
            "upstream_compatible": {
                "description": "fixture upstream protocol",
                "references": [
                    {
                        "path": "src/kernelbench/eval.py",
                        "sha256": hashlib.sha256(
                            files["src/kernelbench/eval.py"].encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            },
            "extended": [],
        },
    }
