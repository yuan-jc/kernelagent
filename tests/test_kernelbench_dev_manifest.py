"""The committed development manifest must be internally consistent.

These tests run in CI without the local snapshot cache: they validate the two
committed JSON artifacts against each other and against the upstream selection
constants recorded from src/kernelbench/dataset.py at the pinned commit.
"""

from pathlib import Path

import pytest

from kernelagent.adapters.benchmarks.kernelbench import (
    check_dev_manifest,
    load_dev_manifest,
    load_files_manifest,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "kernelbench"
FILES_MANIFEST = CONFIGS / "snapshot-files.manifest.json"
DEV_MANIFEST = CONFIGS / "dev-manifest.json"

# LEVEL{1,2,3}_REPRESENTATIVE_IDS exactly as recorded in the pinned
# src/kernelbench/dataset.py (dataset_module_sha256 in the dev manifest pins
# which version of that file the sets came from).
UPSTREAM_REPRESENTATIVE_IDS = {
    1: (1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87),
    2: (1, 2, 8, 18, 23, 28, 33, 43),
    3: (1, 5, 8, 11, 20, 21, 33, 38, 43),
}


def test_committed_dev_manifest_matches_files_manifest():
    files = load_files_manifest(FILES_MANIFEST)
    dev = load_dev_manifest(DEV_MANIFEST)
    assert dev.upstream_commit == files.commit == "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
    report = check_dev_manifest(dev, files, FILES_MANIFEST)
    assert report["problems"] == 35
    assert report["protocol_references"] == "not_checked"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_dev_subset_ids_equal_upstream_representative_sets(level):
    dev = load_dev_manifest(DEV_MANIFEST)
    assert dev.ids_for_level(level) == UPSTREAM_REPRESENTATIVE_IDS[level]


def test_dev_manifest_protocol_sections_are_separate():
    dev = load_dev_manifest(DEV_MANIFEST)
    assert dev.protocols.upstream_references
    assert dev.protocols.extended == ()
    with pytest.raises(Exception, match="separate"):
        dev.protocols.merged()


def test_dev_manifest_binds_files_manifest_bytes():
    import hashlib

    dev = load_dev_manifest(DEV_MANIFEST)
    actual = hashlib.sha256(FILES_MANIFEST.read_bytes()).hexdigest()
    assert dev.files_manifest_sha256 == actual
    assert dev.files_manifest_path.endswith("snapshot-files.manifest.json")


def test_every_dev_problem_references_a_manifest_entry():
    files = load_files_manifest(FILES_MANIFEST)
    dev = load_dev_manifest(DEV_MANIFEST)
    for problem in dev.problems:
        entry = files.entry(f"KernelBench/level{problem.level}/{problem.name}")
        assert entry is not None
        assert entry.sha256 == problem.sha256
