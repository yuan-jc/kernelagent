"""Protocol separation: upstream-compatible and extended never merge."""

import json

import kernelbench_fixtures as fixtures
import pytest

from kernelagent.adapters.benchmarks.kernelbench import (
    DevManifestError,
    ProtocolMergeError,
    check_dev_manifest,
    load_dev_manifest,
)


@pytest.fixture()
def mini(tmp_path):
    root = tmp_path / "snapshot"
    files = fixtures.build_mini_snapshot(root)
    manifest_path = fixtures.write_json(
        tmp_path / "files.manifest.json", fixtures.files_manifest(files)
    )
    dev = fixtures.dev_manifest(files, tmp_path, manifest_path)
    dev_path = fixtures.write_json(tmp_path / "dev.manifest.json", dev)
    return root, files, manifest_path, dev, dev_path


def test_protocols_are_separate_structures(mini):
    _, _, _, dev, dev_path = mini
    loaded = load_dev_manifest(dev_path)
    assert loaded.protocols.upstream_description
    assert loaded.protocols.upstream_references
    assert loaded.protocols.extended == ()
    assert loaded.protocols.upstream_references is not loaded.protocols.extended


def test_merged_view_is_refused(mini):
    loaded = load_dev_manifest(mini[4])
    with pytest.raises(ProtocolMergeError):
        loaded.protocols.merged()


def test_missing_extended_section_is_rejected(mini):
    root, files, manifest_path, dev, dev_path = mini
    payload = json.loads(dev_path.read_text(encoding="utf-8"))
    del payload["protocols"]["extended"]
    bad = fixtures.write_json(dev_path.parent / "bad-dev.json", payload)
    with pytest.raises(DevManifestError, match="exactly"):
        load_dev_manifest(bad)


def test_unknown_protocol_key_is_rejected(mini):
    _, _, _, dev, dev_path = mini
    payload = json.loads(dev_path.read_text(encoding="utf-8"))
    payload["protocols"]["production_reuse"] = []
    bad = fixtures.write_json(dev_path.parent / "bad-dev.json", payload)
    with pytest.raises(DevManifestError, match="exactly"):
        load_dev_manifest(bad)


def test_binding_hash_drift_is_rejected(mini):
    root, files, manifest_path, dev, dev_path = mini
    # Rewrite the files manifest with an extra harmless field -> bytes change.
    payload = fixtures.files_manifest(files)
    payload["note"] = "post-freeze edit"
    fixtures.write_json(manifest_path, payload)
    with pytest.raises(DevManifestError, match="drifted"):
        check_dev_manifest(load_dev_manifest(dev_path), _files(mini), manifest_path)


def test_upstream_reference_hash_is_verified_against_root(mini):
    root, files, manifest_path, dev, dev_path = mini
    loaded = load_dev_manifest(dev_path)
    report = check_dev_manifest(loaded, _files(mini), manifest_path, root=root)
    assert report["protocol_references"] == "verified"
    # Corrupt the referenced upstream source file.
    (root / "src/kernelbench/eval.py").write_text("# drifted\n", encoding="utf-8")
    with pytest.raises(DevManifestError, match="protocol reference hash mismatch"):
        check_dev_manifest(loaded, _files(mini), manifest_path, root=root)


def test_without_root_protocol_references_are_reported_unchecked(mini):
    root, files, manifest_path, dev, dev_path = mini
    report = check_dev_manifest(load_dev_manifest(dev_path), _files(mini), manifest_path)
    assert report["protocol_references"] == "not_checked"


def _files(mini):
    from kernelagent.adapters.benchmarks.kernelbench import load_files_manifest

    return load_files_manifest(mini[2])
