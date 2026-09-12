"""Evidence store: artifacts, experiments, events, audit, persistence."""

import pytest

from kernelagent.adapters.storage import (
    ArtifactIntegrityError,
    EvidenceStore,
    ExperimentConflictError,
    UnknownArtifactError,
    evaluation_key,
    new_experiment_id,
)
from kernelagent.domain import EvidenceRef

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


@pytest.fixture()
def store(tmp_path):
    with EvidenceStore(tmp_path / "evidence") as evidence_store:
        yield evidence_store


def ref_for(data: bytes, kind: str = "timing_samples", version: str = "t08") -> EvidenceRef:
    import hashlib

    return EvidenceRef(
        artifact_sha256=hashlib.sha256(data).hexdigest(), kind=kind, producer_version=version
    )


def record_args(key: str) -> dict:
    return {
        "evaluation_key": key,
        "implementation_id": HEX_A,
        "status": "passed",
    }


# -- artifacts -------------------------------------------------------------


def test_artifact_round_trip_and_idempotent_put(store):
    data = b"timing samples v1"
    ref = store.put_artifact(data, kind="timing_samples", producer_version="t08")
    assert ref.artifact_sha256 != ref.kind
    assert store.has_artifact(ref.artifact_sha256)
    assert store.get_artifact(ref.artifact_sha256) == data
    again = store.put_artifact(data, kind="timing_samples", producer_version="t08")
    assert again == ref
    meta = store.artifact_meta(ref.artifact_sha256)
    assert meta["size"] == len(data) and meta["kind"] == "timing_samples"


def test_artifact_read_rejects_unknown_and_corrupted(store, tmp_path):
    data = b"payload"
    ref = store.put_artifact(data, kind="k", producer_version="v")
    with pytest.raises(UnknownArtifactError):
        store.get_artifact(HEX_B)
    with pytest.raises(UnknownArtifactError):
        store.artifact_meta(HEX_B)
    # Corrupt the stored bytes behind the store's back.

    target = store.objects_dir / ref.artifact_sha256[:2] / ref.artifact_sha256
    target.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        store.get_artifact(ref.artifact_sha256)


def test_existing_artifact_with_drifting_bytes_is_flagged_on_put(store):
    ref = store.put_artifact(b"original", kind="k", producer_version="v")
    target = store.objects_dir / ref.artifact_sha256[:2] / ref.artifact_sha256
    target.write_bytes(b"drifted")
    with pytest.raises(ArtifactIntegrityError):
        store.put_artifact(b"original", kind="k", producer_version="v")


# -- experiments ------------------------------------------------------------


def populated_store(store):
    key = evaluation_key(HEX_A, HEX_B, HEX_C, HEX_D)
    data = b"raw timing samples"
    ref = store.put_artifact(data, kind="timing_samples", producer_version="t08")
    record = store.record_experiment(new_experiment_id(), evidence=[ref], **record_args(key))
    return key, ref, record


def test_record_and_query_experiment(store):
    key, ref, record = populated_store(store)
    fetched = store.get_experiment(record.experiment_id)
    assert fetched == record
    assert fetched.evidence == (ref,)
    assert store.list_experiments(key) == (record,)
    assert store.latest_experiment(key) == record
    assert store.latest_experiment(evaluation_key(HEX_A, HEX_B, HEX_C, "9" * 64)) is None
    assert store.get_experiment("missing-id") is None


def test_reopen_store_preserves_records_and_events(tmp_path):
    key = evaluation_key(HEX_A, HEX_B, HEX_C, HEX_D)
    with EvidenceStore(tmp_path / "evidence") as first:
        ref = first.put_artifact(b"data", kind="k", producer_version="v")
        record = first.record_experiment(
            "exp-1", evaluation_key=key, implementation_id=HEX_A, status="passed", evidence=[ref]
        )
    with EvidenceStore(tmp_path / "evidence") as second:
        assert second.get_experiment("exp-1") == record
        assert second.has_artifact(ref.artifact_sha256)
        kinds = [event["kind"] for event in second.events()]
        assert "experiment_recorded" in kinds


def test_evidence_must_reference_stored_artifacts(store):
    key = evaluation_key(HEX_A, HEX_B, HEX_C, HEX_D)
    ghost = EvidenceRef(artifact_sha256=HEX_B, kind="ncu_report", producer_version="t08")
    with pytest.raises(UnknownArtifactError, match="unstored"):
        store.record_experiment(
            "exp-1", evaluation_key=key, implementation_id=HEX_A, status="passed", evidence=[ghost]
        )
    assert store.get_experiment("exp-1") is None


def test_invalid_status_rejected(store):
    key = evaluation_key(HEX_A, HEX_B, HEX_C, HEX_D)
    with pytest.raises(Exception, match="status must be one of"):
        store.record_experiment(
            "exp-1", evaluation_key=key, implementation_id=HEX_A, status="excellent", evidence=()
        )


def test_identical_resubmission_is_idempotent(store):
    key, ref, record = populated_store(store)
    resubmitted = store.record_experiment(record.experiment_id, evidence=[ref], **record_args(key))
    assert resubmitted == record
    assert len(store.list_experiments(key)) == 1
    kinds = [event["kind"] for event in store.events()]
    assert kinds.count("experiment_deduplicated") == 1


def test_conflicting_resubmission_rejected_original_intact(store):
    key, ref, record = populated_store(store)
    conflicting = dict(record_args(key), status="incorrect")
    with pytest.raises(ExperimentConflictError, match="immutable"):
        store.record_experiment(record.experiment_id, evidence=[ref], **conflicting)
    fetched = store.get_experiment(record.experiment_id)
    assert fetched.status == "passed"
    assert len(store.list_experiments(key)) == 1


def test_conflicting_resubmission_with_different_key_rejected(store):
    key, ref, record = populated_store(store)
    other_key = evaluation_key(HEX_A, HEX_B, HEX_C, "8" * 64)
    with pytest.raises(ExperimentConflictError):
        store.record_experiment(record.experiment_id, evidence=[ref], **record_args(other_key))


def test_environment_change_produces_cache_miss(store):
    key_env_a = evaluation_key(HEX_A, HEX_B, HEX_C, HEX_D)
    key_env_b = evaluation_key(HEX_A, HEX_B, HEX_C, "9" * 64)
    ref = store.put_artifact(b"samples", kind="timing_samples", producer_version="t08")
    store.record_experiment(new_experiment_id(), evidence=[ref], **record_args(key_env_a))
    # Same implementation and workload, different environment: no false hit.
    assert store.latest_experiment(key_env_b) is None
    assert store.latest_experiment(key_env_a) is not None


# -- events and audit -------------------------------------------------------


def test_event_sequence_monotonic_and_since_query(store):
    first = store.append_event("alpha", {"n": 1})
    second = store.append_event("beta", {"n": 2})
    assert first < second
    tail = store.events(since=first)
    assert [event["seq"] for event in tail] == [second]
    assert tail[0]["payload"] == {"n": 2}
    all_events = store.events()
    assert [event["kind"] for event in all_events] == ["alpha", "beta"]


def test_audit_healthy_store_is_clean(store):
    populated_store(store)
    report = store.audit()
    assert report.healthy
    assert report.indexed == 1
    assert report.as_dict()["corrupted"] == []


def test_audit_detects_corrupted_missing_and_orphaned(store):
    key, ref, _record = populated_store(store)
    other = store.put_artifact(b"other bytes", kind="k", producer_version="v")
    # Corrupt one artifact, delete another, and plant a stray object.
    corrupted_target = store.objects_dir / ref.artifact_sha256[:2] / ref.artifact_sha256
    corrupted_target.write_bytes(b"corrupted bytes")
    missing_target = store.objects_dir / other.artifact_sha256[:2] / other.artifact_sha256
    missing_target.unlink()
    orphan_shard = store.objects_dir / "ff"
    orphan_shard.mkdir(exist_ok=True)
    (orphan_shard / (HEX_C[:60] + "ff")).write_bytes(b"orphan")
    report = store.audit()
    assert not report.healthy
    assert report.corrupted == (ref.artifact_sha256,)
    assert report.missing == (other.artifact_sha256,)
    assert (HEX_C[:60] + "ff") in report.orphaned
    assert report.indexed == 2


def test_storage_imports_no_gpu_stack():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import kernelagent.adapters.storage; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
