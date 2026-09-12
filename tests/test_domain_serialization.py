"""Serialization: round-trip fidelity, schema versioning, tamper rejection."""

import json

import domain_samples as samples
import pytest

from kernelagent.domain import (
    SCHEMA_VERSION,
    ContractError,
    DomainError,
    SchemaVersionError,
    content_sha256,
    dumps,
    from_payload,
    implementation_id,
    loads,
    to_jsonable,
)

ALL_SAMPLES = [
    samples.make_evidence_ref(),
    samples.make_io_port(),
    samples.make_operator(),
    samples.make_workload(),
    samples.make_task(),
    samples.make_source_file(),
    samples.make_build_spec(),
    samples.make_implementation(),
    samples.make_hypothesis(),
    samples.make_method_proposal(),
    samples.make_evaluation_result(),
    samples.make_task_result(),
]


@pytest.mark.parametrize("sample", ALL_SAMPLES, ids=lambda value: type(value).__name__)
def test_round_trip_preserves_semantics(sample):
    restored = loads(dumps(sample))
    assert restored == sample
    assert type(restored) is type(sample)


@pytest.mark.parametrize("sample", ALL_SAMPLES, ids=lambda value: type(value).__name__)
def test_envelope_carries_schema_and_type(sample):
    envelope = to_jsonable(sample)
    assert envelope["schema_version"] == SCHEMA_VERSION
    assert envelope["type"] == type(sample).__name__


def test_key_order_does_not_change_identity():
    payload = json.loads(dumps(samples.make_implementation()))
    reordered_data = dict(reversed(list(payload["data"].items())))
    reordered = {**payload, "data": reordered_data}
    restored = from_payload(reordered)
    reference = samples.make_implementation()
    assert restored == reference
    assert content_sha256(restored) == content_sha256(reference)
    assert implementation_id(restored) == implementation_id(reference)


@pytest.mark.parametrize("version", ["0.9", "9.9", 2, None])
def test_unknown_schema_version_rejected(version):
    payload = json.loads(dumps(samples.make_workload()))
    payload["schema_version"] = version
    with pytest.raises(SchemaVersionError, match=SCHEMA_VERSION):
        from_payload(payload)


def test_unknown_type_rejected_with_known_types_listed():
    payload = {"schema_version": SCHEMA_VERSION, "type": "GemmAgent", "data": {}}
    with pytest.raises(DomainError, match="known types"):
        from_payload(payload)


def test_missing_envelope_parts_rejected():
    with pytest.raises(DomainError):
        from_payload({"type": "Workload", "data": {}})
    with pytest.raises(DomainError):
        from_payload("not json at all")
    with pytest.raises(DomainError):
        from_payload([1, 2, 3])


def test_missing_required_field_rejected():
    payload = json.loads(dumps(samples.make_evaluation_result()))
    del payload["data"]["candidate_id"]
    with pytest.raises(DomainError, match="candidate_id"):
        from_payload(payload)


def test_tampered_field_fails_validation_on_load():
    payload = json.loads(dumps(samples.make_evaluation_result()))
    payload["data"]["status"] = "excellent"
    with pytest.raises(ContractError):
        from_payload(payload)


def test_unknown_fields_are_rejected_at_envelope_and_nested_boundaries():
    payload = json.loads(dumps(samples.make_workload()))
    payload["data"]["strides"] = [[16, 1]]
    with pytest.raises(DomainError, match="strides"):
        from_payload(payload)

    nested = json.loads(dumps(samples.make_task()))
    nested["data"]["workloads"][0]["strides"] = [[16, 1]]
    with pytest.raises(DomainError, match="strides"):
        from_payload(nested)

    envelope = json.loads(dumps(samples.make_workload()))
    envelope["future_extension"] = True
    with pytest.raises(DomainError, match="future_extension"):
        from_payload(envelope)


@pytest.mark.parametrize(
    ("sample", "field", "bad_value"),
    [
        (samples.make_build_spec(), "flags", "-O3"),
        (samples.make_workload(), "dtypes", "float32"),
        (samples.make_workload(), "shapes", [[16, 16], "16x16"]),
        (samples.make_implementation(), "source_files", "kernel.py"),
    ],
)
def test_json_collection_fields_reject_non_arrays(sample, field, bad_value):
    payload = json.loads(dumps(sample))
    payload["data"][field] = bad_value
    with pytest.raises(DomainError, match=field):
        from_payload(payload)


def test_content_sha256_changes_only_with_content():
    base = samples.make_workload()
    same = samples.make_workload()
    changed = samples.make_workload(seed=8)
    assert content_sha256(base) == content_sha256(same)
    assert content_sha256(base) != content_sha256(changed)


def test_non_serializable_types_rejected():
    with pytest.raises(DomainError):
        to_jsonable({"not": "a domain object"})
    with pytest.raises(DomainError):
        to_jsonable(42)
