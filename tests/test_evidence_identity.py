"""Identity helpers: evaluation_key derivation and experiment ids."""

import json

import pytest

from kernelagent.adapters.storage import evaluation_key, new_experiment_id

HEX = [f"{i:02d}" * 32 for i in range(4)]  # four distinct 64-hex strings
IMPL, WORKLOAD, PROTOCOL_A, PROTOCOL_B, ENV_A, ENV_B = (
    "a" * 64,
    "b" * 64,
    "c" * 64,
    "d" * 64,
    "e" * 64,
    "0" * 64,
)


def base_key(**overrides) -> str:
    values = {
        "implementation_id": IMPL,
        "workload_content_sha256": WORKLOAD,
        "protocol_sha256": PROTOCOL_A,
        "environment_sha256": ENV_A,
    }
    values.update(overrides)
    return evaluation_key(**values)


def test_evaluation_key_is_stable_and_hex():
    assert base_key() == base_key()
    assert len(base_key()) == 64
    int(base_key(), 16)  # hex parseable


@pytest.mark.parametrize(
    "overrides",
    [
        {"implementation_id": "9" * 64},
        {"workload_content_sha256": "9" * 64},
        {"protocol_sha256": PROTOCOL_B},
        {"environment_sha256": ENV_B},
    ],
    ids=["implementation", "workload", "protocol", "environment"],
)
def test_any_field_change_yields_different_key(overrides):
    assert base_key(**overrides) != base_key()


def test_tool_version_change_yields_different_key():
    assert base_key() != base_key(tool_versions={"triton": "3.0"})
    assert base_key(tool_versions={"triton": "3.0"}) == base_key(tool_versions={"triton": "3.0"})
    assert base_key(tool_versions={"triton": "3.0"}) != base_key(tool_versions={"triton": "3.1"})
    # Tool order must not matter (canonical sorting).
    assert base_key(tool_versions={"a": "1", "b": "2"}) == base_key(
        tool_versions={"b": "2", "a": "1"}
    )


def test_environment_or_protocol_change_cannot_return_stale_cache_hits():
    recorded = base_key()
    assert base_key(protocol_sha256=PROTOCOL_B) != recorded
    assert base_key(environment_sha256=ENV_B) != recorded


@pytest.mark.parametrize(
    "kwargs",
    [
        {"implementation_id": "short"},
        {"workload_content_sha256": "A" * 64},
        {"protocol_sha256": None},
        {"environment_sha256": 42},
    ],
    ids=["impl", "workload", "protocol", "environment"],
)
def test_non_hex_inputs_rejected(kwargs):
    with pytest.raises(Exception, match="64-hex"):
        base_key(**kwargs)


def test_tool_versions_reject_empty_names_and_versions():
    with pytest.raises(Exception, match="non-empty"):
        base_key(tool_versions={"": "1"})
    with pytest.raises(Exception, match="non-empty"):
        base_key(tool_versions={"triton": ""})


def test_new_experiment_ids_are_unique_and_plain_hex():
    ids = {new_experiment_id() for _ in range(50)}
    assert len(ids) == 50
    for experiment_id in ids:
        assert len(experiment_id) == 32
        int(experiment_id, 16)


def test_evaluation_key_payload_is_canonical_json_friendly():
    # The key derivation must not depend on mapping iteration order anywhere.
    a = evaluation_key(IMPL, WORKLOAD, PROTOCOL_A, ENV_A, {"x": "1", "y": "2"})
    b = evaluation_key(IMPL, WORKLOAD, PROTOCOL_A, ENV_A, {"y": "2", "x": "1"})
    assert json.loads(json.dumps(a)) == a
    assert a == b
