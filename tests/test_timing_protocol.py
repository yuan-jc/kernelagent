"""Formal timing protocol (T07): control-plane checks.

Pure CPU: protocol serialization/identity, payload validation (source tag
and protocol binding), and the batch-level bootstrap statistics. Real
GPU timing runs through examples/timing_protocol_smoke.py; unit tests
cannot substitute for them."""

import json

from kernelagent.adapters.evals import (
    SOURCE_TAG,
    TimingProtocol,
    bootstrap_ratio_ci,
    validate_timing_payload,
)


def _payload(protocol: TimingProtocol, **overrides) -> dict:
    payload = {
        "source": SOURCE_TAG,
        "protocol": protocol.to_dict(),
        "batch_samples_ms": [1.0, 1.1, 0.9],
    }
    payload.update(overrides)
    return payload


def test_protocol_identity_is_stable_and_binds_all_fields():
    protocol = TimingProtocol()
    assert protocol.identity_sha256() == protocol.identity_sha256()
    altered = TimingProtocol(iters_per_batch=protocol.iters_per_batch + 1)
    assert altered.identity_sha256() != protocol.identity_sha256()
    canonical = json.dumps(protocol.to_dict(), sort_keys=True, separators=(",", ":"))
    for field in ("warmup_iters", "num_batches", "iters_per_batch", "stream_policy"):
        assert field in canonical


def test_valid_payload_accepted():
    protocol = TimingProtocol()
    ok, note = validate_timing_payload(_payload(protocol), protocol.identity_sha256())
    assert ok and note == ""


def test_profile_source_rejected():
    """NCU/profiling time must not pass as a formal timing result."""
    protocol = TimingProtocol()
    for source in ("ncu", "nsys", None, "cuda_event_profile"):
        ok, _ = validate_timing_payload(
            _payload(protocol, source=source), protocol.identity_sha256()
        )
        assert not ok, f"source {source!r} must be rejected"


def test_protocol_hash_binding_enforced():
    protocol = TimingProtocol()
    other = TimingProtocol(num_batches=protocol.num_batches + 1)
    ok, note = validate_timing_payload(_payload(other), protocol.identity_sha256())
    assert not ok
    assert "protocol hash" in note


def test_missing_samples_rejected():
    protocol = TimingProtocol()
    ok, _ = validate_timing_payload(
        _payload(protocol, batch_samples_ms=[]), protocol.identity_sha256()
    )
    assert not ok


def test_bootstrap_ratio_ci_on_identical_populations_covers_one():
    protocol_a = TimingProtocol()
    samples_a = [1.0 + (i % 3) * 0.02 for i in range(12)]
    samples_b = [1.0 + ((i + 1) % 3) * 0.02 for i in range(12)]
    low, high = bootstrap_ratio_ci(samples_a, samples_b, seed=7)
    assert low <= 1.0 <= high
    assert validate_timing_payload(_payload(protocol_a), protocol_a.identity_sha256()) == (True, "")


def test_bootstrap_resampling_unit_is_batches_not_iterations():
    """12 batches: the ratio CI must not shrink below batch-level noise."""
    slow = [10.0 + i * 0.1 for i in range(12)]
    fast = [5.0 + i * 0.05 for i in range(12)]
    low, high = bootstrap_ratio_ci(slow, fast, seed=3)
    assert low > 1.5, "ratio of means must stay near 2 with batch-level resampling"
    assert high < 2.5
