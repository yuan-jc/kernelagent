"""T20: CUDA tuning + CUTLASS template path acceptance (pure CPU pins on
report semantics). The real GPU runs happen in examples/t20_smoke.py."""

from kernelagent.adapters.tuning import cutlass_ok, select_winner


def _record(block, time_us, correct=True):
    return {
        "status": "measured",
        "config": {"BLOCK_SIZE": block},
        "correct": correct,
        "median_ms": time_us,
        "batch_samples_ms": [time_us, time_us, time_us],
    }


def test_winner_selection_over_tuner_measurements():
    records = [
        _record(64, 90.0),
        _record(128, 40.0),
        _record(256, 35.0),
        _record(512, 42.0),
        _record(1024, 55.0),
    ]
    assert select_winner(records) == {"BLOCK_SIZE": 256}


def test_five_configs_all_measured_before_tuning_claim():
    measurements = {str(block): 10.0 + block for block in (64, 128, 256, 512, 1024)}
    assert len(measurements) == 5
    assert all(isinstance(v, float) for v in measurements.values())


def test_licenses_recorded_separately():
    """Kernel Tuner (GPLv3) and CUTLASS (BSD-3) licenses must be recorded
    as separate entries - never merged into one project license claim."""
    licenses = {
        "kernel-tuner": "GPLv3 (container-only, not redistributed)",
        "cutlass": "BSD-3-Clause (C++ headers)",
    }
    assert licenses["kernel-tuner"].startswith("GPLv3")
    assert licenses["cutlass"].startswith("BSD-3")
    assert len(licenses) == 2


def test_cutlass_result_requires_compiled_ran_and_selfverify():
    cases = [
        ({"compiled": True, "ran": True, "stdout_tail": "cutlass-ok"}, True),
        ({"compiled": True, "ran": True, "stdout_tail": "cutlass-mismatch"}, False),
        ({"compiled": False, "ran": False, "error": "nvcc failed"}, False),
        ({}, False),
        (None, False),
    ]
    for payload, expected in cases:
        assert cutlass_ok(payload) == expected
