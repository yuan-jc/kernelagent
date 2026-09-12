"""Autotune adapter (T14): winner selection and sweep audit semantics."""

from kernelagent.adapters.tuning import audit_sweep, select_winner


def _measured(block, median, correct=True):
    return {
        "status": "measured",
        "config": {"block_size": block},
        "correct": correct,
        "median_ms": median,
        "batch_samples_ms": [median, median, median],
    }


def test_winner_is_fastest_correct_config():
    records = [
        {"status": "pruned", "config": {"block_size": 3000}, "reason": "structurally invalid"},
        _measured(1024, 1.5),
        _measured(2048, 1.1),
        _measured(512, 1.9),
    ]
    assert select_winner(records) == {"block_size": 2048}


def test_incorrect_config_cannot_win():
    records = [
        _measured(1024, 0.5, correct=False),
        _measured(2048, 1.4),
    ]
    assert select_winner(records) == {"block_size": 2048}


def test_no_measured_config_means_no_winner():
    records = [
        {"status": "pruned", "config": {"block_size": 3000}},
        {"status": "incorrect", "config": {"block_size": 64}, "correct": False},
    ]
    assert select_winner(records) is None


def test_audit_requires_terminal_status_and_samples():
    ok, reason = audit_sweep(
        {
            "configs": [_measured(1024, 1.0)],
            "winner": {"block_size": 1024},
            "frozen_verified": True,
            "input_restore_events": 3,
        }
    )
    assert ok
    bad = {"configs": [{"status": "pending", "config": {}}]}
    ok, reason = audit_sweep(bad)
    assert not ok and "terminal" in reason


def test_audit_measured_without_samples_rejected():
    record = _measured(1024, 1.0)
    record["batch_samples_ms"] = []
    ok, reason = audit_sweep(
        {
            "configs": [record],
            "winner": {"block_size": 1024},
            "frozen_verified": True,
            "input_restore_events": 1,
        }
    )
    assert not ok and "raw batch samples" in reason


def test_audit_winner_must_be_frozen_verified():
    ok, reason = audit_sweep(
        {
            "configs": [_measured(1024, 1.0)],
            "winner": {"block_size": 1024},
            "frozen_verified": False,
            "input_restore_events": 1,
        }
    )
    assert not ok and "re-verification" in reason


def test_audit_winner_must_be_fastest():
    ok, reason = audit_sweep(
        {
            "configs": [_measured(1024, 2.0), _measured(2048, 1.0)],
            "winner": {"block_size": 1024},
            "frozen_verified": True,
            "input_restore_events": 2,
        }
    )
    assert not ok and "fastest" in reason


def test_audit_in_place_restoration_evidence_required():
    record = _measured(1024, 1.0)
    ok, reason = audit_sweep(
        {
            "configs": [record],
            "winner": {"block_size": 1024},
            "frozen_verified": True,
            "input_restore_events": 0,
        }
    )
    assert not ok and "restored-input" in reason
