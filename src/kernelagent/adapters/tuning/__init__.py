"""Autotune adapter control plane (T14, design §8.4).

Pure decision logic lives here: winner selection over per-config
measurement records (correct configs only, lowest median batch time),
and the frozen-config verification gate. The GPU sweep itself runs in
configs/kernelbench/autotune_driver.py inside the ADR-0001 boundary."""

from __future__ import annotations


def select_winner(records: list[dict]) -> dict | None:
    """Winner = the correct, measured config with the lowest median batch
    time. Pruned or incorrect configs can never win; an empty measured set
    yields None (no winner is an honest outcome)."""
    measured = [
        record
        for record in records
        if record.get("status") == "measured"
        and record.get("correct") is True
        and isinstance(record.get("median_ms"), (int, float))
    ]
    if not measured:
        return None
    return min(measured, key=lambda record: record["median_ms"])["config"]


def audit_sweep(payload: dict) -> tuple[bool, str]:
    """Structural audit of a sweep report: every config reaches exactly one
    terminal status, measured configs carry raw samples, the winner (if
    any) is a measured config, and a winner must be frozen-verified."""
    records = payload.get("configs")
    if not isinstance(records, list) or not records:
        return False, "sweep has no config records"
    winner = payload.get("winner")
    measured = [record for record in records if record.get("status") == "measured"]
    for record in records:
        status = record.get("status")
        if status not in ("pruned", "incorrect", "measured"):
            return False, f"config {record.get('config')!r} has terminal status {status!r}"
        if status == "measured":
            if record.get("correct") is not True:
                return False, "a measured config must be correct"
            samples = record.get("batch_samples_ms")
            if not isinstance(samples, list) or not samples:
                return False, "measured config without raw batch samples"
    if winner is not None:
        matching = [r for r in measured if r.get("config") == winner]
        if not matching:
            return False, "winner is not among measured configs"
        best = min(r["median_ms"] for r in measured)
        if matching[0]["median_ms"] != best:
            return False, "winner is not the fastest correct config"
        if payload.get("frozen_verified") is not True:
            return False, "frozen winner failed re-verification"
    if payload.get("input_restore_events", 0) < 1 and measured:
        return False, "in-place candidates require restored-input evidence"
    return True, ""


def cutlass_ok(cutlass: dict) -> bool:
    """T20 CUTLASS template acceptance: the path counts only when the
    template compiled, executed, and self-verified (its stdout contains
    the cutlass-ok marker)."""
    return (
        isinstance(cutlass, dict)
        and cutlass.get("compiled") is True
        and cutlass.get("ran") is True
        and "cutlass-ok" in cutlass.get("stdout_tail", "")
    )
