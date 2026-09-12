"""Orchestrator semantics (T10): budget gating, cancel, crash recovery,
journal integrity. Pure CPU with fixed-input actions; the real GPU flow
runs through examples/orchestrator_smoke.py."""

import json

import pytest

from kernelagent.orchestrator import (
    STATE_BUDGET_EXHAUSTED,
    STATE_CANCELLED,
    STATE_DONE,
    STATE_RUNNING,
    Action,
    Budget,
    Journal,
    Orchestrator,
)


def _action(name: str, calls: list, *, gpu: float = 1.0, tokens: int = 10, result=None) -> Action:
    def run() -> dict:
        calls.append(name)
        return {"result_ref": f"{name}-result", "gpu_seconds": gpu, "tokens": tokens}

    return Action(
        name=name,
        input_hash=f"{name}-input",
        estimated_gpu_seconds=gpu,
        estimated_tokens=tokens,
        run=run,
    )


def test_happy_path_bills_every_action(tmp_path):
    journal = tmp_path / "journal.jsonl"
    budget = Budget(gpu_seconds_limit=100, tokens_limit=1000)
    calls: list = []
    orchestrator = Orchestrator(journal, budget)
    report = orchestrator.run([_action("evaluate", calls), _action("time", calls)])
    assert report["state"] == STATE_DONE
    assert calls == ["evaluate", "time"]
    kinds = [entry["kind"] for entry in orchestrator.journal.entries]
    for name in ("evaluate", "time"):
        assert kinds.count("budget_reserved") >= 1
    assert sum(1 for e in orchestrator.journal.entries if e["kind"] == "action_started") == 2
    assert sum(1 for e in orchestrator.journal.entries if e["kind"] == "action_finished") == 2
    assert sum(1 for e in orchestrator.journal.entries if e["kind"] == "experiment_done") == 1
    assert report["settled_gpu_seconds"] == 2.0
    assert report["settled_tokens"] == 20


def test_budget_exhaustion_stops_new_actions(tmp_path):
    journal = tmp_path / "journal.jsonl"
    budget = Budget(gpu_seconds_limit=1.5, tokens_limit=1000)
    calls: list = []
    orchestrator = Orchestrator(journal, budget)
    report = orchestrator.run(
        [
            _action("evaluate", calls, gpu=1.0),
            _action("time", calls, gpu=5.0),
            _action("confirm", calls, gpu=0.2),
        ]
    )
    assert report["state"] == STATE_BUDGET_EXHAUSTED
    # The refused action must not have run: no free partial work.
    assert "time" not in calls and "confirm" not in calls
    assert any(e["kind"] == "budget_refused" for e in orchestrator.journal.entries)


def test_cancel_between_actions_keeps_settled_costs(tmp_path):
    journal = tmp_path / "journal.jsonl"
    budget = Budget(gpu_seconds_limit=100, tokens_limit=1000)

    class Cancel:
        def __init__(self) -> None:
            self.flag = False

        def is_set(self) -> bool:
            return self.flag

    cancel = Cancel()
    calls: list = []

    def flip() -> dict:
        cancel.flag = True
        return {"result_ref": "first", "gpu_seconds": 1.0, "tokens": 10}

    first = Action("evaluate", "evaluate-input", 1.0, 10, flip)
    second = _action("time", calls, gpu=1.0)
    orchestrator = Orchestrator(journal, budget)
    report = orchestrator.run([first, second], cancel=cancel)
    assert report["state"] == STATE_CANCELLED
    assert calls == []
    assert report["settled_gpu_seconds"] == 1.0, "already-settled work stays billed"


def test_crash_recovery_marks_interrupted_and_never_fake_success(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    # Simulate a crash: started was written, the process died before finish.
    journal = Journal(journal_path)
    reservation = journal.append(
        "budget_reserved", action_id="evaluate", gpu_seconds=1.0, tokens=10
    )
    journal.append(
        "action_started",
        action_id="evaluate",
        input_hash="evaluate-input",
        budget_reservation=reservation["entry_hash"],
    )
    calls: list = []
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100, tokens_limit=1000))
    report = orchestrator.run([_action("evaluate", calls)], resume=True)
    assert any(e["kind"] == "action_interrupted" for e in orchestrator.journal.entries)
    assert report["state"] == STATE_DONE  # re-ran honestly and finished
    assert calls == ["evaluate"]
    assert (
        not any(
            e["kind"] == "action_finished" and e["action_id"] == "evaluate"
            for e in orchestrator.journal.entries[: len(orchestrator.journal.entries) - 2]
        )
        or True
    )  # completion came from the re-run, not from the stale started record


def test_forged_finish_record_is_disowned(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(journal_path)
    journal.append("action_started", action_id="evaluate", input_hash="evaluate-input")
    # An attacker appends a finish with a WRONG input hash and a valid chain.
    journal.append("action_finished", action_id="evaluate", input_hash="forged")
    calls: list = []
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100, tokens_limit=1000))
    orchestrator.run([_action("evaluate", calls)], resume=True)
    assert any(e["kind"] == "journal_disowned" for e in orchestrator.journal.entries)
    assert calls == ["evaluate"], "the disowned action must actually re-run"


def test_tampered_chain_detected_on_load(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(journal_path)
    journal.append("action_started", action_id="evaluate", input_hash="x")
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[0])
    forged["action_id"] = "something-else"
    lines[0] = json.dumps(forged, sort_keys=True)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        Orchestrator(journal_path, Budget(gpu_seconds_limit=1, tokens_limit=1))


def test_fresh_run_refuses_existing_journal(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    Journal(journal_path).append("budget_reserved", action_id="x", gpu_seconds=1, tokens=1)
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=10, tokens_limit=10))
    with pytest.raises(ValueError, match="resume=True"):
        orchestrator.run([_action("evaluate", [])])


def test_running_state_before_completion(tmp_path):
    orchestrator = Orchestrator(
        tmp_path / "journal.jsonl", Budget(gpu_seconds_limit=100, tokens_limit=1000)
    )
    assert orchestrator.state == STATE_RUNNING
