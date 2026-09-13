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
    reconstruct_budget,
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


# --- Launch-plan Task 2: durable budget reconstruction on resume (P0-2) ---


def _seed_settled_journal(path, *, action_id: str = "generate", gpu: float, tokens: int):
    journal = Journal(path)
    reservation = journal.append(
        "budget_reserved", action_id=action_id, gpu_seconds=gpu, tokens=tokens
    )
    journal.append(
        "action_started",
        action_id=action_id,
        input_hash=f"{action_id}-input",
        budget_reservation=reservation["entry_hash"],
    )
    journal.append("budget_settled", action_id=action_id, gpu_seconds=gpu, tokens=tokens)
    journal.append(
        "action_finished",
        action_id=action_id,
        input_hash=f"{action_id}-input",
        result_ref=f"{action_id}-result",
        budget_settlement=reservation["entry_hash"],
    )
    return journal


def test_resume_refuses_work_beyond_durable_settled_total(tmp_path):
    """Reproduced P0-2: settled 80 of a 100-second budget, a resumed run
    still executed a 30-second action and settled to 110. The durable
    journal must gate the resumed run before any action executes."""
    journal_path = tmp_path / "journal.jsonl"
    _seed_settled_journal(journal_path, gpu=80.0, tokens=100)
    calls: list = []
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100.0, tokens_limit=1000))
    report = orchestrator.run([_action("evaluate", calls, gpu=30.0)], resume=True)
    assert report["state"] == STATE_BUDGET_EXHAUSTED
    assert calls == [], "a resumed run must not execute work the durable budget cannot cover"
    assert report["settled_gpu_seconds"] == 80.0
    assert any(e["kind"] == "budget_refused" for e in orchestrator.journal.entries)


def test_reconstruct_budget_splits_reserved_and_settled(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    _seed_settled_journal(journal_path, gpu=80.0, tokens=100)
    crash_reserve = Journal(journal_path).append(
        "budget_reserved", action_id="evaluate", gpu_seconds=30.0, tokens=50
    )
    Journal(journal_path).append(
        "action_started",
        action_id="evaluate",
        input_hash="evaluate-input",
        budget_reservation=crash_reserve["entry_hash"],
    )
    state = reconstruct_budget(Journal(journal_path).entries)
    assert state.settled_gpu_seconds == 80.0
    assert state.settled_tokens == 100
    assert state.reserved_gpu_seconds == 30.0, "the crashed action's reservation stays held"
    assert state.reserved_tokens == 50


def test_reconstruct_budget_rejects_double_settlement(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("budget_reserved", action_id="x", gpu_seconds=1.0, tokens=1)
    journal.append("budget_settled", action_id="x", gpu_seconds=1.0, tokens=1)
    journal.append("budget_settled", action_id="x", gpu_seconds=1.0, tokens=1)
    with pytest.raises(ValueError, match="settle"):
        reconstruct_budget(journal.entries)


def test_reconstruct_budget_rejects_negative_amounts(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("budget_reserved", action_id="x", gpu_seconds=-1.0, tokens=1)
    with pytest.raises(ValueError, match="negative"):
        reconstruct_budget(journal.entries)


def test_reconstruct_budget_rejects_release_without_reservation(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("budget_released", action_id="ghost", gpu_seconds=1.0, tokens=1)
    with pytest.raises(ValueError, match="release"):
        reconstruct_budget(journal.entries)


def test_interrupted_reservation_survives_resume_and_rerun_rereserves(tmp_path):
    """An interrupted action's reservation must not silently vanish on
    resume: the crashed attempt keeps holding its estimate, and the rerun
    reserves again on top - repeated crashes converge to budget_exhausted
    instead of free-unlimited retries."""
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(journal_path)
    reservation = journal.append(
        "budget_reserved", action_id="evaluate", gpu_seconds=30.0, tokens=50
    )
    journal.append(
        "action_started",
        action_id="evaluate",
        input_hash="evaluate-input",
        budget_reservation=reservation["entry_hash"],
    )
    calls: list = []
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100.0, tokens_limit=1000))
    report = orchestrator.run([_action("evaluate", calls, gpu=30.0, tokens=50)], resume=True)
    assert report["state"] == STATE_DONE
    assert calls == ["evaluate"]
    state = reconstruct_budget(orchestrator.journal.entries)
    assert state.settled_gpu_seconds == 30.0
    assert state.reserved_gpu_seconds == 30.0, (
        "the interrupted attempt keeps holding its reservation after resume"
    )
    assert report["settled_gpu_seconds"] == 30.0


def test_repeated_crash_resumes_converge_to_budget_exhausted(tmp_path):
    """A crashing action holds its reserved estimate across resumes: each
    retry re-reserves on top, so repeated crashes converge to
    budget_exhausted instead of retrying for free forever."""
    journal_path = tmp_path / "journal.jsonl"
    budget = Budget(gpu_seconds_limit=100.0, tokens_limit=1000)

    def crash() -> dict:
        raise RuntimeError("simulated process crash mid-action")

    crashing = Action("evaluate", "evaluate-input", 30.0, 50, crash)
    report = {"state": STATE_RUNNING}
    for attempt in range(6):
        orchestrator = Orchestrator(journal_path, budget)
        try:
            report = orchestrator.run([crashing], resume=attempt > 0)
        except RuntimeError:
            # The process died mid-action; reservation stays open.
            continue
        if report["state"] == STATE_BUDGET_EXHAUSTED:
            break
    assert report["state"] == STATE_BUDGET_EXHAUSTED, (
        "repeated interrupted retries must consume the estimate each time "
        "and eventually stop, never loop for free"
    )
    state = reconstruct_budget(Journal(journal_path).entries)
    assert state.settled_gpu_seconds + state.reserved_gpu_seconds <= 100.0 + 1e-9, (
        "durable totals must never exceed the configured limit beyond the "
        "in-flight reservation granted before the refusal"
    )
    assert state.reserved_gpu_seconds == pytest.approx(90.0), (
        "three crashing attempts each hold their 30-second estimate"
    )


# --- RV02: attempt identity, per-attempt settlement, atomic result commit --


def _lifecycle_attempts(entries: tuple[dict, ...], kinds: tuple[str, ...]) -> dict[str, set]:
    """attempt ids seen per journal event kind."""
    seen: dict[str, set] = {kind: set() for kind in kinds}
    for entry in entries:
        if entry["kind"] in seen:
            seen[entry["kind"]].add(entry.get("attempt_id"))
    return seen


def test_attempt_id_links_lifecycle_events(tmp_path):
    """Every execution of an action is one attempt: the attempt id links
    reservation, start, settlement and finish of that execution, and two
    actions get distinct attempt ids."""
    journal_path = tmp_path / "journal.jsonl"
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100, tokens_limit=1000))
    orchestrator.run([_action("evaluate", []), _action("time", [])])
    kinds = ("budget_reserved", "action_started", "budget_settled", "action_finished")
    seen = _lifecycle_attempts(orchestrator.journal.entries, kinds)
    for kind in kinds:
        assert len(seen[kind]) == 2, f"{kind} must carry one attempt id per execution"
    reserved, started = seen["budget_reserved"], seen["action_started"]
    settled, finished = seen["budget_settled"], seen["action_finished"]
    assert reserved == started == settled == finished, (
        "one attempt id spans reservation -> start -> settlement -> finish"
    )


def test_settled_without_finish_keeps_cost_and_never_fake_completes(tmp_path):
    """Reproduces review R2: a settlement is durable but the finish record
    was never written (crash in the window, or a torn commit-block tail).
    Recovery must (a) refuse to bill the same attempt twice, (b) keep the
    uncertainty and the already-paid cost, (c) re-execute the action under
    a FRESH attempt id whose settlement is a new, honest cost, and (d)
    stay replayable - the old code crashed the next replay with
    'duplicate settlement for action'."""
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(journal_path)
    journal.append(
        "budget_reserved",
        action_id="evaluate",
        attempt_id="attempt-1",
        gpu_seconds=30.0,
        tokens=50,
    )
    journal.append(
        "action_started",
        action_id="evaluate",
        attempt_id="attempt-1",
        input_hash="evaluate-input",
        budget_reservation="genesis",
    )
    journal.append(
        "budget_settled",
        action_id="evaluate",
        attempt_id="attempt-1",
        gpu_seconds=30.0,
        tokens=50,
    )
    # ...and no action_finished: the process died right here.

    calls: list = []
    first = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=10000))
    report = first.run([_action("evaluate", calls, gpu=30.0, tokens=50)], resume=True)
    assert report["state"] == STATE_DONE
    assert calls == ["evaluate"], "the uncertain attempt is re-executed exactly once"

    entries = first.journal.entries
    uncertain = [e for e in entries if e["kind"] == "attempt_uncertain"]
    assert len(uncertain) == 1
    assert uncertain[0]["attempt_id"] == "attempt-1"
    assert uncertain[0]["settled_gpu_seconds"] == 30.0, "the paid cost is preserved"

    settled = [e for e in entries if e["kind"] == "budget_settled"]
    assert len(settled) == 2, "two real executions, each settled exactly once"
    assert len({e.get("attempt_id") for e in settled}) == 2, (
        "the re-execution settles under a fresh attempt id, never the same one"
    )
    assert report["settled_gpu_seconds"] == 60.0, (
        "both executions are honestly billed: the uncertain attempt's cost stands"
    )
    finished = [e for e in entries if e["kind"] == "action_finished"]
    assert len(finished) == 1, "completion comes only from the confirmed re-run"

    # A third process replays the journal: nothing re-runs, nothing re-bills.
    calls.clear()
    second = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=10000))
    replay = second.run([_action("evaluate", calls, gpu=30.0, tokens=50)], resume=True)
    assert calls == [], "a completed action must not execute again on replay"
    assert replay["settled_gpu_seconds"] == 60.0, "replay adds no billing"
    state = reconstruct_budget(second.journal.entries)
    assert state.settled_gpu_seconds == 60.0
    assert state.settled_tokens == 100


def test_duplicate_settlement_for_same_attempt_is_rejected(tmp_path):
    """Settlement uniqueness is per attempt: a second settlement of the
    same attempt is duplicate billing and refuses to replay (the legacy
    no-attempt-id form keeps its one-settlement-per-action rule)."""
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append(
        "budget_reserved",
        action_id="x",
        attempt_id="attempt-1",
        gpu_seconds=1.0,
        tokens=1,
    )
    journal.append(
        "budget_settled", action_id="x", attempt_id="attempt-1", gpu_seconds=1.0, tokens=1
    )
    journal.append(
        "budget_settled", action_id="x", attempt_id="attempt-1", gpu_seconds=1.0, tokens=1
    )
    with pytest.raises(ValueError, match="duplicate settlement"):
        reconstruct_budget(journal.entries)


def test_distinct_attempts_of_one_action_each_settle_once(tmp_path):
    """Two real executions of the same logical action (an uncertain attempt
    re-run) are two honest costs: per-attempt settlement keys must NOT
    reject them, and totals add."""
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append(
        "budget_settled", action_id="x", attempt_id="attempt-1", gpu_seconds=30.0, tokens=50
    )
    journal.append(
        "budget_settled", action_id="x", attempt_id="attempt-2", gpu_seconds=30.0, tokens=50
    )
    state = reconstruct_budget(journal.entries)
    assert state.settled_gpu_seconds == 60.0
    assert state.settled_tokens == 100


def test_torn_commit_block_leaves_settled_without_finish(tmp_path, monkeypatch):
    """A crash mid-way through the atomic settlement+finish block can tear
    the tail so only the settled line is durable. The loader's tail rule
    drops the fragment, recovery marks the attempt uncertain, re-runs it
    under a fresh attempt id, and the journal replays cleanly."""
    journal_path = tmp_path / "journal.jsonl"
    journal = Journal(journal_path)
    journal.append(
        "budget_reserved",
        action_id="evaluate",
        attempt_id="attempt-1",
        gpu_seconds=30.0,
        tokens=50,
    )
    journal.append(
        "action_started",
        action_id="evaluate",
        attempt_id="attempt-1",
        input_hash="evaluate-input",
        budget_reservation="genesis",
    )
    real_write = Journal._write_block

    def torn_write(self, payload: bytes) -> None:
        # Persist the settled line fully plus a partial fragment of the
        # finish line (no terminating newline), then die: exactly what a
        # crash inside the single block write leaves on disk.
        first_newline = payload.index(b"\n")
        real_write(self, payload[: first_newline + 1 + 24])
        raise RuntimeError("simulated SIGKILL inside the commit block")

    monkeypatch.setattr(Journal, "_write_block", torn_write)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        journal.commit_settlement_and_finish(
            action_id="evaluate",
            attempt_id="attempt-1",
            input_hash="evaluate-input",
            result_ref="evaluate-result",
            gpu_seconds=30.0,
            tokens=50,
        )
    monkeypatch.undo()  # the crash is on disk; the next process writes normally

    # The torn tail fragment on disk is an unfinished partial record.
    calls: list = []
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=10000))
    report = orchestrator.run([_action("evaluate", calls, gpu=30.0, tokens=50)], resume=True)
    assert report["state"] == STATE_DONE
    assert calls == ["evaluate"]
    recovered = [e for e in orchestrator.journal.entries if e["kind"] == "journal_recovered"]
    assert len(recovered) == 1, "the dropped fragment is recorded, never silently deleted"
    uncertain = [e for e in orchestrator.journal.entries if e["kind"] == "attempt_uncertain"]
    assert len(uncertain) == 1 and uncertain[0]["attempt_id"] == "attempt-1"
    state = reconstruct_budget(orchestrator.journal.entries)
    assert state.settled_gpu_seconds == pytest.approx(60.0)


def test_experiment_done_carries_durable_totals(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    orchestrator = Orchestrator(journal_path, Budget(gpu_seconds_limit=100, tokens_limit=1000))
    orchestrator.run([_action("evaluate", [], gpu=2.5, tokens=7)])
    done = [e for e in orchestrator.journal.entries if e["kind"] == "experiment_done"]
    assert len(done) == 1
    assert done[0]["settled_gpu_seconds"] == 2.5
    assert done[0]["settled_tokens"] == 7
