"""RV02 crash-consistency acceptance: fault injection at every journal
write boundary, repeated recovery, the single-writer resume lock, and
journal integrity (truncated tail vs mid-file corruption).

All tests are pure CPU. A crash is simulated at the exact durable-write
boundary a SIGKILL would hit, by monkeypatching the journal's write seams
(``Journal.append`` / ``Journal.commit_settlement_and_finish`` /
``Journal._write_block``); every recovery round builds a fresh
Orchestrator / ``optimize()`` call - a new process view over the same
on-disk journal. The controlled-SIGKILL re-validation on Ubuntu with real
GPU workers is RV07's job and stays NOT_RUN here.

Semantics under test (see kernelagent.orchestrator module docstring):
billing is exactly-once per attempt; execution is at-least-once - an
attempt whose completion cannot be decided keeps its cost and an
``attempt_uncertain`` marker, never a fake completion, and the action is
re-executed under a fresh attempt id."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kernelagent.adapters.models.client import ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.optimization import (
    RUN_LOCK_FILENAME,
    OptimizationConfig,
    RunLockHeldError,
    acquire_run_lock,
    optimize,
)
from kernelagent.orchestrator import (
    Action,
    Budget,
    Journal,
    Orchestrator,
    reconstruct_budget,
    settled_metering_split,
)

# ------------------------------------------------------------------------
# Orchestrator-level fault injection: reservation / start / result /
# settlement / finish write boundaries, two consecutive recoveries each.
# ------------------------------------------------------------------------


def _tracked_action(calls: list, *, gpu: float = 30.0, tokens: int = 50) -> Action:
    def run() -> dict:
        calls.append("exec")
        return {"result_ref": "evaluate-result", "gpu_seconds": gpu, "tokens": tokens}

    return Action("evaluate", "evaluate-input", gpu, tokens, run)


def _crash_after_append(target_kind: str):
    """SIGKILL immediately after ``target_kind`` became durable."""
    original = Journal.append

    def patched(self, kind, **fields):
        entry = original(self, kind, **fields)
        if kind == target_kind:
            raise RuntimeError(f"simulated SIGKILL after {kind}")
        return entry

    return patched


def _crash_before_result_commit(self, **kwargs):
    raise RuntimeError("simulated SIGKILL after the action ran, before the result commit")


def _crash_inside_commit(mode: str):
    """SIGKILL inside the atomic settlement+finish block write: 'settlement'
    leaves the settled line (plus a partial finish fragment) durable - the
    exact torn-tail shape of the R2 window; 'finish' persists the whole
    block and dies right after the finish became durable. Only the commit
    block's write is intercepted; earlier appends (reserve, start) must
    land normally."""
    real_write = Journal._write_block
    real_commit = Journal.commit_settlement_and_finish

    def torn_write(self, payload: bytes) -> None:
        if mode == "settlement":
            first_newline = payload.index(b"\n")
            real_write(self, payload[: first_newline + 1 + 24])
        else:
            real_write(self, payload)
        raise RuntimeError("simulated SIGKILL inside the commit block")

    def patched_commit(self, **kwargs):
        Journal._write_block = torn_write
        try:
            return real_commit(self, **kwargs)
        finally:
            Journal._write_block = real_write

    return patched_commit


def _apply_injection(monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    if boundary == "reservation":
        monkeypatch.setattr(Journal, "append", _crash_after_append("budget_reserved"))
    elif boundary == "start":
        monkeypatch.setattr(Journal, "append", _crash_after_append("action_started"))
    elif boundary == "result":
        monkeypatch.setattr(Journal, "commit_settlement_and_finish", _crash_before_result_commit)
    else:
        monkeypatch.setattr(Journal, "commit_settlement_and_finish", _crash_inside_commit(boundary))


# Per-boundary expectations after two consecutive recoveries. The action
# estimates/settles 30 GPU-seconds and 50 tokens per execution.
_EXPECTATIONS = {
    # boundary: (executions, started attempts, settlements, settled_gpu,
    #            settled_tokens, interruption event, journal_recovered,
    #            reserved_gpu_held)
    "reservation": (1, 1, 1, 30.0, 50, None, 0, 30.0),
    "start": (1, 2, 1, 30.0, 50, "action_interrupted", 0, 30.0),
    "result": (2, 2, 1, 30.0, 50, "action_interrupted", 0, 30.0),
    "settlement": (2, 2, 2, 60.0, 100, "attempt_uncertain", 1, 0.0),
    "finish": (1, 1, 1, 30.0, 50, None, 0, 0.0),
}


@pytest.mark.parametrize("boundary", list(_EXPECTATIONS))
def test_crash_at_write_boundary_recovers_twice_without_double_billing(
    tmp_path: Path, boundary: str
):
    """Crash at each journal write boundary, recover twice. Invariants:
    each attempt settles at most once (no duplicate billing), execution
    counts stay honest, the uncertainty of a settled-but-unfinished
    attempt is preserved, and the final journal replays cleanly."""
    journal_path = tmp_path / "journal.jsonl"
    calls: list = []

    with pytest.MonkeyPatch.context() as mp:
        _apply_injection(mp, boundary)
        with pytest.raises(RuntimeError, match="SIGKILL"):
            Orchestrator(journal_path, Budget(gpu_seconds_limit=1000, tokens_limit=10000)).run(
                [_tracked_action(calls)], resume=False
            )

    for _ in range(2):  # two consecutive recovery rounds ("processes")
        Orchestrator(journal_path, Budget(gpu_seconds_limit=1000, tokens_limit=10000)).run(
            [_tracked_action(calls)], resume=True
        )

    (
        executions,
        started,
        settlements,
        settled_gpu,
        settled_tokens,
        interruption,
        recovered,
        reserved_held,
    ) = _EXPECTATIONS[boundary]
    assert calls.count("exec") == executions, "execution count must match reality"

    journal = Journal(journal_path)
    entries = journal.entries
    settled_entries = [e for e in entries if e["kind"] == "budget_settled"]
    assert len(settled_entries) == settlements
    attempt_ids = [e.get("attempt_id") for e in settled_entries]
    assert len(set(attempt_ids)) == len(attempt_ids), (
        "one attempt is billed at most once - the review R2 duplicate settlement"
    )
    started_entries = [e for e in entries if e["kind"] == "action_started"]
    assert len(started_entries) == started
    assert all(e.get("attempt_id") for e in started_entries), "every attempt carries its identity"
    assert sum(e["gpu_seconds"] for e in settled_entries) == pytest.approx(settled_gpu)
    assert sum(e["tokens"] for e in settled_entries) == settled_tokens

    kinds = [e["kind"] for e in entries]
    if interruption is None:
        assert "action_interrupted" not in kinds and "attempt_uncertain" not in kinds
    else:
        assert kinds.count(interruption) == 1
    assert kinds.count("journal_recovered") == recovered

    state = reconstruct_budget(entries)
    assert state.settled_gpu_seconds == pytest.approx(settled_gpu)
    assert state.reserved_gpu_seconds == pytest.approx(reserved_held), (
        "a crashed attempt keeps holding its estimate; a settled one releases it"
    )

    done = [e for e in entries if e["kind"] == "experiment_done"]
    assert len(done) == 1, "recovery ends in the honest terminal event"


def test_settlement_crash_then_rerun_is_never_a_duplicate_settlement(tmp_path: Path):
    """The exact review R2 counterexample, end to end: crash after the
    settlement became durable but before the finish, resume, re-execute,
    resume again - the second replay must NOT raise 'duplicate settlement'
    and must not bill any attempt twice."""
    journal_path = tmp_path / "journal.jsonl"
    calls: list = []
    budget = Budget(gpu_seconds_limit=1000.0, tokens_limit=10000)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Journal, "commit_settlement_and_finish", _crash_inside_commit("settlement"))
        with pytest.raises(RuntimeError, match="SIGKILL"):
            Orchestrator(journal_path, budget).run([_tracked_action(calls)], resume=False)

    first = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=10000))
    report = first.run([_tracked_action(calls)], resume=True)  # recovery: re-executes
    assert report["state"] == "completed"
    second = Orchestrator(journal_path, Budget(gpu_seconds_limit=1000.0, tokens_limit=10000))
    replay = second.run([_tracked_action(calls)], resume=True)  # replay: adds nothing
    assert replay["settled_gpu_seconds"] == report["settled_gpu_seconds"] == pytest.approx(60.0)
    assert calls.count("exec") == 2, "one execution per uncertain attempt, honestly billed"


# ------------------------------------------------------------------------
# Journal integrity: truncated tail vs corruption anywhere else.
# ------------------------------------------------------------------------


def _seed_journal(path: Path, entries: int = 2) -> None:
    journal = Journal(path)
    for index in range(entries):
        journal.append("budget_reserved", action_id=f"a{index}", gpu_seconds=1.0, tokens=1)


def test_truncated_tail_is_dropped_recorded_and_run_continues(tmp_path: Path):
    """The documented tail rule: a final fragment without its terminating
    newline is an uncommitted partial write - dropped, its content hash
    preserved in a journal_recovered event, and the journal stays usable
    (never silently rewritten)."""
    journal_path = tmp_path / "journal.jsonl"
    _seed_journal(journal_path)
    fragment = b'{"kind": "budget_settled", "action_id": "a1", "gpu_sec'
    with journal_path.open("ab") as handle:
        handle.write(fragment)  # no terminating newline: uncommitted tail

    journal = Journal(journal_path)  # load performs the documented recovery
    entries = journal.entries
    assert len([e for e in entries if e["kind"] == "budget_reserved"]) == 2
    recovered = [e for e in entries if e["kind"] == "journal_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["dropped_bytes"] == len(fragment)
    assert recovered[0]["dropped_sha256"] == hashlib.sha256(fragment).hexdigest()

    raw = journal_path.read_bytes()
    assert fragment not in raw, "the truncated fragment itself is truncated away"
    assert raw.endswith(b"\n")
    next_hash = journal.append("budget_reserved", action_id="a2", gpu_seconds=1.0, tokens=1)
    assert next_hash["prev_hash"] == entries[-1]["entry_hash"], "the chain continues"
    reconstruct_budget(Journal(journal_path).entries)  # still replayable


def test_midfile_corruption_refuses_to_start(tmp_path: Path):
    """A damaged record that is NOT the uncommitted tail must refuse to
    load - never silently deleted, never skipped."""
    journal_path = tmp_path / "journal.jsonl"
    _seed_journal(journal_path)
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[0])
    forged["gpu_seconds"] = 999.0
    lines[0] = json.dumps(forged, sort_keys=True)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        Journal(journal_path)


def test_unparseable_complete_line_refuses_to_start(tmp_path: Path):
    journal_path = tmp_path / "journal.jsonl"
    _seed_journal(journal_path)
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{corrupted payload")
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        Journal(journal_path)


def test_complete_last_line_with_bad_hash_is_not_a_tail(tmp_path: Path):
    """The tail rule only forgives an UNTERMINATED fragment: a complete
    last record whose hash does not match is corruption or tampering and
    refuses to load."""
    journal_path = tmp_path / "journal.jsonl"
    _seed_journal(journal_path)
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[-1])
    forged["tokens"] = 42
    lines[-1] = json.dumps(forged, sort_keys=True)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        Journal(journal_path)


# ------------------------------------------------------------------------
# Single-writer resume lock (concurrent resume counterexample).
# ------------------------------------------------------------------------


def test_run_lock_rejects_second_holder_in_process(tmp_path: Path):
    lock = acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RunLockHeldError, match="single-writer"):
            acquire_run_lock(tmp_path)
    finally:
        lock.release()
    successor = acquire_run_lock(tmp_path)  # released lock is takeable again
    successor.release()


def test_run_lock_records_owner_pid(tmp_path: Path):
    lock = acquire_run_lock(tmp_path)
    try:
        content = lock.path.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in content
    finally:
        lock.release()


def test_concurrent_resume_is_refused_and_touches_nothing(tmp_path: Path):
    """Counterexample for concurrent recovery: while another process holds
    the run lock, a second resume must fail loudly BEFORE any durable
    state is read, written, or billed."""
    snapshot, result = _completed_run(tmp_path)
    output = tmp_path / "run"
    journal_bytes = (output / "journal.jsonl").read_bytes()

    holder_script = (
        "import sys, time\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "try:\n"
        "    import fcntl\n"
        "\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except ImportError:  # Windows: lock the hint byte, past the pid header\n"
        "    import msvcrt, os\n"
        "\n"
        "    os.lseek(handle.fileno(), 4096, 0)\n"
        "    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)\n"
        "print('locked', flush=True)\n"
        "time.sleep(30)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(output / RUN_LOCK_FILENAME)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        responder = ScriptedResponder([])
        config = _config(
            tmp_path,
            snapshot,
            max_candidates=1,
            resume=True,
            output=output,
        )
        with pytest.raises(RunLockHeldError, match="owned by another process"):
            _run(config, responder)
        assert responder.calls == 0, "a refused resume must make no model call"
        assert (output / "journal.jsonl").read_bytes() == journal_bytes, (
            "a refused resume must not touch the journal"
        )
    finally:
        holder.kill()
        holder.wait(timeout=10)

    # After the owner died, the OS-released lock allows recovery again.
    successor = acquire_run_lock(output)
    successor.release()


# ------------------------------------------------------------------------
# Optimization-loop-level: a full optimize run crashed at the settlement
# boundary, recovered twice, billed honestly, and finished.
# ------------------------------------------------------------------------


def test_optimize_settlement_crash_recovers_twice_and_never_double_bills(tmp_path: Path):
    snapshot = _stage_problem(tmp_path)
    output = tmp_path / "run"
    real_write = Journal._write_block

    def torn_settlement(self, payload: bytes) -> None:
        if b'"kind": "budget_settled"' in payload and b'"candidate-000"' in payload:
            first_newline = payload.index(b"\n")
            real_write(self, payload[: first_newline + 1 + 24])
            raise RuntimeError("simulated SIGKILL in the settlement commit")
        real_write(self, payload)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Journal, "_write_block", torn_settlement)
        responder1 = ScriptedResponder([GOOD_CANDIDATE])
        with pytest.raises(RuntimeError, match="SIGKILL"):
            _run(_config(tmp_path, snapshot, output=output, max_candidates=1), responder1)
    assert responder1.calls == 1, "the crashed attempt did generate and must be billed for it"

    responder2 = ScriptedResponder([GOOD_CANDIDATE])
    result = _run(
        _config(tmp_path, snapshot, output=output, max_candidates=1, resume=True), responder2
    )
    assert result.state == "completed", "the uncertain candidate is re-run and promotes"
    assert responder2.calls == 1, "the re-run generates under a fresh attempt"

    responder3 = ScriptedResponder([])
    result2 = _run(
        _config(tmp_path, snapshot, output=output, max_candidates=1, resume=True), responder3
    )
    assert responder3.calls == 0, "the third process replays without executing anything"
    assert result2.state == "completed"
    assert result2.champion_sha256 == result.champion_sha256

    entries = [
        json.loads(line)
        for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_settled = [
        e for e in entries if e["kind"] == "budget_settled" and e["action_id"] == "candidate-000"
    ]
    assert len(candidate_settled) == 2, "two real executions, each billed once"
    assert len({e["attempt_id"] for e in candidate_settled}) == 2, (
        "distinct attempts - never the same attempt billed twice"
    )
    uncertain = [e for e in entries if e["kind"] == "attempt_uncertain"]
    assert len(uncertain) == 1, "the settled-without-finish attempt keeps its marker"
    state = reconstruct_budget(Journal(output / "journal.jsonl").entries)
    assert state.settled_tokens == 60, "baseline 0 + two generations x 30 tokens"
    # RV04: settlements are the MEASURED offline lease times (well below
    # one old fixed 300-second quota), each labeled actual - never the
    # retired fixed quota. All three executions are real work, so every
    # settled GPU second is measured, none estimated.
    assert 0.0 <= state.settled_gpu_seconds < 300.0
    actual, estimated = settled_metering_split(Journal(output / "journal.jsonl").entries)
    assert actual == pytest.approx(state.settled_gpu_seconds)
    assert estimated == 0.0
    report = json.loads(result2.report_path.read_text(encoding="utf-8"))
    assert report["budget"]["settled_tokens"] == 60
    assert len(report["candidates"]) == 1


# ------------------------------------------------------------------------
# Local copies of the offline loop fixtures (tests/ is not a package; the
# originals live in tests/test_optimization_loop.py).
# ------------------------------------------------------------------------


class ScriptedResponder:
    """Order-based model double that builds each response against the
    actual request (so hashes validate) and records every request seen."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        content = self._contents.pop(0)
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=content,
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20),
        )


class RecordingLedger:
    def __init__(self):
        self._total_tokens = 0

    def record(self, response):
        self._total_tokens += response.usage.total_tokens

    def total_tokens(self) -> int:
        return self._total_tokens


def candidate_module(body: str) -> str:
    return json.dumps({"code": f"class ModelNew:\n    {body}\n"})


GOOD_CANDIDATE = candidate_module("def forward(self, x):\n        return x")


def _stage_problem(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text(
        "# pinned problem\n", encoding="utf-8"
    )
    return snapshot


def _passing_evaluate(candidate_source: str, candidate_name: str) -> dict:
    passed = "return x" in candidate_source
    return {
        "adapter_pass": passed,
        "compiled": passed,
        "correct": passed,
        "outcome_status": "completed",
        "stderr_tail": "" if passed else "assert torch.allclose failed",
        "upstream_metadata": "" if passed else "max abs diff 1.0",
    }


def _fast_candidate_timing(candidate_source: str, role: str) -> dict:
    if role == "eager":
        return {
            "source_ok": True,
            "precondition": True,
            "async_leak": False,
            "batches": [10.0] * 12,
        }
    return {
        "source_ok": True,
        "precondition": True,
        "async_leak": False,
        "batches": [5.0] * 12,
    }


def _config(tmp_path: Path, snapshot: Path, **overrides) -> OptimizationConfig:
    defaults = dict(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=3,
        max_repair_rounds=2,
        gpu_budget_seconds=3600.0,
        token_budget=100000,
        output=tmp_path / "run",
        resume=False,
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _run(config, responder):
    ledger = RecordingLedger()
    generator = CandidateGenerator(responder, ledger=ledger)
    return optimize(
        config,
        generator=generator,
        evaluate=_passing_evaluate,
        timing=_fast_candidate_timing,
        correctness_pro=None,
    )


def _completed_run(tmp_path: Path) -> tuple[Path, object]:
    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([GOOD_CANDIDATE])
    config = _config(
        tmp_path,
        snapshot,
        max_candidates=1,
        max_repair_rounds=0,
    )
    result = _run(config, responder)
    assert result.state == "completed"
    return snapshot, result
