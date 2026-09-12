"""Experiment state machine with budget accounting and crash recovery
(T10, design §7 and §12).

The orchestrator executes an ordered list of actions and records every
transition in an append-only JSONL journal:

- ``action_started``  - written BEFORE the action runs, with the action
  name, an input-identity hash, and the budget reservation;
- ``action_finished`` - written after the action returns, with a result
  reference and the settled cost;
- ``action_interrupted`` - written by recovery for a started action that
  has no matching finish: an interrupted action is re-runnable and NEVER
  counts as completed;
- ``budget_reserved`` / ``budget_settled`` - every action pays: reserve
  before start, settle actuals after;
- ``experiment_done`` - terminal, written only when every action finished.

A hash chain over journal entries (each entry records the sha256 of the
previous one) makes retro-forgery detectable: a fabricated
``action_finished`` breaks the chain or references an input hash that the
recovery replay can disown. Cancel is honoured at action boundaries;
between actions the machine stops with state ``cancelled`` and keeps the
settled costs. When the budget cannot cover a reservation the machine
refuses to start the next action and ends ``budget_exhausted`` - no
partial free work.

The state machine is pure control plane: actions are callables supplied
by the caller (the T10 smoke wires the real T05/T07/T09 adapters)."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

STATE_RUNNING = "running"
STATE_DONE = "completed"
STATE_CANCELLED = "cancelled"
STATE_BUDGET_EXHAUSTED = "budget_exhausted"


def _hash_entry(entry: dict) -> str:
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Journal:
    """Append-only JSONL journal with a per-entry hash chain."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: list[dict] = []
        self._chain = "genesis"
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            position = len(self._entries)
            stored = entry.get("entry_hash")
            payload = {k: v for k, v in entry.items() if k != "entry_hash"}
            if stored != _hash_entry(payload):
                raise ValueError(f"journal entry hash mismatch at entry {position}")
            if entry.get("prev_hash") != self._chain:
                raise ValueError(f"journal hash chain broken at entry {position}")
            self._chain = stored
            self._entries.append(entry)

    @property
    def entries(self) -> tuple[dict, ...]:
        return tuple(self._entries)

    def append(self, kind: str, **fields) -> dict:
        entry = {"kind": kind, "prev_hash": self._chain, **fields}
        entry["entry_hash"] = _hash_entry(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        self._chain = entry["entry_hash"]
        self._entries.append(entry)
        return entry


@dataclass
class Budget:
    """GPU-seconds / token budget: reserve before starting, settle after."""

    gpu_seconds_limit: float
    tokens_limit: int
    reserved_gpu_seconds: float = 0.0
    reserved_tokens: int = 0
    settled_gpu_seconds: float = 0.0
    settled_tokens: int = 0
    events: list = field(default_factory=list, repr=False)

    def try_reserve(
        self, journal: Journal, action_id: str, gpu_seconds: float, tokens: int
    ) -> dict | None:
        if (
            self.reserved_gpu_seconds + gpu_seconds > self.gpu_seconds_limit
            or self.reserved_tokens + tokens > self.tokens_limit
        ):
            return None
        self.reserved_gpu_seconds += gpu_seconds
        self.reserved_tokens += tokens
        entry = journal.append(
            "budget_reserved",
            action_id=action_id,
            gpu_seconds=gpu_seconds,
            tokens=tokens,
        )
        self.events.append(entry)
        return entry

    def settle(self, journal: Journal, action_id: str, gpu_seconds: float, tokens: int) -> dict:
        self.settled_gpu_seconds += gpu_seconds
        self.settled_tokens += tokens
        entry = journal.append(
            "budget_settled",
            action_id=action_id,
            gpu_seconds=gpu_seconds,
            tokens=tokens,
        )
        self.events.append(entry)
        return entry

    @property
    def exhausted(self) -> bool:
        return (
            self.reserved_gpu_seconds >= self.gpu_seconds_limit
            or self.reserved_tokens >= self.tokens_limit
        )


@dataclass(frozen=True)
class Action:
    """One billable step. ``run`` receives no arguments and returns a
    JSON-serializable result reference; ``input_hash`` pins the identity
    of what was executed so a replay can disown forged records."""

    name: str
    input_hash: str
    estimated_gpu_seconds: float
    estimated_tokens: int
    run: object  # Callable[[], dict]


class Orchestrator:
    """Recoverable sequential execution of actions with budget gating."""

    def __init__(self, journal_path: Path, budget: Budget):
        self.journal = Journal(Path(journal_path))
        self.budget = budget
        self.state = STATE_RUNNING

    def _trusted_finished(self, actions: list[Action]) -> set[str]:
        """Action names with a finish record whose input hash matches the
        declared action identity; forged records are ignored for
        completion and surfaced as disowned events."""
        by_name = {action.name: action for action in actions}
        done: set[str] = set()
        for entry in self.journal.entries:
            if entry["kind"] != "action_finished":
                continue
            action = by_name.get(entry.get("action_id"))
            if action is None:
                continue
            if entry.get("input_hash") != action.input_hash:
                self.journal.append(
                    "journal_disowned",
                    action_id=entry["action_id"],
                    reason="finish record input_hash does not match the declared action identity",
                )
                continue
            done.add(entry["action_id"])
        return done

    def _interrupted_actions(self, actions: list[Action]) -> list[str]:
        started = {}
        for entry in self.journal.entries:
            if entry["kind"] == "action_started":
                started[entry["action_id"]] = entry
            if entry["kind"] == "action_finished":
                started.pop(entry["action_id"], None)
        for name in started:
            if any(action.name == name for action in actions):
                self.journal.append("action_interrupted", action_id=name)
        return list(started)

    def _settled_from_journal(self) -> tuple[float, int]:
        """Durable settlement totals: the journal, not this process's
        in-memory budget, is the source of truth for billing."""
        gpu = sum(
            entry.get("gpu_seconds", 0.0)
            for entry in self.journal.entries
            if entry["kind"] == "budget_settled"
        )
        tokens = sum(
            entry.get("tokens", 0)
            for entry in self.journal.entries
            if entry["kind"] == "budget_settled"
        )
        return (gpu, tokens)

    def run(
        self,
        actions: list[Action],
        *,
        resume: bool = False,
        cancel: object | None = None,
    ) -> dict:
        """Execute actions in order. With ``resume`` the journal decides
        what still needs to run; without it a non-empty journal is an
        error (refusing to silently rerun or skip billed work)."""
        if not resume and self.journal.entries:
            raise ValueError("journal already exists; call run(resume=True) to recover")
        if resume:
            self._interrupted_actions(actions)
        done = self._trusted_finished(actions)
        cancel_requested = bool(getattr(cancel, "is_set", lambda: False)()) if cancel else False
        for action in actions:
            if action.name in done:
                continue
            if cancel_requested or (cancel is not None and cancel.is_set()):
                self.state = STATE_CANCELLED
                break
            reservation = self.budget.try_reserve(
                self.journal,
                action.name,
                action.estimated_gpu_seconds,
                action.estimated_tokens,
            )
            if reservation is None:
                self.state = STATE_BUDGET_EXHAUSTED
                self.journal.append(
                    "budget_refused",
                    action_id=action.name,
                    reserved_gpu_seconds=self.budget.reserved_gpu_seconds,
                    reserved_tokens=self.budget.reserved_tokens,
                )
                break
            self.journal.append(
                "action_started",
                action_id=action.name,
                input_hash=action.input_hash,
                budget_reservation=reservation["entry_hash"],
            )
            result = action.run() or {}
            settled = self.budget.settle(
                self.journal,
                action.name,
                float(result.get("gpu_seconds", action.estimated_gpu_seconds)),
                int(result.get("tokens", action.estimated_tokens)),
            )
            self.journal.append(
                "action_finished",
                action_id=action.name,
                input_hash=action.input_hash,
                result_ref=result.get("result_ref"),
                budget_settlement=settled["entry_hash"],
            )
        journal_settled = self._settled_from_journal()
        already_done = any(entry["kind"] == "experiment_done" for entry in self.journal.entries)
        all_trusted = all(action.name in self._trusted_finished(actions) for action in actions)
        if self.state == STATE_RUNNING and already_done and all_trusted:
            # Recovery read: the durable journal already proves completion;
            # this process adds nothing and must not downgrade the state.
            self.state = STATE_DONE
        if self.state == STATE_RUNNING and not already_done and all_trusted:
            self.state = STATE_DONE
            self.journal.append(
                "experiment_done",
                actions=[action.name for action in actions],
                settled_gpu_seconds=journal_settled[0],
                settled_tokens=journal_settled[1],
            )
        return {
            "state": self.state,
            "finished": sorted(self._trusted_finished(actions)),
            "settled_gpu_seconds": journal_settled[0],
            "settled_tokens": journal_settled[1],
            "entries": len(self.journal.entries),
        }
