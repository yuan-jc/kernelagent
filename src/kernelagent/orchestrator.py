"""Experiment state machine with budget accounting and crash recovery
(T10, design §7 and §12; RV02 attempt-consistent settlement).

The orchestrator executes an ordered list of actions and records every
transition in an append-only JSONL journal:

- ``budget_reserved`` / ``action_started`` - written BEFORE the attempt
  runs, with the action name, the input-identity hash (on start), the
  budget reservation, and a fresh ``attempt_id`` (RV02) that identifies
  this one execution of the logical action;
- ``action_finished`` - written after the action returns, with a result
  reference. The settlement and the finish are committed as ONE atomic
  journal block (a single ``write`` + ``fsync``), so a process crash can
  never split a settlement from its completion record in a healthy tail;
- ``action_interrupted`` - written by recovery for a started attempt that
  has neither settled nor finished: interrupted work is re-runnable and
  NEVER counts as completed;
- ``attempt_uncertain`` - written by recovery for an attempt whose
  settlement is durable but whose finish record is missing (the RV02
  "settled but not finished" window): the billed cost stands (billed
  exactly once for that attempt), completion stays UNKNOWN - it is never
  reported as done - and the logical action is re-executed under a fresh
  attempt id;
- ``budget_reserved`` / ``budget_settled`` - every attempt pays: reserve
  before start, settle actuals after;
- ``journal_recovered`` - written when the loader drops a truncated,
  never-committed tail record (see the recovery rule below);
- ``stage_started`` - written by the optimization loop when a candidate
  enters a pipeline stage (see kernelagent.domain.run_manifest.PipelineStage),
  so an external reader can rebuild each candidate's stage timeline;
- ``manifest_revised`` - written when a resume explicitly overrides a
  revisable RunManifest field (budgets, candidate allowance);
- ``experiment_done`` - terminal, written only when every action finished.

Attempt identity (RV02): a logical action may be attempted several times
across crashes. Every lifecycle event carries the ``attempt_id`` of one
execution, and settlement uniqueness is enforced per attempt - re-executing
an uncertain attempt under a new id bills the genuinely new work once and
cannot collide with the earlier settlement during replay (the review R2
"duplicate settlement" crash). Settlements of the same attempt id are
rejected as duplicate billing.

Durability policy (RV02): every journal append is a single ``write()`` of
the complete serialized record(s) into the file opened in append mode,
followed by ``flush()`` and ``os.fsync()`` of the file; the directory
entry is fsynced once when the journal file is first created. On local
POSIX filesystems this makes an appended record durable against both
process death and power loss; it cannot say anything about remote or
non-POSIX storage.

Tail recovery rule (RV02): ONLY a trailing fragment without its
terminating newline is an uncommitted partial write, and it may be
dropped - after truncating it away and preserving its size and content
hash in a ``journal_recovered`` event. Any other damage (an unparseable
complete line, an entry-hash mismatch, a broken hash chain, including on
the last complete line) refuses to load: the ledger is never silently
rewritten.

Honesty statement (RV02): these are file-persistence semantics. Billing
is exactly-once per attempt (replay rejects a second settlement of the
same attempt), but execution is at-least-once: an attempt whose
completion cannot be decided keeps its cost and its uncertainty marker,
and the action runs again under a fresh attempt id. Cross-process
exactly-once execution of external side effects (GPU work, model calls)
is NOT promised by file persistence and must come from idempotent ports
if it is required at all.

A hash chain over journal entries (each entry records the sha256 of the
previous one) makes retro-forgery detectable: a fabricated
``action_finished`` breaks the chain or references an input hash that the
recovery replay can disown. Cancel is honoured at action boundaries;
between actions the machine stops with state ``cancelled`` and keeps the
settled costs. When the budget cannot cover a reservation the machine
refuses to start the next action and ends ``budget_exhausted`` - no
partial free work.

Metering honesty (RV04): every settlement carries a ``gpu_metering``
label - ``"actual"`` when the amount is the measured wall-clock GPU-lease
time the action reported, ``"estimated"`` when only the action's
reservation estimate is known (legacy results without a measurement, or
any caller that cannot measure). An attempt interrupted before its
settlement keeps its reservation held as a conservative ESTIMATE and its
``action_interrupted`` event says so explicitly (``estimated=true`` plus
the held amounts); it is never counted as measured usage. The journal is
the spending truth; :func:`reconstruct_budget` totals both kinds and
:func:`settled_metering_split` separates actual from estimated so
reports never have to claim a guess was a measurement.

The state machine is pure control plane: actions are callables supplied
by the caller (the T10 smoke wires the real T05/T07/T09 adapters)."""

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

STATE_RUNNING = "running"
STATE_DONE = "completed"
STATE_CANCELLED = "cancelled"
STATE_BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class DurableBudgetState:
    """Budget totals replayed from the journal: the durable truth that any
    new process must restore before doing work (launch-plan P0-2)."""

    reserved_gpu_seconds: float
    reserved_tokens: int
    settled_gpu_seconds: float
    settled_tokens: int


def _settlement_key(entry: dict) -> tuple[str, str]:
    """Uniqueness key of a settlement: the attempt id when the entry has
    one, else the action id (legacy journals without attempt identity keep
    their one-settlement-per-action rule)."""
    attempt_id = entry.get("attempt_id")
    if attempt_id:
        return ("attempt", str(attempt_id))
    return ("action", str(entry.get("action_id")))


def _reservation_key(entry: dict) -> tuple[str, str]:
    """Stack key of a reservation: reservations and settlements of one
    attempt share the attempt id, so a settlement consumes exactly its own
    attempt's reservation."""
    return _settlement_key(entry)


def _replay_budget(
    entries: tuple[dict, ...],
) -> tuple[
    DurableBudgetState,
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[int]],
]:
    """Replay budget events into (state, open_gpu_stacks, open_token_stacks).

    Open reservations are per-attempt LIFO stacks: a settlement consumes
    its own attempt's reservation, while reservations from interrupted
    attempts stay held - a crashed attempt keeps occupying its estimate.
    Rules: amounts must be non-negative finite numbers; a second
    settlement for the same attempt (or, in legacy journals without
    attempt ids, for the same action) is a hard duplicate-billing error;
    a release without an open reservation is a hard error. A settlement
    for an attempt with no open reservation counts as a legacy billed
    cost (the journal is the spending truth)."""
    reserved_gpu = 0.0
    reserved_tokens = 0
    settled_gpu = 0.0
    settled_tokens = 0
    open_gpu: dict[tuple[str, str], list[float]] = {}
    open_tokens: dict[tuple[str, str], list[int]] = {}
    settled_keys: set[tuple[str, str]] = set()

    def _amount(entry: dict, field: str, *, integral: bool = False) -> float | int:
        value = entry.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"journal entry {entry.get('kind')} has negative/invalid {field}")
        if integral:
            if not isinstance(value, int):
                raise ValueError(f"journal entry {entry.get('kind')} has non-integer {field}")
            return value
        if not math.isfinite(value):
            raise ValueError(f"journal entry {entry.get('kind')} has non-finite {field}")
        return value

    def _consume(key: tuple[str, str]) -> None:
        nonlocal reserved_gpu, reserved_tokens
        gpu_stack = open_gpu.get(key)
        token_stack = open_tokens.get(key)
        if gpu_stack:
            reserved_gpu -= gpu_stack.pop()
        if token_stack:
            reserved_tokens -= token_stack.pop()

    for entry in entries:
        kind = entry.get("kind")
        if kind == "budget_reserved":
            key = _reservation_key(entry)
            gpu = _amount(entry, "gpu_seconds")
            tokens = _amount(entry, "tokens", integral=True)
            open_gpu.setdefault(key, []).append(gpu)
            open_tokens.setdefault(key, []).append(tokens)
            reserved_gpu += gpu
            reserved_tokens += tokens
        elif kind == "budget_settled":
            key = _settlement_key(entry)
            gpu = _amount(entry, "gpu_seconds")
            tokens = _amount(entry, "tokens", integral=True)
            if key in settled_keys:
                raise ValueError(
                    f"duplicate settlement for {key[0]} {key[1]!r}: one attempt is "
                    "billed at most once"
                )
            settled_keys.add(key)
            settled_gpu += gpu
            settled_tokens += tokens
            _consume(key)
        elif kind == "budget_released":
            key = _reservation_key(entry)
            if not open_gpu.get(key) and not open_tokens.get(key):
                raise ValueError(f"release without an open reservation for {key!r}")
            _consume(key)
    return (
        DurableBudgetState(
            reserved_gpu_seconds=reserved_gpu,
            reserved_tokens=reserved_tokens,
            settled_gpu_seconds=settled_gpu,
            settled_tokens=settled_tokens,
        ),
        open_gpu,
        open_tokens,
    )


def reconstruct_budget(entries: tuple[dict, ...]) -> DurableBudgetState:
    """Durable budget totals from journal entries (see :func:`_replay_budget`)."""
    return _replay_budget(entries)[0]


def settled_metering_split(entries: tuple[dict, ...]) -> tuple[float, float]:
    """Split settled GPU seconds into ``(actual, estimated)`` by each
    settlement's ``gpu_metering`` label (RV04).

    Only settlements explicitly labeled ``"actual"`` count as measured;
    unlabeled settlements (legacy journals, or callers that never measured)
    count as estimated - a missing label can never be claimed as a
    measurement. Tokens keep no split: they are provider-reported usage or
    the conservative per-call estimate either way."""
    actual = 0.0
    estimated = 0.0
    for entry in entries:
        if entry.get("kind") != "budget_settled":
            continue
        gpu = entry.get("gpu_seconds", 0.0)
        if isinstance(gpu, bool) or not isinstance(gpu, (int, float)) or gpu < 0:
            continue
        if entry.get("gpu_metering") == "actual":
            actual += float(gpu)
        else:
            estimated += float(gpu)
    return (actual, estimated)


def _hash_entry(entry: dict) -> str:
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _serialize_entry(entry: dict) -> bytes:
    line = json.dumps(entry, sort_keys=True, default=str) + "\n"
    return line.encode("utf-8")


class Journal:
    """Append-only JSONL journal with a per-entry hash chain.

    Durability policy (RV02, see the module docstring): each append is one
    ``write()`` of the fully serialized record(s) + ``flush`` + ``fsync``
    of the file; the directory entry is fsynced when the file is created.
    The settlement+finish commit block (:meth:`commit_settlement_and_finish`)
    is a single such write, so a crash can tear at most the tail of the
    file. Loading applies the tail rule: a trailing fragment without a
    terminating newline is an uncommitted partial record - it is dropped
    (and preserved by hash inside a ``journal_recovered`` event); corrupt
    records anywhere else refuse to load."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: list[dict] = []
        self._chain = "genesis"
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        torn: bytes | None = None
        if lines and lines[-1] != b"":
            # The file does not end with a newline: the trailing fragment
            # is an uncommitted partial record (tail rule).
            torn = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            self._load_line(line)
        if torn is not None:
            self._recover_torn_tail(torn, kept_bytes=len(raw) - len(torn))

    def _load_line(self, line: bytes) -> None:
        position = len(self._entries)
        try:
            entry = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"journal record at entry {position} is corrupt and not a truncated tail: {exc}"
            ) from exc
        stored = entry.get("entry_hash")
        payload = {k: v for k, v in entry.items() if k != "entry_hash"}
        if stored != _hash_entry(payload):
            raise ValueError(f"journal entry hash mismatch at entry {position}")
        if entry.get("prev_hash") != self._chain:
            raise ValueError(f"journal hash chain broken at entry {position}")
        self._chain = stored
        self._entries.append(entry)

    def _recover_torn_tail(self, torn: bytes, *, kept_bytes: int) -> None:
        """Apply the documented tail rule: truncate the uncommitted fragment
        (never silently - its size and content hash are preserved in a
        ``journal_recovered`` event) and keep the chain anchored at the last
        complete record."""
        with self.path.open("r+b") as handle:
            handle.truncate(kept_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        self.append(
            "journal_recovered",
            reason="truncated uncommitted tail record dropped per the documented "
            "recovery rule; content hash below proves what was dropped",
            dropped_bytes=len(torn),
            dropped_sha256=hashlib.sha256(torn).hexdigest(),
            dropped_head=torn.decode("utf-8", errors="replace")[:200],
        )

    @property
    def entries(self) -> tuple[dict, ...]:
        return tuple(self._entries)

    def _write_block(self, payload: bytes) -> None:
        """Durable append of one commit block (see the class docstring for
        the flush/fsync policy). Single injection seam for tests that
        simulate torn writes at a crash boundary."""
        is_new = not self.path.exists()
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if is_new:
            self._fsync_parent_dir()

    def _fsync_parent_dir(self) -> None:
        if os.name != "posix":  # pragma: no cover - Windows dev boxes
            return
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(self, kind: str, **fields) -> dict:
        entry = {"kind": kind, "prev_hash": self._chain, **fields}
        entry["entry_hash"] = _hash_entry(entry)
        self._write_block(_serialize_entry(entry))
        self._chain = entry["entry_hash"]
        self._entries.append(entry)
        return entry

    def commit_settlement_and_finish(
        self,
        *,
        action_id: str,
        attempt_id: str,
        input_hash: str,
        result_ref: object,
        gpu_seconds: float,
        tokens: int,
        gpu_metering: str = "estimated",
    ) -> tuple[dict, dict]:
        """Atomically commit one attempt's settlement and finish record as a
        single durable block (RV02): a crash before this call leaves an
        interrupted attempt, a crash during it can tear only the tail
        (handled by the loader rule), and a crash after it leaves both
        records durable - the review R2 window where a settlement existed
        without a completion record is closed for whole-block writes and
        reducible to the tail rule for torn ones.

        ``gpu_metering`` (RV04) labels the settlement's GPU-seconds amount
        as ``"actual"`` (measured lease time) or ``"estimated"`` (only the
        reservation estimate is known); the conservative default is
        ``"estimated"``."""
        settled: dict = {
            "kind": "budget_settled",
            "prev_hash": self._chain,
            "action_id": action_id,
            "attempt_id": attempt_id,
            "gpu_seconds": gpu_seconds,
            "tokens": tokens,
            "gpu_metering": gpu_metering,
        }
        settled["entry_hash"] = _hash_entry(settled)
        finished: dict = {
            "kind": "action_finished",
            "prev_hash": settled["entry_hash"],
            "action_id": action_id,
            "attempt_id": attempt_id,
            "input_hash": input_hash,
            "result_ref": result_ref,
            "budget_settlement": settled["entry_hash"],
        }
        finished["entry_hash"] = _hash_entry(finished)
        self._write_block(_serialize_entry(settled) + _serialize_entry(finished))
        self._chain = finished["entry_hash"]
        self._entries.extend((settled, finished))
        return settled, finished


@dataclass
class Budget:
    """GPU-seconds / token budget: reserve before starting, settle after.

    The spending gate is ``settled + reserved + requested``: settled costs
    are durable and count forever; a reservation is released only when its
    own attempt settles (the estimate becomes the settled floor). Repeated
    crash-resume cycles re-reserve on top of the held estimates, so a
    crashing action converges to budget_exhausted instead of looping for
    free (launch-plan P0-2)."""

    gpu_seconds_limit: float
    tokens_limit: int
    reserved_gpu_seconds: float = 0.0
    reserved_tokens: int = 0
    settled_gpu_seconds: float = 0.0
    settled_tokens: int = 0
    events: list = field(default_factory=list, repr=False)
    _open_gpu: dict = field(default_factory=dict, repr=False)
    _open_tokens: dict = field(default_factory=dict, repr=False)

    def try_reserve(
        self,
        journal: Journal,
        action_id: str,
        gpu_seconds: float,
        tokens: int,
        *,
        attempt_id: str | None = None,
    ) -> dict | None:
        key = _settlement_key({"attempt_id": attempt_id, "action_id": action_id})
        if (
            self.settled_gpu_seconds + self.reserved_gpu_seconds + gpu_seconds
            > self.gpu_seconds_limit
            or self.settled_tokens + self.reserved_tokens + tokens > self.tokens_limit
        ):
            return None
        self.reserved_gpu_seconds += gpu_seconds
        self.reserved_tokens += tokens
        self._open_gpu.setdefault(key, []).append(gpu_seconds)
        self._open_tokens.setdefault(key, []).append(tokens)
        fields: dict = {
            "action_id": action_id,
            "gpu_seconds": gpu_seconds,
            "tokens": tokens,
        }
        if attempt_id:
            fields["attempt_id"] = attempt_id
        entry = journal.append("budget_reserved", **fields)
        self.events.append(entry)
        return entry

    def apply_settlement(
        self, action_id: str, attempt_id: str | None, gpu_seconds: float, tokens: int
    ) -> None:
        """Adopt a settlement that is already durably committed to the
        journal. Per-attempt in-memory accounting mirrors the replay rules:
        each attempt settles at most once, and the settlement consumes only
        its own attempt's reservation (a crashed attempt's stays held)."""
        self.settled_gpu_seconds += gpu_seconds
        self.settled_tokens += tokens
        key = _settlement_key({"attempt_id": attempt_id, "action_id": action_id})
        if self._open_gpu.get(key):
            self.reserved_gpu_seconds -= self._open_gpu[key].pop()
        if self._open_tokens.get(key):
            self.reserved_tokens -= self._open_tokens[key].pop()

    def restore(self, state: DurableBudgetState, open_gpu: dict, open_tokens: dict) -> None:
        """Adopt durable totals replayed from the journal before any work
        runs in this process."""
        self.reserved_gpu_seconds = state.reserved_gpu_seconds
        self.reserved_tokens = state.reserved_tokens
        self.settled_gpu_seconds = state.settled_gpu_seconds
        self.settled_tokens = state.settled_tokens
        self._open_gpu = dict(open_gpu)
        self._open_tokens = dict(open_tokens)

    @property
    def exhausted(self) -> bool:
        return (
            self.settled_gpu_seconds + self.reserved_gpu_seconds >= self.gpu_seconds_limit
            or self.settled_tokens + self.reserved_tokens >= self.tokens_limit
        )


@dataclass(frozen=True)
class Action:
    """One billable step. ``run`` receives no arguments and returns a
    JSON-serializable result reference; ``input_hash`` pins the identity
    of what was executed so a replay can disown forged records. One
    execution of an action is one *attempt* and gets a fresh ``attempt_id``
    in the journal (RV02); the action name stays the logical identity."""

    name: str
    input_hash: str
    estimated_gpu_seconds: float
    estimated_tokens: int
    run: object  # Callable[[], dict]


class Orchestrator:
    """Recoverable sequential execution of actions with budget gating.

    Semantics (RV02): execution is at-least-once with exactly-once billing
    per attempt - see the honesty statement in the module docstring. A
    settled-but-unfinished attempt keeps its cost and an
    ``attempt_uncertain`` marker and is never reported as completed; the
    action is re-executed under a fresh attempt id, whose settlement is a
    real new cost billed once."""

    def __init__(self, journal_path: Path, budget: Budget):
        self.journal = Journal(Path(journal_path))
        self.budget = budget
        self.state = STATE_RUNNING
        self._disowned: set[str] = set()

    @staticmethod
    def _attempt_key(entry: dict) -> tuple[str, str]:
        return _settlement_key(entry)

    def _trusted_finished(self, actions: list[Action]) -> set[str]:
        """Action names with a finish record whose input hash matches the
        declared action identity; forged records are ignored for
        completion and surfaced as disowned events (once per process)."""
        by_name = {action.name: action for action in actions}
        done: set[str] = set()
        for entry in self.journal.entries:
            if entry["kind"] != "action_finished":
                continue
            action = by_name.get(entry.get("action_id"))
            if action is None:
                continue
            if entry.get("input_hash") != action.input_hash:
                if entry.get("entry_hash") not in self._disowned:
                    self._disowned.add(entry.get("entry_hash"))
                    self.journal.append(
                        "journal_disowned",
                        action_id=entry["action_id"],
                        attempt_id=entry.get("attempt_id"),
                        reason="finish record input_hash does not match the declared "
                        "action identity",
                    )
                continue
            done.add(entry["action_id"])
        return done

    def _reconcile_attempts(
        self,
        actions: list[Action],
        open_gpu: dict[tuple[str, str], list[float]] | None = None,
        open_tokens: dict[tuple[str, str], list[int]] | None = None,
    ) -> None:
        """Recovery pass (RV02): pair every started attempt with its finish.
        A started attempt with no finish and no settlement becomes
        ``action_interrupted`` (re-runnable, never completed). A started
        attempt whose settlement IS durable but whose finish is missing
        becomes ``attempt_uncertain``: its cost stays billed exactly once,
        its completion stays unknown, and the run loop re-executes the
        action under a fresh attempt id - the settlement is never repeated
        for the same attempt. Reconciliation is idempotent: an attempt
        already carrying its marker is never marked a second time, so any
        number of consecutive recoveries produces exactly one marker per
        unresolved attempt.

        RV04 metering honesty: an interrupted attempt's GPU cost is
        UNKNOWN - only its reservation estimate is held - so the
        ``action_interrupted`` event is labeled ``estimated=true`` and
        records the still-held conservative estimate amounts."""
        open_gpu = open_gpu or {}
        open_tokens = open_tokens or {}
        known = {action.name for action in actions}
        open_starts: dict[tuple[str, str], dict] = {}
        settled: dict[tuple[str, str], dict] = {}
        reconciled: set[tuple[str, str]] = set()
        for entry in self.journal.entries:
            kind = entry.get("kind")
            if entry.get("action_id") not in known:
                continue
            key = self._attempt_key(entry)
            if kind == "action_started":
                open_starts[key] = entry
            elif kind == "action_finished":
                open_starts.pop(key, None)
            elif kind == "budget_settled":
                settled[key] = entry
            elif kind in ("action_interrupted", "attempt_uncertain"):
                reconciled.add(key)
        for key, start in open_starts.items():
            if key in reconciled:
                continue
            settlement = settled.get(key)
            if settlement is not None:
                self.journal.append(
                    "attempt_uncertain",
                    action_id=start["action_id"],
                    attempt_id=start.get("attempt_id"),
                    settled_gpu_seconds=settlement.get("gpu_seconds"),
                    settled_tokens=settlement.get("tokens"),
                    reason="settlement is durable without a finish record; completion "
                    "cannot be decided, so the cost stands and the action is "
                    "re-executed under a fresh attempt id",
                )
            else:
                held_gpu = sum(open_gpu.get(key, ()))
                held_tokens = sum(open_tokens.get(key, ()))
                self.journal.append(
                    "action_interrupted",
                    action_id=start["action_id"],
                    attempt_id=start.get("attempt_id"),
                    estimated=True,
                    held_gpu_estimate=held_gpu,
                    held_tokens_estimate=held_tokens,
                    reason="started attempt has no settlement and no finish: its GPU "
                    "cost is unknown and only the reservation estimate stays "
                    "conservatively held (billed if the lease really ran, never "
                    "reported as measured usage)",
                )

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
        error (refusing to silently rerun or skip billed work).

        Every execution gets a fresh ``attempt_id`` (RV02). The settlement
        and the finish of one attempt are committed as a single atomic
        journal block, so the review R2 crash window (settled without
        finish) is closed for whole-block writes and reduces to the
        documented tail rule for torn ones. See the honesty statement in
        the module docstring: execution is at-least-once, billing exactly
        once per attempt."""
        if not resume and self.journal.entries:
            raise ValueError("journal already exists; call run(resume=True) to recover")
        if resume:
            # Durable budget first: no action in this process may start
            # before settled+reserved totals are restored from the journal
            # (launch-plan P0-2).
            state, open_gpu, open_tokens = _replay_budget(self.journal.entries)
            self.budget.restore(state, open_gpu, open_tokens)
            self._reconcile_attempts(actions, open_gpu, open_tokens)
        done = self._trusted_finished(actions)
        cancel_requested = bool(getattr(cancel, "is_set", lambda: False)()) if cancel else False
        for action in actions:
            if action.name in done:
                continue
            if cancel_requested or (cancel is not None and cancel.is_set()):
                self.state = STATE_CANCELLED
                break
            attempt_id = uuid.uuid4().hex
            reservation = self.budget.try_reserve(
                self.journal,
                action.name,
                action.estimated_gpu_seconds,
                action.estimated_tokens,
                attempt_id=attempt_id,
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
                attempt_id=attempt_id,
                input_hash=action.input_hash,
                budget_reservation=reservation["entry_hash"],
            )
            result = action.run() or {}
            # RV04 metering honesty: a result that carries a measured
            # ``gpu_seconds`` settles as "actual"; falling back to the
            # action's reservation estimate settles as "estimated" - a
            # guess is never booked as a measurement.
            raw_gpu = result.get("gpu_seconds")
            if raw_gpu is None:
                settled_gpu = float(action.estimated_gpu_seconds)
                gpu_metering = "estimated"
            else:
                settled_gpu = float(raw_gpu)
                gpu_metering = str(result.get("gpu_metering", "actual"))
            settled_tokens = int(result.get("tokens", action.estimated_tokens))
            # One atomic durable block: settlement + finish of THIS attempt
            # (see Journal.commit_settlement_and_finish).
            self.journal.commit_settlement_and_finish(
                action_id=action.name,
                attempt_id=attempt_id,
                input_hash=action.input_hash,
                result_ref=result.get("result_ref"),
                gpu_seconds=settled_gpu,
                tokens=settled_tokens,
                gpu_metering=gpu_metering,
            )
            self.budget.apply_settlement(action.name, attempt_id, settled_gpu, settled_tokens)
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
