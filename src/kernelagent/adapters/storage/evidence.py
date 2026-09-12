"""Evidence storage and experiment identity (design §10-11).

Three identity hashes per §11.1: ``implementation_id`` lives in the domain
(content-derived); this module provides ``evaluation_key`` (adds workload,
protocol, environment and tool versions) and ``experiment_id`` (one
independent execution; equal evaluation keys may carry many experiments).

The store is append-only: artifacts are content-addressed files published by
temporary-file rename, and the SQLite index has no UPDATE or DELETE path.
Repeated experiment submission with identical content is idempotent; with
different content it is a hard conflict, so a recorded terminal state can
never be contradicted. Cache lookups key on the full evaluation key, so an
environment or protocol change can never return a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from kernelagent.domain import EVALUATION_STATUSES, EvidenceRef
from kernelagent.domain._validation import require_text

SCHEMA_VERSION = "1.0"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_STAGING = ".staging"


class EvidenceStoreError(ValueError):
    """Base class for evidence storage contract violations."""


class ArtifactIntegrityError(EvidenceStoreError):
    """Stored bytes do not match their content address."""


class UnknownArtifactError(EvidenceStoreError):
    """An artifact hash is not present in the store."""


class ExperimentConflictError(EvidenceStoreError):
    """An experiment id was resubmitted with different content."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _require_hex64(value: object, field: str) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise EvidenceStoreError(f"{field} must be a lowercase 64-hex sha256 string; got {value!r}")


def new_experiment_id() -> str:
    """A fresh identity for one independent execution."""
    return uuid.uuid4().hex


def evaluation_key(
    implementation_id: str,
    workload_content_sha256: str,
    protocol_sha256: str,
    environment_sha256: str,
    tool_versions: Mapping[str, str] | None = None,
) -> str:
    """Bind implementation, workload, protocol, environment and tool versions
    into one lookup key (design §11.1). Any change yields a different key."""
    _require_hex64(implementation_id, "implementation_id")
    _require_hex64(workload_content_sha256, "workload_content_sha256")
    _require_hex64(protocol_sha256, "protocol_sha256")
    _require_hex64(environment_sha256, "environment_sha256")
    tools: list[list[str]] = []
    if tool_versions is not None:
        for name, version in tool_versions.items():
            require_text(name, "tool name")
            require_text(version, "tool version")
            tools.append([name, version])
    tools.sort()
    payload = {
        "implementation_id": implementation_id,
        "workload": workload_content_sha256,
        "protocol": protocol_sha256,
        "environment": environment_sha256,
        "tools": tools,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    evaluation_key: str
    implementation_id: str
    status: str
    evidence: tuple[EvidenceRef, ...]
    created_at: str
    record_sha256: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    indexed: int
    corrupted: tuple[str, ...]
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not (self.corrupted or self.missing or self.orphaned)

    def as_dict(self) -> dict:
        return {
            "indexed": self.indexed,
            "corrupted": list(self.corrupted),
            "missing": list(self.missing),
            "orphaned": list(self.orphaned),
            "healthy": self.healthy,
        }


class EvidenceStore:
    """Append-only evidence store: content-addressed artifacts plus a SQLite
    index of artifacts, experiments and events."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.objects_dir = self.artifacts_dir / "objects"
        self.staging_dir = self.artifacts_dir / _STAGING
        self.db_path = self.root / "index.db"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              sha256 TEXT PRIMARY KEY,
              size INTEGER NOT NULL,
              kind TEXT NOT NULL,
              producer_version TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
              experiment_id TEXT PRIMARY KEY,
              evaluation_key TEXT NOT NULL,
              implementation_id TEXT NOT NULL,
              status TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              record_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_key
              ON experiments(evaluation_key);
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- artifacts ---------------------------------------------------------

    def put_artifact(self, data: bytes, kind: str, producer_version: str) -> EvidenceRef:
        """Publish bytes under their content address; publishing is atomic and
        idempotent, and an existing entry with drifting bytes is corruption."""
        require_text(kind, "kind")
        require_text(producer_version, "producer_version")
        digest = hashlib.sha256(data).hexdigest()
        target = self.objects_dir / digest[:2] / digest
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ArtifactIntegrityError(
                    f"artifact {digest} exists with mismatched bytes; run audit()"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = self.staging_dir / f"{uuid.uuid4().hex}.tmp"
            staging.write_bytes(data)
            os.replace(staging, target)
        with self._db as db:
            db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (digest, len(data), kind, producer_version, _now()),
            )
        return EvidenceRef(artifact_sha256=digest, kind=kind, producer_version=producer_version)

    def has_artifact(self, sha256: str) -> bool:
        return (self.objects_dir / sha256[:2] / sha256).is_file()

    def get_artifact(self, sha256: str) -> bytes:
        target = self.objects_dir / sha256[:2] / sha256
        if not target.is_file():
            raise UnknownArtifactError(f"artifact {sha256} is not in the store")
        data = target.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            raise ArtifactIntegrityError(
                f"artifact {sha256} content hashes to {digest}; the stored bytes are corrupted"
            )
        return data

    def artifact_meta(self, sha256: str) -> dict:
        row = self._db.execute(
            "SELECT size, kind, producer_version, created_at FROM artifacts WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            raise UnknownArtifactError(f"artifact {sha256} is not indexed")
        return {
            "sha256": sha256,
            "size": row[0],
            "kind": row[1],
            "producer_version": row[2],
            "created_at": row[3],
        }

    # -- experiments -------------------------------------------------------

    def record_experiment(
        self,
        experiment_id: str,
        evaluation_key: str,
        implementation_id: str,
        status: str,
        evidence: Sequence[EvidenceRef],
    ) -> ExperimentRecord:
        """Record one experiment result. Evidence references must already be
        stored; an identical resubmission is idempotent, a differing one is a
        conflict that leaves the original record untouched."""
        require_text(experiment_id, "experiment_id")
        _require_hex64(evaluation_key, "evaluation_key")
        _require_hex64(implementation_id, "implementation_id")
        if status not in EVALUATION_STATUSES:
            raise EvidenceStoreError(
                f"status must be one of {sorted(EVALUATION_STATUSES)}; got {status!r}"
            )
        missing = [
            ref.artifact_sha256 for ref in evidence if not self.has_artifact(ref.artifact_sha256)
        ]
        if missing:
            raise UnknownArtifactError(
                f"evidence references unstored artifacts: {sorted(set(missing))}"
            )
        refs = tuple(
            EvidenceRef(
                artifact_sha256=ref.artifact_sha256,
                kind=ref.kind,
                producer_version=ref.producer_version,
            )
            for ref in evidence
        )
        created_at = _now()
        identity = {
            "experiment_id": experiment_id,
            "evaluation_key": evaluation_key,
            "implementation_id": implementation_id,
            "status": status,
            "evidence": [
                {
                    "artifact_sha256": ref.artifact_sha256,
                    "kind": ref.kind,
                    "producer_version": ref.producer_version,
                }
                for ref in refs
            ],
        }
        # Identity excludes created_at: a resubmission is judged on semantic
        # content, not on wall-clock time of arrival.
        record_sha256 = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        existing = self.get_experiment(experiment_id)
        if existing is not None:
            if (
                existing.record_sha256 == record_sha256
                and existing.evaluation_key == evaluation_key
            ):
                self.append_event(
                    "experiment_deduplicated",
                    {"experiment_id": experiment_id, "record_sha256": record_sha256},
                )
                return existing
            raise ExperimentConflictError(
                f"experiment {experiment_id} already recorded with different content "
                f"(existing record {existing.record_sha256[:12]}…, submitted "
                f"{record_sha256[:12]}…); recorded terminal states are immutable"
            )
        with self._db as db:
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    evaluation_key,
                    implementation_id,
                    status,
                    json.dumps(identity["evidence"], sort_keys=True),
                    created_at,
                    record_sha256,
                ),
            )
        self.append_event(
            "experiment_recorded",
            {"experiment_id": experiment_id, "evaluation_key": evaluation_key, "status": status},
        )
        return ExperimentRecord(
            experiment_id=experiment_id,
            evaluation_key=evaluation_key,
            implementation_id=implementation_id,
            status=status,
            evidence=refs,
            created_at=created_at,
            record_sha256=record_sha256,
        )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        row = self._db.execute(
            "SELECT experiment_id, evaluation_key, implementation_id, status, evidence_json,"
            " created_at, record_sha256 FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        evidence = tuple(
            EvidenceRef(
                artifact_sha256=item["artifact_sha256"],
                kind=item["kind"],
                producer_version=item["producer_version"],
            )
            for item in json.loads(row[4])
        )
        return ExperimentRecord(
            experiment_id=row[0],
            evaluation_key=row[1],
            implementation_id=row[2],
            status=row[3],
            evidence=evidence,
            created_at=row[5],
            record_sha256=row[6],
        )

    def list_experiments(self, evaluation_key: str) -> tuple[ExperimentRecord, ...]:
        rows = self._db.execute(
            "SELECT experiment_id FROM experiments WHERE evaluation_key = ? ORDER BY rowid",
            (evaluation_key,),
        ).fetchall()
        records = [self.get_experiment(row[0]) for row in rows]
        return tuple(record for record in records if record is not None)

    def latest_experiment(self, evaluation_key: str) -> ExperimentRecord | None:
        row = self._db.execute(
            "SELECT experiment_id FROM experiments WHERE evaluation_key = ?"
            " ORDER BY rowid DESC LIMIT 1",
            (evaluation_key,),
        ).fetchone()
        return None if row is None else self.get_experiment(row[0])

    # -- events ------------------------------------------------------------

    def append_event(self, kind: str, payload: Mapping[str, object]) -> int:
        require_text(kind, "kind")
        cursor = self._db.execute(
            "INSERT INTO events (ts, kind, payload_json) VALUES (?, ?, ?)",
            (_now(), kind, json.dumps(dict(payload), sort_keys=True)),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def events(self, since: int = 0) -> tuple[dict, ...]:
        rows = self._db.execute(
            "SELECT seq, ts, kind, payload_json FROM events WHERE seq > ? ORDER BY seq",
            (since,),
        ).fetchall()
        return tuple(
            {"seq": row[0], "ts": row[1], "kind": row[2], "payload": json.loads(row[3])}
            for row in rows
        )

    # -- integrity ---------------------------------------------------------

    def audit(self) -> AuditReport:
        """Detect corrupted, missing and orphaned artifacts. Never repairs."""
        indexed = {row[0] for row in self._db.execute("SELECT sha256 FROM artifacts").fetchall()}
        corrupted: list[str] = []
        missing: list[str] = []
        for sha256 in sorted(indexed):
            target = self.objects_dir / sha256[:2] / sha256
            if not target.is_file():
                missing.append(sha256)
                continue
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
                corrupted.append(sha256)
        on_disk: set[str] = set()
        for shard in self.objects_dir.iterdir():
            if not shard.is_dir():
                continue
            for target in shard.iterdir():
                on_disk.add(target.name)
        orphaned = sorted(on_disk - indexed)
        return AuditReport(
            indexed=len(indexed),
            corrupted=tuple(corrupted),
            missing=tuple(missing),
            orphaned=tuple(orphaned),
        )
