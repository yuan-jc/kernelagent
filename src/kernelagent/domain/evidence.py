"""Evidence pointers. An EvidenceRef names a persisted artifact; it is never
the artifact itself and never proves anything without the stored bytes."""

from __future__ import annotations

from dataclasses import dataclass

from kernelagent.domain._validation import require_sha256, require_text


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    artifact_sha256: str
    kind: str
    producer_version: str

    def __post_init__(self) -> None:
        require_sha256(self.artifact_sha256, "artifact_sha256")
        require_text(self.kind, "kind")
        require_text(self.producer_version, "producer_version")
