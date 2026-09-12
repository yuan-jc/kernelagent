"""Static reader for a verified KernelBench problem snapshot.

The snapshot is the pinned upstream commit recorded in
configs/kernelbench/snapshot-files.manifest.json. This adapter only reads
bytes and metadata: it never executes problem code, never imports torch or
Triton, and never scores anything. Loading and running problems on a GPU
worker belongs to T04/T05.

The upstream-compatible protocol and the extended robustness rules are two
separate sections of the development manifest, and this module offers no
merged view of them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_MEMBER = re.compile(r"^KernelBench/level(\d+)/(\d+)_.+\.py$")


class TaskNotFoundError(LookupError):
    """A requested problem id does not exist in the snapshot level."""

    def __init__(self, level: int, problem_id: int, available: list[int]):
        self.level = level
        self.problem_id = problem_id
        self.available = available
        super().__init__(
            f"problem_id {problem_id} not found in level {level}; available ids: {available}"
        )


class SnapshotIntegrityError(ValueError):
    """Snapshot bytes do not match the committed manifest."""

    def __init__(self, report: dict):
        self.report = report
        problems = len(report["missing"]) + len(report["corrupted"])
        super().__init__(f"snapshot integrity check failed: {problems} problem file(s) drifted")


class DevManifestError(ValueError):
    """A manifest is malformed, internally inconsistent, or drifted from its
    recorded binding."""


class ProtocolMergeError(ValueError):
    """Refused to combine the upstream-compatible and extended protocols."""


def task_id(level: int, problem_id: int) -> str:
    return f"kernelbench-l{level}-p{problem_id:03d}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevManifestError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_content_sha256(path: Path) -> str:
    """Hash the canonical JSON content of a manifest, independent of line
    endings, key order and checkout platform (raw bytes differ between a
    Windows working tree and an LF checkout; canonical content does not)."""
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(_canonical_bytes(parsed)).hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str
    git_blob_sha1: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class FilesManifest:
    repository: str
    commit: str
    entries: tuple[ManifestEntry, ...]

    def entry(self, path: str) -> ManifestEntry | None:
        for item in self.entries:
            if item.path == path:
                return item
        return None

    def problem_members(self) -> list[tuple[int, int, str]]:
        """(level, problem_id, snapshot-relative path) for every problem file."""
        found = []
        for item in self.entries:
            match = _SNAPSHOT_MEMBER.match(item.path)
            if match:
                found.append((int(match.group(1)), int(match.group(2)), item.path))
        return sorted(found)


def load_files_manifest(path: Path) -> FilesManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "files manifest must be a JSON object")
    _require(
        raw.get("schema_version") == SCHEMA_VERSION,
        f"unsupported files manifest schema_version {raw.get('schema_version')!r}",
    )
    repository = raw.get("repository")
    _require(
        isinstance(repository, str) and bool(repository), "repository must be a non-empty string"
    )
    commit = raw.get("commit")
    _require(isinstance(commit, str) and bool(_COMMIT.fullmatch(commit)), "commit must be 40-hex")
    files = raw.get("files")
    _require(isinstance(files, list) and bool(files), "files must be a non-empty list")
    seen: set[str] = set()
    entries: list[ManifestEntry] = []
    for item in files:
        _require(isinstance(item, dict), "file entries must be objects")
        rel = item.get("path")
        _require(isinstance(rel, str), "path must be a string")
        assert isinstance(rel, str)
        parts = rel.split("/")
        _require(
            rel.endswith(".py")
            and "/" in rel
            and "\\" not in rel
            and not rel.startswith("/")
            and not re.match(r"^[A-Za-z]:", rel)
            and all(part not in ("", ".", "..") for part in parts),
            f"invalid snapshot-relative path {rel!r}",
        )
        _require(rel not in seen, f"duplicate manifest path {rel!r}")
        seen.add(rel)
        blob = item.get("git_blob_sha1")
        _require(
            isinstance(blob, str) and bool(_SHA1.fullmatch(blob)), f"bad git_blob_sha1 for {rel!r}"
        )
        sha = item.get("sha256")
        _require(isinstance(sha, str) and bool(_SHA256.fullmatch(sha)), f"bad sha256 for {rel!r}")
        size = item.get("size")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"bad size for {rel!r}",
        )
        entries.append(ManifestEntry(rel, blob, sha, size))
    return FilesManifest(repository, commit, tuple(entries))


def verify_snapshot(root: Path, manifest: FilesManifest) -> dict:
    """Hash every manifest file under ``root``; never silently pass drift."""
    base = Path(root)
    matched = 0
    missing: list[str] = []
    corrupted: list[dict[str, str]] = []
    for entry in manifest.entries:
        target = base / entry.path
        if not target.is_file():
            missing.append(entry.path)
            continue
        digest = _sha256_file(target)
        if digest != entry.sha256:
            corrupted.append(
                {
                    "path": entry.path,
                    "expected_sha256": entry.sha256,
                    "actual_sha256": digest,
                }
            )
        else:
            matched += 1
    return {
        "total": len(manifest.entries),
        "matched": matched,
        "missing": missing,
        "corrupted": corrupted,
    }


@dataclass(frozen=True, slots=True)
class ProblemRef:
    level: int
    problem_id: int
    name: str
    sha256: str
    task_id: str


@dataclass(frozen=True, slots=True)
class Problem:
    ref: ProblemRef
    code: str


class KernelBenchSnapshot:
    """Read-only view of one verified snapshot directory.

    The committed manifest is the authority: files on disk that are not in the
    manifest are invisible; manifest files that are missing or drifted make
    construction fail loudly.
    """

    def __init__(self, root: Path, files: FilesManifest, *, verify: bool = True):
        self._root = Path(root)
        self._files = files
        if verify:
            report = verify_snapshot(self._root, files)
            if report["missing"] or report["corrupted"]:
                raise SnapshotIntegrityError(report)
        self._by_key: dict[tuple[int, int], ProblemRef] = {}
        for level, problem_id, rel in files.problem_members():
            entry = files.entry(rel)
            assert entry is not None  # problem_members derives from entries
            name = rel.rsplit("/", 1)[1]
            self._by_key[(level, problem_id)] = ProblemRef(
                level, problem_id, name, entry.sha256, task_id(level, problem_id)
            )

    @property
    def root(self) -> Path:
        return self._root

    def levels(self) -> tuple[int, ...]:
        return tuple(sorted({level for level, _ in self._by_key}))

    def problem_ids(self, level: int) -> tuple[int, ...]:
        return tuple(problem_id for (lvl, problem_id) in sorted(self._by_key) if lvl == level)

    def list_problems(self, level: int) -> tuple[ProblemRef, ...]:
        return tuple(ref for (lvl, _), ref in sorted(self._by_key.items()) if lvl == level)

    def get_problem(self, level: int, problem_id: int) -> Problem:
        """Return the problem with its source code, re-hashed at read time."""
        ref = self._by_key.get((level, problem_id))
        if ref is None:
            raise TaskNotFoundError(level, problem_id, self.problem_ids(level))
        entry = self._files.entry(f"KernelBench/level{level}/{ref.name}")
        assert entry is not None
        target = self._root / entry.path
        code = target.read_text(encoding="utf-8")
        digest = _sha256_file(target)
        if digest != ref.sha256:
            raise SnapshotIntegrityError(
                {
                    "total": 1,
                    "matched": 0,
                    "missing": [],
                    "corrupted": [
                        {
                            "path": entry.path,
                            "expected_sha256": ref.sha256,
                            "actual_sha256": digest,
                        }
                    ],
                }
            )
        return Problem(ref, code)


@dataclass(frozen=True, slots=True)
class ProtocolReference:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProtocolSet:
    upstream_description: str
    upstream_references: tuple[ProtocolReference, ...]
    extended: tuple[str, ...]

    def merged(self) -> None:
        """Deliberately unsupported: upstream and extended protocols stay
        separate so an upstream-comparable run can never silently inherit
        extended rules."""
        raise ProtocolMergeError(
            "upstream_compatible and extended protocols must stay separate; "
            "task contracts declare one track explicitly (T05+)"
        )


@dataclass(frozen=True, slots=True)
class DevProblem:
    level: int
    problem_id: int
    name: str
    sha256: str
    task_id: str


@dataclass(frozen=True, slots=True)
class DevManifest:
    upstream_repository: str
    upstream_commit: str
    dataset_module: str
    dataset_module_sha256: str
    files_manifest_path: str
    files_manifest_sha256: str
    problems: tuple[DevProblem, ...]
    protocols: ProtocolSet

    def ids_for_level(self, level: int) -> tuple[int, ...]:
        return tuple(p.problem_id for p in self.problems if p.level == level)


def load_dev_manifest(path: Path) -> DevManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "dev manifest must be a JSON object")
    _require(
        raw.get("schema_version") == SCHEMA_VERSION,
        f"unsupported dev manifest schema_version {raw.get('schema_version')!r}",
    )
    upstream = raw.get("upstream")
    _require(isinstance(upstream, dict), "upstream section is required")
    assert isinstance(upstream, dict)
    commit = upstream.get("commit")
    _require(
        isinstance(commit, str) and bool(_COMMIT.fullmatch(commit)),
        "upstream.commit must be 40-hex",
    )
    dataset_sha = upstream.get("dataset_module_sha256")
    _require(
        isinstance(dataset_sha, str) and bool(_SHA256.fullmatch(dataset_sha)),
        "upstream.dataset_module_sha256 must be 64-hex",
    )
    binding = upstream.get("files_manifest_sha256")
    _require(
        isinstance(binding, str) and bool(_SHA256.fullmatch(binding)),
        "upstream.files_manifest_sha256 must be 64-hex",
    )
    problems_raw = raw.get("problems")
    _require(
        isinstance(problems_raw, list) and bool(problems_raw), "problems must be a non-empty list"
    )
    problems: list[DevProblem] = []
    for item in problems_raw:
        _require(isinstance(item, dict), "problem entries must be objects")
        level, problem_id = item.get("level"), item.get("problem_id")
        _require(
            isinstance(level, int) and not isinstance(level, bool) and level >= 1,
            "problem level must be a positive integer",
        )
        _require(
            isinstance(problem_id, int) and not isinstance(problem_id, bool) and problem_id >= 1,
            "problem problem_id must be a positive integer",
        )
        name, sha = item.get("name"), item.get("sha256")
        _require(isinstance(name, str) and bool(name), "problem name must be a non-empty string")
        _require(isinstance(sha, str) and bool(_SHA256.fullmatch(sha)), f"bad sha256 for {name!r}")
        problems.append(DevProblem(level, problem_id, name, sha, task_id(level, problem_id)))
    protocols_raw = raw.get("protocols")
    _require(isinstance(protocols_raw, dict), "protocols section is required")
    assert isinstance(protocols_raw, dict)
    _require(
        set(protocols_raw) == {"upstream_compatible", "extended"},
        "protocols must contain exactly the separate keys 'upstream_compatible' and 'extended'",
    )
    upstream_protocol = protocols_raw["upstream_compatible"]
    _require(isinstance(upstream_protocol, dict), "upstream_compatible must be an object")
    assert isinstance(upstream_protocol, dict)
    description = upstream_protocol.get("description")
    _require(
        isinstance(description, str) and bool(description),
        "upstream_compatible.description must be a non-empty string",
    )
    references_raw = upstream_protocol.get("references")
    _require(isinstance(references_raw, list), "upstream_compatible.references must be a list")
    references: list[ProtocolReference] = []
    for item in references_raw or []:
        _require(isinstance(item, dict), "protocol references must be objects")
        ref_path, ref_sha = item.get("path"), item.get("sha256")
        _require(
            isinstance(ref_path, str) and bool(ref_path),
            "reference path must be a non-empty string",
        )
        _require(
            isinstance(ref_sha, str) and bool(_SHA256.fullmatch(ref_sha)),
            f"bad sha256 for {ref_path!r}",
        )
        references.append(ProtocolReference(ref_path, ref_sha))
    extended_raw = protocols_raw["extended"]
    _require(
        isinstance(extended_raw, list) and all(isinstance(e, str) for e in extended_raw),
        "extended must be a list of rule identifiers",
    )
    protocols = ProtocolSet(
        upstream_description=description,
        upstream_references=tuple(references),
        extended=tuple(extended_raw),
    )
    return DevManifest(
        upstream_repository=upstream.get("repository", ""),
        upstream_commit=commit,
        dataset_module=upstream.get("dataset_module", ""),
        dataset_module_sha256=dataset_sha,
        files_manifest_path=upstream.get("files_manifest", ""),
        files_manifest_sha256=binding,
        problems=tuple(problems),
        protocols=protocols,
    )


def check_dev_manifest(
    dev: DevManifest,
    files: FilesManifest,
    files_manifest_path: Path,
    root: Path | None = None,
) -> dict:
    """Cross-check the dev manifest against the files manifest it binds, and
    (with ``root``) against the actual upstream source bytes."""
    if dev.upstream_commit != files.commit:
        raise DevManifestError(
            f"dev manifest pins commit {dev.upstream_commit!r} but files manifest pins "
            f"{files.commit!r}; refusing to mix snapshot versions"
        )
    actual_binding = canonical_content_sha256(Path(files_manifest_path))
    if dev.files_manifest_sha256 != actual_binding:
        raise DevManifestError(
            "files manifest bytes drifted from the hash recorded in the dev manifest; "
            "regenerate or repin before use"
        )
    seen: set[tuple[int, int]] = set()
    for problem in dev.problems:
        key = (problem.level, problem.problem_id)
        if key in seen:
            raise DevManifestError(f"duplicate dev manifest entry {key}")
        seen.add(key)
        rel = f"KernelBench/level{problem.level}/{problem.name}"
        entry = files.entry(rel)
        if entry is None:
            raise DevManifestError(
                f"dev manifest problem {rel!r} is absent from the files manifest"
            )
        if entry.sha256 != problem.sha256:
            raise DevManifestError(f"dev manifest sha256 mismatch for {rel!r}")
    report = {"problems": len(dev.problems), "protocol_references": "not_checked"}
    if root is not None:
        references = list(dev.protocols.upstream_references)
        if dev.dataset_module:
            references.append(ProtocolReference(dev.dataset_module, dev.dataset_module_sha256))
        base = Path(root)
        for reference in references:
            target = base / reference.path
            if not target.is_file():
                raise DevManifestError(f"protocol reference missing on disk: {reference.path}")
            if _sha256_file(target) != reference.sha256:
                raise DevManifestError(f"protocol reference hash mismatch: {reference.path}")
        report["protocol_references"] = "verified"
    return report


def run_verify(root: Path, manifest_path: Path, dev_manifest_path: Path) -> tuple[dict, int]:
    """CLI entry: verify snapshot bytes, dev manifest bindings and dev subset
    presence. Returns (summary, exit_code); manifest errors become a FAIL
    summary instead of an exception so the CLI always prints one report."""
    try:
        files = load_files_manifest(manifest_path)
        dev = load_dev_manifest(dev_manifest_path)
        dev_check = check_dev_manifest(dev, files, manifest_path, root=root)
    except DevManifestError as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "root": str(root),
            "status": "FAIL",
            "error": f"dev manifest check failed: {exc}",
        }
        return summary, 1
    file_report = verify_snapshot(root, files)
    # Listing never re-hashes; integrity was already reported above.
    snapshot = KernelBenchSnapshot(root, files, verify=False)
    subset: dict[str, dict[str, object]] = {}
    for level in sorted({p.level for p in dev.problems}):
        expected = list(dev.ids_for_level(level))
        available = list(snapshot.problem_ids(level))
        subset[str(level)] = {
            "expected": len(expected),
            "present": len([pid for pid in expected if pid in available]),
            "missing_ids": sorted(set(expected) - set(available)),
        }
    healthy = (
        not file_report["missing"]
        and not file_report["corrupted"]
        and all(not entry["missing_ids"] for entry in subset.values())
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "commit": files.commit,
        "files": file_report,
        "dev_manifest_check": dev_check,
        "dev_subset": subset,
        "status": "PASS" if healthy else "FAIL",
    }
    return summary, 0 if healthy else 1
