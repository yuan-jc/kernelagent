"""MultiKernelBench snapshot adapter (NVIDIA-platform subset, MIT upstream).

The pinned snapshot (commit + per-file sha256 in
configs/multikernelbench/snapshot-files.manifest.json) uses the same
module-level ``Model`` / ``get_inputs`` / ``get_init_inputs`` problem format
as KernelBench. This adapter only reads bytes and metadata: it never
executes problem code, never imports torch or Triton, and never scores
anything.

Protocol note: the upstream NVIDIA track (utils/correctness.py +
utils/performance.py + config.py, pinned beside the problems) judges with
torch.allclose atol=rtol=1e-4 (bool outputs require exact equality),
num_correct_trials=5, seed_num=1024, and times with CUDA events
(num_warmup=3, num_perf_trials=100, per-trial elapsed ms, no L2 cache
flush). That protocol is registered here under its own name and never
mixes tolerances or timing numbers with the kernelbench, reference-kernels
or userbench tracks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"
_PROTOCOL_NAME = "mkb-upstream-cuda-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_MEMBER = re.compile(r"^reference/([a-z0-9_]+)/([A-Za-z0-9_]+)\.py$")
_NPU_CATEGORY = "npukernelbench"


class MkbManifestError(ValueError):
    """A manifest is malformed, internally inconsistent, or drifted."""


class SnapshotIntegrityError(ValueError):
    """Snapshot bytes do not match the committed manifest."""


class TaskNotFoundError(LookupError):
    """A requested problem does not exist in the snapshot."""

    def __init__(self, category: str, op: str, available: list[str]):
        self.category = category
        self.op = op
        self.available = available
        super().__init__(
            f"problem {category}/{op} not found in snapshot; nearby ops: {available[:8]}"
        )


def task_id(category: str, op: str) -> str:
    return f"mkb-{category}-{op}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MkbManifestError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_content_sha256(path: Path) -> str:
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

    def problem_members(self) -> list[tuple[str, str, str]]:
        """(category, op, snapshot-relative path) for every problem file."""
        found = []
        for item in self.entries:
            match = _SNAPSHOT_MEMBER.match(item.path)
            if match and not match.group(1).startswith(_NPU_CATEGORY):
                found.append((match.group(1), match.group(2), item.path))
        return sorted(found)


def load_files_manifest(path: Path) -> FilesManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "files manifest must be a JSON object")
    _require(
        raw.get("schema_version") == SCHEMA_VERSION,
        f"unsupported files manifest schema_version {raw.get('schema_version')!r}",
    )
    repository = raw.get("repository")
    _require(repository == "wzzll123/MultiKernelBench", f"unexpected repository {repository!r}")
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
        parts = rel.split("/")
        _require(
            rel.endswith(".py")
            and ("/" in rel or rel == "config.py")
            and "\\" not in rel
            and not rel.startswith("/")
            and not re.match(r"^[A-Za-z]:", rel)
            and all(part not in ("", ".", "..") for part in parts),
            f"invalid snapshot-relative path {rel!r}",
        )
        _require(
            rel.startswith("reference/") or rel.startswith("utils/") or rel == "config.py",
            f"path out of snapshot scope {rel!r}",
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
    category: str
    op: str
    sha256: str
    task_id: str


@dataclass(frozen=True, slots=True)
class Problem:
    ref: ProblemRef
    code: str


class MkbSnapshot:
    """Read-only view of one verified MultiKernelBench snapshot directory.

    The committed manifest is the authority: files on disk that are not in
    the manifest are invisible; manifest files that are missing or drifted
    make construction fail loudly."""

    def __init__(self, root: Path, files: FilesManifest, *, verify: bool = True):
        self._root = Path(root)
        self._files = files
        if verify:
            report = verify_snapshot(self._root, files)
            if report["missing"] or report["corrupted"]:
                raise SnapshotIntegrityError(report)
        self._by_key: dict[tuple[str, str], ProblemRef] = {}
        for category, op, rel in files.problem_members():
            entry = files.entry(rel)
            assert entry is not None  # problem_members derives from entries
            self._by_key[(category, op)] = ProblemRef(
                category, op, entry.sha256, task_id(category, op)
            )

    @property
    def root(self) -> Path:
        return self._root

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({category for category, _ in self._by_key}))

    def ops(self, category: str) -> tuple[str, ...]:
        return tuple(op for (cat, op) in sorted(self._by_key) if cat == category)

    def list_problems(self, category: str) -> tuple[ProblemRef, ...]:
        return tuple(ref for (cat, _), ref in sorted(self._by_key.items()) if cat == category)

    def get_problem(self, category: str, op: str) -> Problem:
        ref = self._by_key.get((category, op))
        if ref is None:
            raise TaskNotFoundError(category, op, list(self.ops(category)))
        rel = f"reference/{category}/{op}.py"
        entry = self._files.entry(rel)
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

    def protocol_source(self, rel: str) -> str:
        """Read one pinned protocol file (e.g. utils/correctness.py),
        re-hashed at read time."""
        entry = self._files.entry(rel)
        if entry is None:
            raise MkbManifestError(f"path {rel!r} is not part of the pinned snapshot")
        target = self._root / rel
        if not target.is_file():
            raise SnapshotIntegrityError(
                {"total": 1, "matched": 0, "missing": [rel], "corrupted": []}
            )
        digest = _sha256_file(target)
        if digest != entry.sha256:
            raise SnapshotIntegrityError(
                {
                    "total": 1,
                    "matched": 0,
                    "missing": [],
                    "corrupted": [
                        {"path": rel, "expected_sha256": entry.sha256, "actual_sha256": digest}
                    ],
                }
            )
        return target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Protocol identity: name + content hash over the pinned protocol references.
# ---------------------------------------------------------------------------


def protocol_identity(
    files: FilesManifest, references: tuple[tuple[str, str], ...], *, name: str = _PROTOCOL_NAME
) -> dict:
    payload = {
        "name": name,
        "commit": files.commit,
        "references": {path: sha for path, sha in sorted(references)},
    }
    return {
        "name": name,
        "identity_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "references": dict(sorted(payload["references"].items())),
    }


# ---------------------------------------------------------------------------
# Dev manifest (frozen development subset + protocol registration).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevProblem:
    category: str
    op: str
    path: str
    sha256: str
    task_id: str
    estimated_input_bytes: int


@dataclass(frozen=True, slots=True)
class DevManifest:
    upstream_repository: str
    upstream_commit: str
    files_manifest_path: str
    files_manifest_sha256: str
    problems: tuple[DevProblem, ...]
    protocol: dict
    excluded_over_input_budget: tuple[str, ...]
    excluded_unresolved_input_estimate: tuple[str, ...]


def load_dev_manifest(path: Path) -> DevManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "dev manifest must be a JSON object")
    _require(
        raw.get("schema_version") == SCHEMA_VERSION,
        f"unsupported dev manifest schema_version {raw.get('schema_version')!r}",
    )
    upstream = raw.get("upstream")
    _require(isinstance(upstream, dict), "upstream section is required")
    commit = upstream.get("commit")
    _require(
        isinstance(commit, str) and bool(_COMMIT.fullmatch(commit)),
        "upstream.commit must be 40-hex",
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
        category, op = item.get("category"), item.get("op")
        _require(
            isinstance(category, str) and bool(category) and not category.startswith(_NPU_CATEGORY),
            "problem category must be a non-NPU string",
        )
        _require(isinstance(op, str) and bool(op), "problem op must be a non-empty string")
        path = item.get("path")
        _require(
            path == f"reference/{category}/{op}.py",
            f"problem path {path!r} does not match category/op",
        )
        sha = item.get("sha256")
        _require(isinstance(sha, str) and bool(_SHA256.fullmatch(sha)), f"bad sha256 for {path!r}")
        estimated = item.get("estimated_input_bytes")
        _require(
            isinstance(estimated, int) and not isinstance(estimated, bool) and estimated >= 0,
            f"bad estimated_input_bytes for {path!r}",
        )
        problems.append(DevProblem(category, op, path, sha, task_id(category, op), estimated))
    protocol = raw.get("protocol")
    _require(isinstance(protocol, dict), "protocol section is required")
    _require(
        isinstance(protocol.get("name"), str) and isinstance(protocol.get("identity_sha256"), str),
        "protocol name/identity_sha256 required",
    )
    references_raw = protocol.get("references")
    _require(
        isinstance(references_raw, list) and bool(references_raw), "protocol.references required"
    )
    references = []
    for item in references_raw:
        _require(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
            and bool(_SHA256.fullmatch(item.get("sha256", ""))),
            f"bad protocol reference {item!r}",
        )
        references.append((item["path"], item["sha256"]))
    over = tuple(
        item["path"] if isinstance(item, dict) else item
        for item in raw.get("excluded_over_input_budget", [])
    )
    unresolved = tuple(
        item["path"] if isinstance(item, dict) else item
        for item in raw.get("excluded_unresolved_input_estimate", [])
    )
    return DevManifest(
        upstream_repository=upstream.get("repository", ""),
        upstream_commit=commit,
        files_manifest_path=upstream.get("files_manifest", ""),
        files_manifest_sha256=binding,
        problems=tuple(problems),
        protocol=protocol,
        excluded_over_input_budget=over,
        excluded_unresolved_input_estimate=unresolved,
    )


def check_dev_manifest(
    dev: DevManifest,
    files: FilesManifest,
    files_manifest_path: Path,
    root: Path | None = None,
) -> dict:
    """Cross-check the dev manifest against the files manifest it binds and
    (with ``root``) against actual snapshot bytes including the protocol
    identity hash."""
    if dev.upstream_commit != files.commit:
        raise MkbManifestError(
            f"dev manifest pins commit {dev.upstream_commit!r} but files manifest pins "
            f"{files.commit!r}; refusing to mix snapshot versions"
        )
    actual_binding = canonical_content_sha256(Path(files_manifest_path))
    if dev.files_manifest_sha256 != actual_binding:
        raise MkbManifestError(
            "files manifest bytes drifted from the hash recorded in the dev manifest; "
            "regenerate or repin before use"
        )
    seen: set[str] = set()
    for problem in dev.problems:
        if problem.task_id in seen:
            raise MkbManifestError(f"duplicate dev manifest entry {problem.task_id}")
        seen.add(problem.task_id)
        entry = files.entry(problem.path)
        if entry is None:
            raise MkbManifestError(
                f"dev manifest problem {problem.path!r} is absent from the files manifest"
            )
        if entry.sha256 != problem.sha256:
            raise MkbManifestError(f"dev manifest sha256 mismatch for {problem.path!r}")
    report = {"problems": len(dev.problems), "protocol_references": "not_checked"}
    if root is not None:
        snapshot = MkbSnapshot(root, files, verify=False)
        references = tuple(
            (item["path"], item["sha256"]) for item in dev.protocol.get("references", [])
        )
        identity = protocol_identity(files, references, name=dev.protocol["name"])
        if identity["identity_sha256"] != dev.protocol["identity_sha256"]:
            raise MkbManifestError(
                "protocol identity hash drifted from the recorded value; protocol files or the "
                "snapshot commit changed - repin instead of overriding"
            )
        for path, sha in references:
            target = snapshot.root / path
            if not target.is_file():
                raise MkbManifestError(f"protocol reference missing on disk: {path}")
            if _sha256_file(target) != sha:
                raise MkbManifestError(f"protocol reference hash mismatch: {path}")
        for problem in dev.problems:
            snapshot.get_problem(problem.category, problem.op)
        report["protocol_references"] = "verified"
    return report


def run_verify(root: Path, manifest_path: Path, dev_manifest_path: Path) -> tuple[dict, int]:
    """CLI entry: verify snapshot bytes, dev manifest bindings, protocol
    identity and dev subset presence."""
    try:
        files = load_files_manifest(manifest_path)
        dev = load_dev_manifest(dev_manifest_path)
        dev_check = check_dev_manifest(dev, files, manifest_path, root=root)
    except MkbManifestError as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "root": str(root),
            "status": "FAIL",
            "error": f"dev manifest check failed: {exc}",
        }
        return summary, 1
    file_report = verify_snapshot(root, files)
    snapshot = MkbSnapshot(root, files, verify=False)
    expected = [p.task_id for p in dev.problems]
    available = [
        ref.task_id for cat in snapshot.categories() for ref in snapshot.list_problems(cat)
    ]
    subset = {
        "expected": len(expected),
        "present": len([tid for tid in expected if tid in set(available)]),
        "missing_ids": sorted(set(expected) - set(available)),
    }
    healthy = (
        not file_report["missing"] and not file_report["corrupted"] and not subset["missing_ids"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "commit": files.commit,
        "files": file_report,
        "dev_manifest_check": dev_check,
        "dev_subset": subset,
        "protocol": {
            "name": dev.protocol["name"],
            "identity_sha256": dev.protocol["identity_sha256"],
        },
        "status": "PASS" if healthy else "FAIL",
    }
    return summary, 0 if healthy else 1
