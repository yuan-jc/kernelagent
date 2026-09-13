"""GPU-MODE reference-kernels snapshot adapter (pmpp_v2 + linalg subset).

The pinned snapshot (commit + per-file sha256 in
configs/reference-kernels/snapshot-files.manifest.json) is read verbatim:
this module parses the declarative ``task.yml`` workload lists into
domain objects and records protocol identity, but never executes problem
code and never imports torch.

Protocol note: upstream judges each set with its own pinned ``eval.py`` /
``utils.py`` (verbose_allclose rtol=1e-5/atol=1e-8 by default, per-problem
``check_implementation`` overrides, subprocess isolation, clear_l2_cache
between timed runs, ns duration statistics with max_repeats/max_time
bounds). Local runs replicate that protocol; the official leaderboard runs
on B200/H100/A100/L4 hardware this project does not have, so every local
number is a local replication and must never be compared with official
leaderboard results. This protocol identity is separate from kernelbench,
multikernelbench and userbench tracks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from kernelagent.adapters.benchmarks._restricted_yaml import (
    RestrictedYamlError,
    parse_restricted_yaml,
)
from kernelagent.domain.workload import Workload

SCHEMA_VERSION = "1.0"
_PROTOCOL_NAME = "reference-kernels-local-replica-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROBLEM_MEMBER = re.compile(r"^problems/([a-z0-9_]+)/([a-z0-9_]+)/(.+)$")
_SET_METADATA = re.compile(r"^problems/([a-z0-9_]+)\.yaml$")
_SET_LEVEL_FILE = re.compile(r"^problems/([a-z0-9_]+)/([^/]+)$")
_IN_SCOPE_SETS = ("pmpp_v2", "linalg")


class RkManifestError(ValueError):
    """A manifest is malformed, internally inconsistent, or drifted."""


class SnapshotIntegrityError(ValueError):
    """Snapshot bytes do not match the committed manifest."""


class TaskNotFoundError(LookupError):
    """A requested problem id does not exist in the snapshot."""


def task_id(set_name: str, name: str) -> str:
    return f"rk-{set_name}-{name}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RkManifestError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_content_sha256(path: Path) -> str:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(_canonical_bytes(parsed)).hexdigest()


# ---------------------------------------------------------------------------
# Restricted task.yml parser. Upstream task.yml files use a tiny YAML subset:
# nested mappings by indentation, block scalars (``|`` / ``>``) for
# descriptions, JSON-compatible flow lists/mappings for tests/benchmarks and
# files, and ``#`` comments. The project control plane has no YAML dependency,
# so this adapter parses exactly that subset and rejects anything else rather
# than guessing.
# ---------------------------------------------------------------------------


class TaskYamlError(RestrictedYamlError):
    """task.yml uses syntax outside the supported restricted subset."""


def parse_task_yml(text: str) -> dict:
    """Parse a GPU-MODE ``task.yml`` (or set metadata yaml) under the
    restricted subset; see adapters.benchmarks._restricted_yaml."""
    return parse_restricted_yaml(text)


# ---------------------------------------------------------------------------
# Manifest model (same schema family as configs/kernelbench).
# ---------------------------------------------------------------------------


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

    def set_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    _SET_METADATA.match(e.path).group(1)
                    for e in self.entries
                    if _SET_METADATA.match(e.path)
                }
            )
        )

    def problem_directories(self, set_name: str) -> tuple[str, ...]:
        found = set()
        for entry in self.entries:
            match = _PROBLEM_MEMBER.match(entry.path)
            if match and match.group(1) == set_name:
                found.add(f"{set_name}/{match.group(2)}")
        return tuple(sorted(found))

    def set_members(self, directory: str) -> tuple[str, ...]:
        """Snapshot-relative member files of one problem directory."""
        prefix = f"problems/{directory}/"
        return tuple(sorted(e.path for e in self.entries if e.path.startswith(prefix)))


def _validate_rel_path(rel: str) -> None:
    parts = rel.split("/")
    _require(
        (bool(rel[:-1]) if rel.endswith("/") else bool(rel))
        and ("/" in rel or rel in ("LICENSE",))
        and "\\" not in rel
        and not rel.startswith("/")
        and not re.match(r"^[A-Za-z]:", rel)
        and all(part not in ("", ".", "..") for part in parts),
        f"invalid snapshot-relative path {rel!r}",
    )


def load_files_manifest(path: Path) -> FilesManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "files manifest must be a JSON object")
    _require(
        raw.get("schema_version") == SCHEMA_VERSION,
        f"unsupported files manifest schema_version {raw.get('schema_version')!r}",
    )
    repository = raw.get("repository")
    _require(repository == "gpu-mode/reference-kernels", f"unexpected repository {repository!r}")
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
        _validate_rel_path(rel)
        _require(rel.startswith("problems/") or rel == "LICENSE", f"path out of scope {rel!r}")
        parts = rel.split("/")
        if rel.startswith("problems/") and len(parts) >= 2:
            # only the in-scope problem sets may enter the snapshot
            _require(
                parts[1] in _IN_SCOPE_SETS or "." in parts[1],
                f"problem set out of snapshot scope: {rel!r}",
            )
        _require(rel not in seen, f"duplicate manifest path {rel!r}")
        seen.add(rel)
        blob = item.get("git_blob_sha1")
        _require(
            isinstance(blob, str) and bool(re.fullmatch(r"[0-9a-f]{40}", blob)),
            f"bad git_blob_sha1 for {rel!r}",
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


# ---------------------------------------------------------------------------
# Problem model: task.yml -> declarative domain Workloads.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RkInputPort:
    """Declarative port binding for one problem input tensor.

    ``dims`` entries are integers or the names of TestSpec keys; the dev
    manifest freezes the table from the pinned reference.py semantics, so
    workload shapes stay static facts, never guesses."""

    name: str
    dtype: str
    dims: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class RkProblem:
    set_name: str
    name: str
    directory: str
    task_id: str
    members: tuple[str, ...]  # snapshot-relative paths
    task_yml: dict
    inputs: tuple[RkInputPort, ...]
    official_gpus: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RkWorkloadSpec:
    workload_id: str
    task_id: str
    kind: str  # test | benchmark
    spec: dict  # raw TestSpec, e.g. {"size": 1023, "seed": 4242}
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    seed: int

    def workload(self, *, operator_id: str | None = None, weight: float = 1.0) -> Workload:
        return Workload(
            workload_id=self.workload_id,
            operator_id=operator_id or self.task_id,
            shapes=self.shapes,
            dtypes=self.dtypes,
            seed=self.seed,
            weight=weight,
        )


def _resolve_dims(dims, spec: dict, task_id_str: str) -> tuple[int, ...]:
    resolved = []
    for dim in dims:
        if isinstance(dim, bool):
            raise RkManifestError(f"bool dim in {task_id_str}")
        if isinstance(dim, int):
            resolved.append(dim)
            continue
        if isinstance(dim, str) and dim in spec:
            value = spec[dim]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RkManifestError(
                    f"spec key {dim!r} for {task_id_str} must be a positive int; got {value!r}"
                )
            resolved.append(value)
            continue
        raise RkManifestError(
            f"dim {dim!r} for {task_id_str} is neither an int nor a TestSpec key of {sorted(spec)}"
        )
    return tuple(resolved)


def _validate_problem_ports(problem: RkProblem) -> None:
    yml = problem.task_yml
    for kind, key in (("test", "tests"), ("benchmark", "benchmarks")):
        specs = yml.get(key, [])
        if not isinstance(specs, list) or not specs:
            raise RkManifestError(f"{problem.task_id}: {key} must be a non-empty list")
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict) or "seed" not in spec:
                raise RkManifestError(
                    f"{problem.task_id}: {key}[{index}] must be a mapping containing 'seed'"
                )
            for port in problem.inputs:
                _resolve_dims(port.dims, spec, f"{problem.task_id}:{kind}[{index}]")
    for key in ("test_timeout", "benchmark_timeout"):
        value = yml.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{problem.task_id}: {key} must be a positive int",
        )


def problem_from_manifest(
    files: FilesManifest,
    dev_problem: "DevProblem",
    set_yamls: dict[str, dict],
) -> RkProblem:
    set_name, directory = dev_problem.set_name, dev_problem.directory
    task_yml_path = f"problems/{directory}/task.yml"
    entry = files.entry(task_yml_path)
    _require(entry is not None, f"{task_yml_path} absent from files manifest")
    if entry.sha256 != dev_problem.task_yml_sha256:
        raise RkManifestError(f"dev manifest task.yml sha256 mismatch for {directory!r}")
    embedded_sha = hashlib.sha256(dev_problem.task_yml_text.encode("utf-8")).hexdigest()
    if embedded_sha != dev_problem.task_yml_sha256:
        raise RkManifestError(
            f"embedded task.yml text does not hash to task_yml_sha256 for {directory!r}"
        )
    parsed = parse_task_yml(dev_problem.task_yml_text)
    for key, value in dev_problem.timeouts.items():
        if value is None:
            continue
        _require(
            parsed.get(key) == value,
            f"{directory}: {key} drifted between dev manifest and task.yml",
        )
    inputs = tuple(
        RkInputPort(name=port.name, dtype=port.dtype, dims=tuple(port.dims))
        for port in dev_problem.inputs
    )
    gpus = []
    for item in set_yamls.get(set_name, {}).get("problems", []):
        if item.get("directory") == directory:
            name = item.get("name")
            _require(
                isinstance(name, str) and name == dev_problem.name,
                f"set metadata name mismatch for {directory!r}",
            )
            gpus = list(item.get("gpus", []))
    problem = RkProblem(
        set_name=set_name,
        name=dev_problem.name,
        directory=directory,
        task_id=task_id(set_name, dev_problem.name),
        members=files.set_members(directory),
        task_yml=parsed,
        inputs=inputs,
        official_gpus=tuple(gpus),
    )
    _validate_problem_ports(problem)
    return problem


def workload_specs(problem: RkProblem) -> tuple[RkWorkloadSpec, ...]:
    """Declarative (shape, seed) workloads for the pinned task.yml.

    tests feed correctness, benchmarks feed the rk timing track; both are
    materialized without executing any problem code."""
    specs: list[RkWorkloadSpec] = []
    for kind, key in (("test", "tests"), ("benchmark", "benchmarks")):
        for index, spec in enumerate(problem.task_yml[key]):
            shapes = tuple(
                _resolve_dims(port.dims, spec, f"{problem.task_id}:{kind}[{index}]")
                for port in problem.inputs
            )
            dtypes = tuple(port.dtype for port in problem.inputs)
            specs.append(
                RkWorkloadSpec(
                    workload_id=f"{problem.task_id}-{kind}{index:02d}",
                    task_id=problem.task_id,
                    kind=kind,
                    spec=dict(spec),
                    shapes=shapes,
                    dtypes=dtypes,
                    seed=int(spec["seed"]),
                )
            )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Protocol identity: name + content hash over the pinned protocol closure.
# ---------------------------------------------------------------------------


def protocol_identity(
    files: FilesManifest,
    problems: tuple[RkProblem, ...],
    *,
    name: str = _PROTOCOL_NAME,
) -> dict:
    """Named identity hash over every file the local protocol depends on:
    set-level eval/utils/template files, each problem's eval/reference/task
    sources and the LICENSE. Any snapshot or protocol change re-hashes it."""
    closure: dict[str, str] = {}
    for entry in files.entries:
        rel = entry.path
        is_set_level = bool(_SET_LEVEL_FILE.match(rel)) and rel.rsplit("/", 1)[1] in (
            "eval.py",
            "utils.py",
            "template.py",
        )
        is_problem_source = bool(_PROBLEM_MEMBER.match(rel)) and rel.rsplit("/", 1)[1] in (
            "eval.py",
            "reference.py",
            "task.py",
            "validation.py",
        )
        if is_set_level or is_problem_source or rel == "LICENSE":
            closure[rel] = entry.sha256
    payload = {
        "name": name,
        "commit": files.commit,
        "problems": sorted(problem.task_id for problem in problems),
        "files": dict(sorted(closure.items())),
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return {
        "name": name,
        "identity_sha256": digest,
        "files": payload["files"],
        "problems": payload["problems"],
    }


# ---------------------------------------------------------------------------
# Snapshot facade (mirrors KernelBenchSnapshot: manifest is the authority).
# ---------------------------------------------------------------------------


class ReferenceKernelsSnapshot:
    def __init__(self, root: Path, files: FilesManifest, *, verify: bool = True):
        self._root = Path(root)
        self._files = files
        if verify:
            report = verify_snapshot(self._root, files)
            if report["missing"] or report["corrupted"]:
                raise SnapshotIntegrityError(report)
        self._set_yamls: dict[str, dict] = {}
        for entry in files.entries:
            match = _SET_METADATA.match(entry.path)
            if match:
                self._set_yamls[match.group(1)] = parse_task_yml(
                    (self._root / entry.path).read_text(encoding="utf-8")
                )

    @property
    def root(self) -> Path:
        return self._root

    def set_names(self) -> tuple[str, ...]:
        return self._files.set_names()

    def read_member(self, rel: str) -> str:
        entry = self._files.entry(rel)
        if entry is None:
            raise TaskNotFoundError(f"{rel!r} is not part of the pinned snapshot")
        target = self._root / rel
        if not target.is_file():
            raise SnapshotIntegrityError(
                {
                    "total": 1,
                    "matched": 0,
                    "missing": [rel],
                    "corrupted": [],
                }
            )
        digest = _sha256_file(target)
        if digest != entry.sha256:
            raise SnapshotIntegrityError(
                {
                    "total": 1,
                    "matched": 0,
                    "missing": [],
                    "corrupted": [
                        {
                            "path": rel,
                            "expected_sha256": entry.sha256,
                            "actual_sha256": digest,
                        }
                    ],
                }
            )
        return target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dev manifest (frozen development subset + protocol identity registration).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevInput:
    name: str
    dtype: str
    dims: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class DevProblem:
    set_name: str
    name: str
    directory: str
    task_id: str
    members: tuple[str, ...]
    member_sha256: tuple[str, ...]
    task_yml_sha256: str
    task_yml_text: str
    inputs: tuple[DevInput, ...]
    timeouts: dict


@dataclass(frozen=True, slots=True)
class DevManifest:
    upstream_repository: str
    upstream_commit: str
    files_manifest_path: str
    files_manifest_sha256: str
    problems: tuple[DevProblem, ...]
    protocol: dict
    license_note: str


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
    _require(isinstance(problems_raw, list) and bool(problems_raw), "problems must be non-empty")
    problems: list[DevProblem] = []
    for item in problems_raw:
        _require(isinstance(item, dict), "problem entries must be objects")
        set_name = item.get("set")
        name = item.get("name")
        directory = item.get("directory")
        _require(
            isinstance(set_name, str) and set_name in _IN_SCOPE_SETS,
            f"problem set out of scope: {set_name!r}",
        )
        _require(isinstance(name, str) and bool(name), "problem name must be a non-empty string")
        _require(
            isinstance(directory, str) and bool(directory),
            "problem directory must be a non-empty string",
        )
        _require(directory.startswith(f"{set_name}/"), f"directory {directory!r} outside its set")
        members = item.get("members")
        _require(isinstance(members, list) and bool(members), f"{directory}: members required")
        member_sha = item.get("member_sha256")
        _require(
            isinstance(member_sha, list) and len(member_sha) == len(members),
            f"{directory}: member_sha256 must align with members",
        )
        for rel, sha in zip(members, member_sha):
            _require(
                isinstance(rel, str) and rel.startswith(f"problems/{directory}/"),
                f"{directory}: member {rel!r} outside the problem directory",
            )
            _require(
                isinstance(sha, str) and bool(_SHA256.fullmatch(sha)),
                f"{directory}: bad member sha256 for {rel!r}",
            )
        task_yml_sha = item.get("task_yml_sha256")
        _require(
            isinstance(task_yml_sha, str) and bool(_SHA256.fullmatch(task_yml_sha)),
            f"{directory}: bad task_yml_sha256",
        )
        task_yml_text = item.get("task_yml_text")
        _require(
            isinstance(task_yml_text, str) and bool(task_yml_text),
            f"{directory}: embedded task_yml_text required",
        )
        inputs_raw = item.get("inputs")
        _require(isinstance(inputs_raw, list) and bool(inputs_raw), f"{directory}: inputs required")
        inputs = []
        for port in inputs_raw:
            _require(
                isinstance(port, dict)
                and isinstance(port.get("name"), str)
                and isinstance(port.get("dtype"), str)
                and isinstance(port.get("dims"), list)
                and all(
                    isinstance(d, (int, str)) and not isinstance(d, bool) for d in port["dims"]
                ),
                f"{directory}: malformed input port {port!r}",
            )
            inputs.append(DevInput(port["name"], port["dtype"], tuple(port["dims"])))
        timeouts = {k: item.get(k) for k in ("test_timeout", "benchmark_timeout", "ranked_timeout")}
        problems.append(
            DevProblem(
                set_name=set_name,
                name=name,
                directory=directory,
                task_id=task_id(set_name, name),
                members=tuple(members),
                member_sha256=tuple(member_sha),
                task_yml_sha256=task_yml_sha,
                task_yml_text=task_yml_text,
                inputs=tuple(inputs),
                timeouts=timeouts,
            )
        )
    protocol = raw.get("protocol")
    _require(isinstance(protocol, dict), "protocol section is required")
    _require(
        isinstance(protocol.get("name"), str) and isinstance(protocol.get("identity_sha256"), str),
        "protocol name/identity_sha256 required",
    )
    license_note = raw.get("license_note")
    _require(isinstance(license_note, str) and bool(license_note), "license_note required")
    return DevManifest(
        upstream_repository=upstream.get("repository", ""),
        upstream_commit=commit,
        files_manifest_path=upstream.get("files_manifest", ""),
        files_manifest_sha256=binding,
        problems=tuple(problems),
        protocol=protocol,
        license_note=license_note,
    )


def check_dev_manifest(
    dev: DevManifest,
    files: FilesManifest,
    files_manifest_path: Path,
    root: Path | None = None,
) -> dict:
    """Cross-check dev manifest bindings and (with root) snapshot bytes."""
    if dev.upstream_commit != files.commit:
        raise RkManifestError(
            f"dev manifest pins commit {dev.upstream_commit!r} but files manifest pins "
            f"{files.commit!r}; refusing to mix snapshot versions"
        )
    actual_binding = canonical_content_sha256(Path(files_manifest_path))
    if dev.files_manifest_sha256 != actual_binding:
        raise RkManifestError(
            "files manifest bytes drifted from the hash recorded in the dev manifest; "
            "regenerate or repin before use"
        )
    seen = set()
    snapshot = ReferenceKernelsSnapshot(root, files, verify=False) if root is not None else None
    for problem in dev.problems:
        if problem.task_id in seen:
            raise RkManifestError(f"duplicate dev manifest entry {problem.task_id}")
        seen.add(problem.task_id)
        for rel, sha in zip(problem.members, problem.member_sha256):
            entry = files.entry(rel)
            if entry is None:
                raise RkManifestError(f"dev manifest member {rel!r} absent from files manifest")
            if entry.sha256 != sha:
                raise RkManifestError(f"dev manifest sha256 mismatch for {rel!r}")
        task_yml = f"problems/{problem.directory}/task.yml"
        entry = files.entry(task_yml)
        if entry is None or entry.sha256 != problem.task_yml_sha256:
            raise RkManifestError(f"dev manifest task.yml binding mismatch for {task_yml!r}")
    built = tuple(
        problem_from_manifest(
            files,
            DevProblem(
                set_name=p.set_name,
                name=p.name,
                directory=p.directory,
                task_id=p.task_id,
                members=p.members,
                member_sha256=p.member_sha256,
                task_yml_sha256=p.task_yml_sha256,
                task_yml_text=p.task_yml_text,
                inputs=tuple(DevInput(i.name, i.dtype, i.dims) for i in p.inputs),
                timeouts=p.timeouts,
            ),
            set_yamls={} if snapshot is None else snapshot._set_yamls,
        )
        for p in dev.problems
    )
    report: dict = {"problems": len(dev.problems), "protocol_references": "not_checked"}
    if root is not None:
        identity = protocol_identity(files, built, name=dev.protocol["name"])
        if identity["identity_sha256"] != dev.protocol["identity_sha256"]:
            raise RkManifestError(
                "protocol identity hash drifted from the recorded value; snapshot or protocol "
                "files changed - repin instead of overriding"
            )
        for rel, sha in identity["files"].items():
            # read_member re-hashes the file on disk and raises on drift.
            snapshot.read_member(rel)
            if files.entry(rel).sha256 != sha:
                raise RkManifestError(f"protocol file hash mismatch: {rel}")
        report["protocol_references"] = "verified"
        for problem in built:
            for rel in problem.members:
                snapshot.read_member(rel)
    return report


def run_verify(root: Path, manifest_path: Path, dev_manifest_path: Path) -> tuple[dict, int]:
    """CLI entry: verify snapshot bytes, dev manifest bindings, protocol
    identity and declarative workload materialization."""
    try:
        files = load_files_manifest(manifest_path)
        dev = load_dev_manifest(dev_manifest_path)
        dev_check = check_dev_manifest(dev, files, manifest_path, root=root)
    except (RkManifestError, TaskYamlError, SnapshotIntegrityError, TaskNotFoundError) as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "root": str(root),
            "status": "FAIL",
            "error": f"dev manifest check failed: {exc}",
        }
        return summary, 1
    file_report = verify_snapshot(root, files)
    workload_counts = {"test": 0, "benchmark": 0}
    if not file_report["missing"] and not file_report["corrupted"]:
        snapshot = ReferenceKernelsSnapshot(root, files, verify=False)
        for problem_dev in dev.problems:
            problem = problem_from_manifest(
                files,
                problem_dev,
                set_yamls=snapshot._set_yamls,
            )
            for spec in workload_specs(problem):
                workload_counts[spec.kind] += 1
    healthy = (
        not file_report["missing"]
        and not file_report["corrupted"]
        and dev_check.get("protocol_references") == "verified"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "commit": files.commit,
        "files": file_report,
        "dev_manifest_check": dev_check,
        "workloads": workload_counts,
        "protocol": {
            "name": dev.protocol["name"],
            "identity_sha256": dev.protocol["identity_sha256"],
            "scope": "local replication; official leaderboard numbers are not comparable",
        },
        "status": "PASS" if healthy else "FAIL",
    }
    return summary, 0 if healthy else 1
