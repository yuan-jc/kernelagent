"""user-bench: user-authored torch operator problems (suite loader).

A suite is a directory of problem definitions authored by a user:

    <suite_root>/
      suite.yaml          protocol declaration (timing/correctness identity)
      manifest.json       frozen sha256 over suite.yaml and every problem file
      problems/<name>/
        problem.yaml      declarative problem spec
        reference.py      the user reference torch callable (worker-side only)

The loader maps each problem onto the existing domain objects
(OperatorSpec/IOPort, Workload, OptimizationTask) and dispatch guards
(WorkloadSpec) without any code change to domain, and WITHOUT importing
torch: the reference callable is only ever imported inside a worker/driver
process (see configs/userbench/userbench_driver.py).

Protocol identity: every suite declares its correctness/timing protocol
names (default ``userbench-correctness-v1`` / ``userbench-timing-v1``) and
timing parameters; the loader registers a content hash over the frozen
manifest plus protocol parameters, so any tolerance/timing/problem change
changes the identity. Tolerances are authored per problem and frozen at
manifest generation; they must stay within the suite tolerance ceiling and
can never be relaxed to pass a failing candidate.
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
from kernelagent.dispatch import WorkloadSpec
from kernelagent.domain.operator import GRANULARITIES, IOPort, OperatorSpec
from kernelagent.domain.task import OptimizationTask
from kernelagent.domain.workload import Workload

API_VERSION = "kernelagent/userbench/v1alpha1"
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
DISTRIBUTIONS = frozenset({"randn", "rand", "uniform", "zeros", "ones", "arange"})
POLICIES = frozenset({"allclose", "exact"})
DEFAULT_CORRECTNESS_PROTOCOL = "userbench-correctness-v1"
DEFAULT_TIMING_PROTOCOL = "userbench-timing-v1"


class UserBenchError(ValueError):
    """A suite or problem definition is malformed or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UserBenchError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Restricted YAML parsing (same subset family as the reference-kernels
# adapter: nested mappings, flow JSON values, scalars, block scalars, '#'
# comments). The control plane has no YAML dependency.
# ---------------------------------------------------------------------------


class SuiteYamlError(RestrictedYamlError):
    """suite.yaml/problem.yaml uses syntax outside the supported subset."""


def parse_suite_yaml(text: str, *, where: str = "") -> dict:
    """Parse a user-bench suite/problem yaml under the restricted subset;
    see adapters.benchmarks._restricted_yaml. ``where`` only labels errors
    raised by this module's own validation."""
    return parse_restricted_yaml(text)


# ---------------------------------------------------------------------------
# Manifest: frozen sha256 over suite.yaml and every problem file.
# ---------------------------------------------------------------------------


def generate_manifest(suite_root: Path) -> dict:
    """Freeze a suite: sha256 over suite.yaml and every problem file.
    Regenerating is the only way to change the manifest - and any tolerance,
    timing or reference change then changes the protocol identity."""
    suite_root = Path(suite_root)
    suite_yaml = suite_root / "suite.yaml"
    files: dict[str, str] = {"suite.yaml": _sha256_bytes(suite_yaml.read_bytes())}
    problems_dir = suite_root / "problems"
    if problems_dir.is_dir():
        for path in sorted(problems_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(suite_root).as_posix()
                _require(
                    all(not part.startswith(".") for part in Path(rel).parts),
                    f"unexpected hidden file in suite: {rel}",
                )
                files[rel] = _sha256_bytes(path.read_bytes())
    return {
        "schema_version": "1.0",
        "suite_root": Path(suite_root).name,
        "files": files,
    }


def load_manifest(suite_root: Path) -> dict:
    raw = json.loads((Path(suite_root) / "manifest.json").read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "suite manifest must be a JSON object")
    _require(
        raw.get("schema_version") == "1.0",
        f"unsupported suite manifest schema_version {raw.get('schema_version')!r}",
    )
    _require(isinstance(raw.get("files"), dict) and raw["files"], "suite manifest files required")
    return raw


def verify_manifest(suite_root: Path, manifest: dict) -> dict:
    suite_root = Path(suite_root)
    matched, missing, corrupted = 0, [], []
    for rel, sha in sorted(manifest["files"].items()):
        _require(
            not rel.startswith("/") and ".." not in Path(rel).parts and "\\" not in rel,
            f"unsafe manifest path {rel!r}",
        )
        target = suite_root / rel
        if not target.is_file():
            missing.append(rel)
            continue
        digest = _sha256_bytes(target.read_bytes())
        if digest != sha:
            corrupted.append({"path": rel, "expected_sha256": sha, "actual_sha256": digest})
        else:
            matched += 1
    return {
        "total": len(manifest["files"]),
        "matched": matched,
        "missing": missing,
        "corrupted": corrupted,
    }


# ---------------------------------------------------------------------------
# Suite and problem model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tolerance:
    rtol: float
    atol: float
    equal_nan: bool
    policy: str


@dataclass(frozen=True, slots=True)
class InputGeneration:
    distribution: str
    scale: float
    offset: float


@dataclass(frozen=True, slots=True)
class WorkloadDef:
    workload_id: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    seed: int
    weight: float
    guard_max_shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class UserBenchProblem:
    suite_name: str
    name: str
    directory: str  # suite-relative
    granularity: str
    description: str
    reference_source: str  # suite-relative path to reference.py
    reference_entry: str
    nondeterministic: bool
    inputs: tuple[IOPort, ...]
    outputs: tuple[IOPort, ...]
    tolerance: Tolerance
    input_generation: InputGeneration
    workloads: tuple[WorkloadDef, ...]
    problem_sha256: str
    reference_sha256: str

    @property
    def task_id(self) -> str:
        return f"userbench-{self.suite_name}-{self.name}"

    @property
    def operator_id(self) -> str:
        return f"userbench/{self.suite_name}/{self.name}"

    def to_task(self) -> OptimizationTask:
        """Map onto the existing domain model (domain stays untouched)."""
        spec = OperatorSpec(
            operator_id=self.operator_id,
            granularity=self.granularity,  # validated against GRANULARITIES at load
            inputs=self.inputs,
            outputs=self.outputs,
            notes=f"userbench problem {self.name} (suite {self.suite_name})",
        )
        workloads = tuple(
            Workload(
                workload_id=w.workload_id,
                operator_id=self.operator_id,
                shapes=w.shapes,
                dtypes=w.dtypes,
                seed=w.seed,
                weight=w.weight,
            )
            for w in self.workloads
        )
        return OptimizationTask(
            task_id=self.task_id,
            operator=spec,
            workloads=workloads,
            track="upstream_compatible",
        )

    def workload_specs(self) -> tuple[WorkloadSpec, ...]:
        """Dispatch guards: one WorkloadSpec per declared workload; missing
        workloads keep the existing unknown-score semantics."""
        return tuple(
            WorkloadSpec(
                workload_id=w.workload_id,
                weight=w.weight,
                required=True,
                max_shape=w.guard_max_shape,
            )
            for w in self.workloads
        )


@dataclass(frozen=True, slots=True)
class UserBenchSuite:
    name: str
    root: Path
    suite: dict
    problems: tuple[UserBenchProblem, ...]
    protocol: dict
    manifest: dict
    manifest_sha256: str

    @property
    def correctness_protocol(self) -> str:
        return self.protocol.get("correctness", DEFAULT_CORRECTNESS_PROTOCOL)

    @property
    def timing_protocol(self) -> str:
        return self.protocol.get("timing", DEFAULT_TIMING_PROTOCOL)

    def protocol_identity(self) -> dict:
        """Named protocol identity: content hash over the frozen suite
        manifest and the declared protocol parameters. Any tolerance,
        timing, problem or reference change re-hashes the identity."""
        payload = {
            "correctness": self.correctness_protocol,
            "timing": self.timing_protocol,
            "timing_params": self.protocol.get("timing_params", {}),
            "default_tolerance_ceiling": self.protocol.get("default_tolerance_ceiling", {}),
            "allowed_dependencies": self.suite.get("allowed_dependencies", []),
            "manifest_sha256": self.manifest_sha256,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {
            "correctness": self.correctness_protocol,
            "timing": self.timing_protocol,
            "identity_sha256": _sha256_bytes(canonical.encode("utf-8")),
        }


def _parse_ports(raw, *, where: str) -> tuple[IOPort, ...]:
    _require(isinstance(raw, list) and bool(raw), f"{where}: must be a non-empty list")
    seen = set()
    ports = []
    for index, item in enumerate(raw):
        _require(
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("dtype"), str)
            and isinstance(item.get("shape"), list),
            f"{where}[{index}]: must declare name/dtype/shape",
        )
        name = item["name"]
        _require(bool(name), f"{where}[{index}]: name must be non-empty")
        _require(name not in seen, f"{where}: duplicate name {name!r}")
        seen.add(name)
        shape = item["shape"]
        _require(
            all(
                isinstance(d, int) and not isinstance(d, bool) and (d == -1 or d > 0) for d in shape
            ),
            f"{where}[{index}]: shape entries must be positive ints or -1",
        )
        ports.append(IOPort(name=name, dtype=item["dtype"], shape=tuple(shape)))
    return tuple(ports)


def _parse_workloads(raw, problem: dict, *, where: str) -> tuple[WorkloadDef, ...]:
    _require(isinstance(raw, list) and bool(raw), f"{where}: must be a non-empty list")
    inputs = problem["inputs"]
    _require(
        isinstance(inputs, list) and inputs,
        f"{where}: problem must declare inputs before workloads",
    )
    seen = set()
    defs = []
    for index, item in enumerate(raw):
        _require(
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("shapes"), list)
            and isinstance(item.get("dtypes"), list),
            f"{where}[{index}]: must declare id/shapes/dtypes",
        )
        wid = item["id"]
        _require(bool(wid), f"{where}[{index}]: id must be non-empty")
        _require(wid not in seen, f"{where}: duplicate workload id {wid!r}")
        seen.add(wid)
        shapes_raw = item["shapes"]
        dtypes = item["dtypes"]
        _require(
            len(shapes_raw) == len(inputs) == len(dtypes),
            f"{wid}: shapes/dtypes length must match declared inputs ({len(inputs)})",
        )
        shapes = []
        for t_index, shape in enumerate(shapes_raw):
            _require(
                isinstance(shape, list)
                and shape
                and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape),
                f"{wid}: workload shapes must be concrete positive ints (got {shape!r})",
            )
            declared = inputs[t_index].get("shape", [])
            _require(
                len(declared) == len(shape),
                f"{wid}: input {t_index} rank {len(shape)} does not match declared rank "
                f"{len(declared)}",
            )
            shapes.append(tuple(shape))
        seed = item.get("seed", 0)
        _require(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            f"{wid}: seed must be a non-negative int",
        )
        weight = item.get("weight", 1.0)
        _require(
            isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight > 0,
            f"{wid}: weight must be a positive number",
        )
        guard = item.get("guard_max_shape")
        guard_shape = None
        if guard is not None:
            _require(
                isinstance(guard, list)
                and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in guard),
                f"{wid}: guard_max_shape must be positive ints",
            )
            guard_shape = tuple(guard)
        defs.append(
            WorkloadDef(
                workload_id=wid,
                shapes=tuple(shapes),
                dtypes=tuple(dtypes),
                seed=seed,
                weight=float(weight),
                guard_max_shape=guard_shape,
            )
        )
    return tuple(defs)


def _parse_tolerance(raw, ceiling: dict, *, where: str) -> Tolerance:
    _require(
        isinstance(raw, dict)
        and isinstance(raw.get("rtol"), (int, float))
        and isinstance(raw.get("atol"), (int, float)),
        f"{where}: tolerance must declare numeric rtol/atol",
    )
    rtol, atol = float(raw["rtol"]), float(raw["atol"])
    _require(rtol >= 0 and atol >= 0, f"{where}: rtol/atol must be non-negative")
    policy = raw.get("policy", "allclose")
    _require(
        policy in POLICIES,
        f"{where}: tolerance policy {policy!r} not in {sorted(POLICIES)}",
    )
    equal_nan = raw.get("equal_nan", False)
    _require(isinstance(equal_nan, bool), f"{where}: equal_nan must be a bool")
    if ceiling:
        ceiling_rtol = float(ceiling.get("rtol", 0.0))
        ceiling_atol = float(ceiling.get("atol", 0.0))
        _require(
            rtol <= ceiling_rtol and atol <= ceiling_atol,
            f"{where}: tolerance (rtol={rtol}, atol={atol}) exceeds the suite ceiling "
            f"(rtol={ceiling_rtol}, atol={ceiling_atol}); relaxing tolerance to pass a "
            "candidate is not allowed",
        )
    return Tolerance(rtol=rtol, atol=atol, equal_nan=equal_nan, policy=policy)


def load_problem(suite: "UserBenchSuite", problem_name: str) -> UserBenchProblem:
    """Validate one problem definition against the suite rules and map it to
    domain objects. Raises UserBenchError on missing fields, illegal
    tolerances, workload conflicts or drift from the frozen manifest."""
    directory = f"problems/{problem_name}"
    problem_path = suite.root / directory / "problem.yaml"
    reference_path = suite.root / directory / "reference.py"
    _require(problem_path.is_file(), f"{directory}/problem.yaml missing")
    _require(reference_path.is_file(), f"{directory}/reference.py missing")
    _require(
        bool(NAME_PATTERN.fullmatch(problem_name)),
        f"problem name {problem_name!r} must match {NAME_PATTERN.pattern}",
    )
    problem_rel = f"{directory}/problem.yaml"
    reference_rel = f"{directory}/reference.py"
    recorded = suite.manifest["files"]
    _require(problem_rel in recorded, f"{problem_rel} is not covered by the frozen suite manifest")
    _require(
        reference_rel in recorded,
        f"{reference_rel} is not covered by the frozen suite manifest",
    )
    problem_sha = recorded[problem_rel]
    reference_sha = recorded[reference_rel]
    _require(
        _sha256_bytes(problem_path.read_bytes()) == problem_sha,
        f"{problem_rel} drifted from the frozen suite manifest",
    )
    _require(
        _sha256_bytes(reference_path.read_bytes()) == reference_sha,
        f"{reference_rel} drifted from the frozen suite manifest",
    )
    where = problem_rel
    problem = parse_suite_yaml(problem_path.read_text(encoding="utf-8"), where=where)
    _require(
        problem.get("api_version") == API_VERSION,
        f"{where}: api_version must be {API_VERSION!r}",
    )
    body = problem.get("problem")
    _require(isinstance(body, dict), f"{where}: 'problem' section required")
    name = body.get("name")
    _require(name == problem_name, f"{where}: problem.name must equal its directory name")
    granularity = body.get("granularity", "operator")
    _require(granularity in GRANULARITIES, f"{where}: granularity {granularity!r} unsupported")
    reference = body.get("reference")
    _require(
        isinstance(reference, dict)
        and reference.get("source") == "reference.py"
        and isinstance(reference.get("entry"), str)
        and bool(reference["entry"]),
        f"{where}: reference must declare source 'reference.py' and an entry symbol",
    )
    _require(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reference["entry"]) is not None,
        f"{where}: reference entry must be a valid Python identifier",
    )
    inputs = _parse_ports(body.get("inputs"), where=f"{where}: inputs")
    outputs = _parse_ports(body.get("outputs"), where=f"{where}: outputs")
    ceiling = suite.protocol.get("default_tolerance_ceiling", {})
    tolerance = _parse_tolerance(body.get("tolerance"), ceiling, where=f"{where}: tolerance")
    generation_raw = body.get("input_generation", {})
    _require(isinstance(generation_raw, dict), f"{where}: input_generation must be a mapping")
    distribution = generation_raw.get("distribution", "randn")
    _require(
        distribution in DISTRIBUTIONS,
        f"{where}: input distribution {distribution!r} not in {sorted(DISTRIBUTIONS)}",
    )
    scale = float(generation_raw.get("scale", 1.0))
    offset = float(generation_raw.get("offset", 0.0))
    workloads = _parse_workloads(body.get("workloads"), body, where=f"{where}: workloads")
    nondeterministic = bool(body.get("nondeterministic", False))
    _require(
        not nondeterministic,
        f"{where}: nondeterministic references are out of scope for this loader "
        "(statistical tolerance path not implemented; declare a deterministic reference)",
    )
    return UserBenchProblem(
        suite_name=suite.name,
        name=name,
        directory=directory,
        granularity=granularity,
        description=body.get("description", ""),
        reference_source=reference_rel,
        reference_entry=reference["entry"],
        nondeterministic=nondeterministic,
        inputs=inputs,
        outputs=outputs,
        tolerance=tolerance,
        input_generation=InputGeneration(distribution, scale, offset),
        workloads=workloads,
        problem_sha256=problem_sha,
        reference_sha256=reference_sha,
    )


def load_suite(suite_root: Path) -> UserBenchSuite:
    """Load and fully validate a frozen suite (manifest drift is fatal)."""
    suite_root = Path(suite_root)
    suite_yaml = suite_root / "suite.yaml"
    _require(suite_yaml.is_file(), f"{suite_yaml} missing")
    manifest = load_manifest(suite_root)
    report = verify_manifest(suite_root, manifest)
    _require(
        not report["missing"] and not report["corrupted"],
        f"suite manifest verification failed: missing={report['missing']} "
        f"corrupted={report['corrupted']}",
    )
    suite = parse_suite_yaml(suite_yaml.read_text(encoding="utf-8"), where="suite.yaml")
    _require(suite.get("api_version") == API_VERSION, "suite.yaml: api_version mismatch")
    name = suite.get("suite")
    _require(
        isinstance(name, str) and bool(NAME_PATTERN.fullmatch(name)),
        "suite.yaml: suite name must match [a-z0-9_]+",
    )
    protocol = suite.get("protocol")
    _require(isinstance(protocol, dict), "suite.yaml: protocol section required")
    deps = suite.get("allowed_dependencies", [])
    _require(
        isinstance(deps, list) and all(isinstance(d, str) for d in deps),
        "suite.yaml: allowed_dependencies must be a list of strings",
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    holder = UserBenchSuite(
        name=name,
        root=suite_root,
        suite=suite,
        problems=(),
        protocol=protocol,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )
    problems_dir = suite_root / "problems"
    problem_names = sorted(p.name for p in problems_dir.iterdir() if p.is_dir())
    _require(bool(problem_names), "suite declares no problems")
    problems = tuple(load_problem(holder, problem_name) for problem_name in problem_names)
    return UserBenchSuite(
        name=name,
        root=suite_root,
        suite=suite,
        problems=problems,
        protocol=protocol,
        manifest=manifest,
        manifest_sha256=holder.manifest_sha256,
    )
