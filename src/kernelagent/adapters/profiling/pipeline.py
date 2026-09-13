"""Baseline NCU profiling pipeline (profile-loop wiring, B3).

Produces real NCU evidence for a pinned problem's BASELINE reference and
turns it into the existing ``evidence_view`` structure consumed by the
method generator (``GeneratorInput.ncu_view``). Honest semantics follow
``adapters/profiling/ncu.py`` and ADR-0003:

- Scope: only parent-authored, fixed inputs enter the ADR-0003 diagnostic
  lease - the pinned problem file (sha256-tracked), the frozen kernelbench
  loader helpers, and the versioned driver below. Candidates are never
  profiled here (ADR-0001 is untouched); NCU evidence characterizes the
  baseline only, and stays strictly separated from T07 formal timing
  (``source="ncu_profile"``, never a batch sample).
- Controllable capture: ``--set basic`` or an explicit metric list, a
  kernel-launch cap (``--launch-count``), skip offset and a hard wall-clock
  timeout with container cleanup. Every knob is recorded in the journal
  event and in the record's ``profile`` field.
- Honesty: anything that prevents collection yields ``status="not_run"``
  with an explicit reason (missing host ncu, counter-permission blocker,
  timeout, driver failure, report not produced). Missing metrics are
  absent from the evidence view, never zero-filled; the measured GPU wall
  time is reported and billed, never silently dropped.

The outcome of :func:`profile_baseline` is a plain JSON-serializable dict
so the optimization loop can journal it verbatim."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernelagent.adapters.profiling.ncu import (
    MetricCatalog,
    detect_profiling_blocker,
    evidence_view,
    import_report,
)

DEFAULT_TOOLKIT_ROOT = Path("/home/y/toolchains/cuda")
DEFAULT_EVAL_IMAGE = "kernelagent-eval:cuda124-pytorch251-t05"
DEFAULT_NCU_CONTAINER_PATH = "/cuda/bin/ncu"
PROFILE_DIR_NAME = "profile"
BASELINE_REPORT_NAME = "baseline.ncu-rep"
BASELINE_EVIDENCE_NAME = "baseline-evidence.json"

# Host-side import needs an ncu binary (no GPU / no elevation needed).
_DEFAULT_NCU_CANDIDATES = (
    str(DEFAULT_TOOLKIT_ROOT / "bin" / "ncu"),
    "ncu",
)


def host_ncu_bin(preferred: str | None = None) -> str | None:
    """Resolve the host ``ncu`` used for ``--import`` (interpretation half).
    ``None`` = honestly unavailable: collection is skipped, never faked."""
    if preferred:
        if Path(preferred).is_file():
            return preferred
        return shutil.which(preferred)
    for candidate in _DEFAULT_NCU_CANDIDATES:
        if candidate.startswith("/"):
            if Path(candidate).exists():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


# Tier-A metric list mirroring MetricCatalog.default() (generators/classify.py
# CORE_TIER_A_METRICS is kept in sync by tests). Used when ``metrics`` is
# requested explicitly instead of a named set.
DEFAULT_METRIC_LIST: tuple[str, ...] = tuple(MetricCatalog.default().entries)


def driver_sha256() -> str:
    """Content identity of the parent-authored driver (journal evidence)."""
    return hashlib.sha256(PROFILE_DRIVER_SOURCE.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfileCapture:
    """Controllable NCU capture parameters; every field is recorded.

    Default mode is the explicit Tier-A metric list (mirrors
    ``MetricCatalog.default()``): it is deterministic against the catalog
    and on the pinned target (RTX 4060, CC 8.9) collects every Tier-A
    signal, so classification reaches ``ncu_full``. ``--set basic`` covers
    less (DRAM throughput is not in the basic set there) and yields an
    honest ``ncu_partial`` instead."""

    set_name: str | None = None
    metrics: tuple[str, ...] | None = DEFAULT_METRIC_LIST
    launch_count: int = 12
    launch_skip: int = 0
    timeout_seconds: float = 300.0
    warmup_iters: int = 3
    measured_iters: int = 5
    seed: int = 42
    device: int = 0
    image: str = DEFAULT_EVAL_IMAGE
    toolkit_root: Path = DEFAULT_TOOLKIT_ROOT
    ncu_container_path: str = DEFAULT_NCU_CONTAINER_PATH
    ncu_bin: str | None = None  # host binary for --import; None = auto
    docker_command: tuple[str, ...] = ("docker",)

    def __post_init__(self) -> None:
        if (self.set_name is None) == (self.metrics is None):
            raise ValueError(
                "ProfileCapture needs exactly one of set_name or metrics; got "
                f"set_name={self.set_name!r} metrics={self.metrics!r}"
            )
        if isinstance(self.launch_count, bool) or self.launch_count < 1:
            raise ValueError(f"launch_count must be an integer >= 1; got {self.launch_count!r}")
        if isinstance(self.launch_skip, bool) or self.launch_skip < 0:
            raise ValueError(f"launch_skip must be an integer >= 0; got {self.launch_skip!r}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0; got {self.timeout_seconds!r}")

    def to_dict(self) -> dict:
        return {
            "mode": "metrics" if self.metrics is not None else "set",
            "set_name": self.set_name,
            "metrics": list(self.metrics) if self.metrics is not None else None,
            "launch_count": self.launch_count,
            "launch_skip": self.launch_skip,
            "timeout_seconds": self.timeout_seconds,
            "warmup_iters": self.warmup_iters,
            "measured_iters": self.measured_iters,
            "seed": self.seed,
            "device": self.device,
            "image": self.image,
        }

    def ncu_arguments(self) -> list[str]:
        args = ["--target-processes", "all", "-f", "-o", "/out/baseline"]
        if self.metrics is not None:
            args += ["--metrics", ",".join(self.metrics)]
        else:
            args += ["--set", str(self.set_name)]
        if self.launch_skip:
            args += ["--launch-skip", str(self.launch_skip)]
        args += ["--launch-count", str(self.launch_count)]
        return args


# Parent-authored baseline driver: runs INSIDE the privileged ADR-0003
# diagnostic container under NCU. It only executes the pinned problem's
# reference model - it measures nothing and reports no numbers; all kernel
# evidence comes from NCU itself. Versioned by content hash (driver_sha256).
PROFILE_DRIVER_SOURCE = '''"""Baseline profile driver (diagnostic lease, ADR-0003).

Runs the pinned problem's reference model only: warmup, then a fixed
number of measured forward calls. Produces no timing claims - kernel
metrics come from NCU, never from this process. Output: driver_meta.json
(an execution marker, not evidence)."""
import json
import os
import sys

TASK_ROOT = "/task"


def main() -> int:
    os.environ.setdefault("HOME", "/tmp")
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_cache"
    sys.path.insert(0, f"{TASK_ROOT}/src")
    import torch
    from kernelbench.eval import load_original_model_and_inputs

    case = json.load(open(f"{TASK_ROOT}/case.json"))
    problem_src = open(f"{TASK_ROOT}/{case['problem_path']}").read()
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})
    device = torch.device(f"cuda:{case['device']}")
    torch.manual_seed(case["seed"])
    model = Model(*get_init_inputs()).to(device)
    inputs = [t.to(device) for t in get_inputs()]
    with torch.no_grad():
        for _ in range(case["warmup_iters"]):
            model(*inputs)
        torch.cuda.synchronize()
        for _ in range(case["measured_iters"]):
            model(*inputs)
        torch.cuda.synchronize()
    with open("/out/driver_meta.json", "w") as handle:
        json.dump(
            {
                "driver": "baseline-profile-v1",
                "problem_path": case["problem_path"],
                "device": str(device),
                "torch_version": torch.__version__,
                "warmup_iters": case["warmup_iters"],
                "measured_iters": case["measured_iters"],
            },
            handle,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_EVAL_HELPERS = ("dataset.py", "eval.py", "timing.py", "utils.py")


def stage_profile_inputs(
    workspace_root: Path,
    *,
    snapshot_root: Path,
    level: int,
    problem_name: str,
    problem_source: str,
    capture: ProfileCapture,
) -> Path:
    """Stage the parent-authored, fixed inputs under ``<workspace>/profile-inputs``
    (mounted read-only at /task): frozen loader helpers, the pinned problem
    file, the versioned driver and case.json. Nothing candidate-generated
    ever enters this directory."""
    inputs = Path(workspace_root) / "profile-inputs"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / f"KernelBench/level{level}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(snapshot_root)
    for harness in _EVAL_HELPERS:
        (evaluator_dir / harness).write_bytes(
            (snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        )
    (problem_dir / problem_name).write_text(problem_source, encoding="utf-8")
    (inputs / "profile_driver.py").write_text(PROFILE_DRIVER_SOURCE, encoding="utf-8")
    (inputs / "case.json").write_text(
        json.dumps(
            {
                "problem_path": f"KernelBench/level{level}/{problem_name}",
                "device": capture.device,
                "seed": capture.seed,
                "warmup_iters": capture.warmup_iters,
                "measured_iters": capture.measured_iters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return inputs


def _not_run(
    reason: str,
    detail: str,
    capture: ProfileCapture,
    *,
    gpu_wall_seconds: float | None = None,
    stderr_tail: str = "",
) -> dict:
    return {
        "status": "not_run",
        "reason": reason,
        "detail": detail[:1200],
        "capture": capture.to_dict(),
        "driver_sha256": driver_sha256(),
        "gpu_wall_seconds": gpu_wall_seconds,
        "stderr_tail": stderr_tail[-1200:],
        "report_path": None,
        "report_sha256": None,
        "evidence_path": None,
        "evidence_view": None,
        "summary": None,
    }


def profile_baseline(
    *,
    profile_dir: Path,
    workspace_root: Path,
    snapshot_root: Path,
    level: int,
    problem_name: str,
    problem_source: str,
    gpu_device: str,
    capture: ProfileCapture | None = None,
    problem_sha256: str | None = None,
) -> dict:
    """Collect real NCU evidence for the pinned baseline reference.

    Returns a JSON-serializable outcome dict with ``status`` of either
    ``collected`` (report + evidence files written under ``profile_dir``)
    or ``not_run`` (explicit reason; no fabricated data). Never raises for
    expected environment failures - the optimization loop must survive an
    unusable NCU and record the degradation instead."""
    capture = capture or ProfileCapture()
    profile_dir = Path(profile_dir)
    ncu_bin = host_ncu_bin(capture.ncu_bin)
    if ncu_bin is None:
        return _not_run(
            "ncu_unavailable_on_host",
            "no ncu binary found for the host-side --import step; collection "
            "would be unverifiable, so nothing was profiled",
            capture,
        )
    try:
        inputs = stage_profile_inputs(
            workspace_root,
            snapshot_root=snapshot_root,
            level=level,
            problem_name=problem_name,
            problem_source=problem_source,
            capture=capture,
        )
    except OSError as exc:
        return _not_run("staging_failed", str(exc), capture)

    profile_dir.mkdir(parents=True, exist_ok=True)
    workdir = profile_dir / "scratch"
    workdir.mkdir(parents=True, exist_ok=True)
    script = (
        f"cd /out && {capture.ncu_container_path} "
        + " ".join(capture.ncu_arguments())
        + " python3 /task/profile_driver.py\n"
        "echo ncu_rc=$?\n"
    )
    container_name = f"kernelagent-prof-{uuid.uuid4().hex[:12]}"
    command = [
        *capture.docker_command,
        "run",
        "--rm",
        "--name",
        container_name,
        "--privileged",
        "--network",
        "none",
        "--device",
        gpu_device,
        "-v",
        f"{Path(capture.toolkit_root).resolve()}:/cuda:ro",
        "-v",
        f"{inputs.resolve()}:/task:ro",
        "-v",
        f"{workdir.resolve()}:/out",
        capture.image,
        "bash",
        "-c",
        script,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=capture.timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired:
        _force_remove_container(capture.docker_command, container_name)
        return _not_run(
            "timeout",
            f"ncu container exceeded timeout_seconds={capture.timeout_seconds}; "
            "container was force-removed",
            capture,
            gpu_wall_seconds=time.monotonic() - started,
        )
    except OSError as exc:
        return _not_run(
            "docker_unavailable",
            f"failed to launch the diagnostic container: {exc}",
            capture,
            gpu_wall_seconds=time.monotonic() - started,
        )
    wall = time.monotonic() - started
    combined = completed.stdout + "\n" + completed.stderr

    driver_marker = workdir / "driver_meta.json"
    if not driver_marker.is_file():
        blocker = detect_profiling_blocker(combined)
        return _not_run(
            blocker or "driver_failed",
            f"baseline driver did not complete inside the diagnostic container "
            f"(rc={completed.returncode})",
            capture,
            gpu_wall_seconds=wall,
            stderr_tail=combined,
        )

    raw_report = workdir / "baseline.ncu-rep"
    if not raw_report.is_file():
        blocker = detect_profiling_blocker(combined)
        return _not_run(
            blocker or "report_not_produced",
            f"ncu produced no report (rc={completed.returncode})",
            capture,
            gpu_wall_seconds=wall,
            stderr_tail=combined,
        )

    report_path = profile_dir / BASELINE_REPORT_NAME
    shutil.move(str(raw_report), report_path)
    try:
        launches = import_report(ncu_bin, report_path, timeout=120.0)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired, OSError) as exc:
        return _not_run(
            "import_failed",
            f"host-side report import failed: {exc}",
            capture,
            gpu_wall_seconds=wall,
            stderr_tail=combined,
        )
    if not launches:
        return _not_run(
            "no_kernels_profiled",
            "report parsed but contains zero launches",
            capture,
            gpu_wall_seconds=wall,
            stderr_tail=combined,
        )
    view = evidence_view(launches, MetricCatalog.default())
    evidence_path = profile_dir / BASELINE_EVIDENCE_NAME
    from kernelagent.adapters.profiling.ncu import to_json

    evidence_path.write_text(to_json(view), encoding="utf-8")
    return {
        "status": "collected",
        "reason": None,
        "detail": "",
        "capture": capture.to_dict(),
        "driver_sha256": driver_sha256(),
        "gpu_wall_seconds": wall,
        "stderr_tail": combined[-1200:],
        "report_path": report_path.name,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "evidence_path": evidence_path.name,
        "evidence_view": view,
        "summary": summarize_evidence(view),
        "problem_sha256": problem_sha256,
        "ncu_bin": ncu_bin,
        "source": "ncu_profile",
    }


def _force_remove_container(docker_command: tuple[str, ...], name: str) -> None:
    try:
        subprocess.run(
            [*docker_command, "rm", "-f", name],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - best effort cleanup
        pass


def summarize_evidence(view: dict) -> dict:
    """Compact, frontend-friendly summary of an ``evidence_view``. A metric
    that was not collected is ``None`` - never zero."""
    launches = view.get("launches") or []

    def max_value(metric: str):
        values = []
        for launch in launches:
            entry = (launch.get("metrics") or {}).get(metric)
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
                if not isinstance(entry["value"], bool):
                    values.append(float(entry["value"]))
        return max(values) if values else None

    return {
        "launch_count": len(launches),
        "kernels": sorted({str(launch.get("kernel_name")) for launch in launches}),
        "gpu_time_us_max": max_value("gpu__time_duration.sum"),
        "dram_throughput_pct_max": max_value("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_throughput_pct_max": max_value("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "compute_memory_throughput_pct_max": max_value(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        ),
        "warps_active_pct_max": max_value("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "registers_per_thread_max": max_value("launch__registers_per_thread"),
        "grid_size_max": max_value("launch__grid_size"),
        "metrics_missing_in_all_launches": list(
            view.get("catalog_metrics_missing_in_all_launches") or ()
        ),
    }


def load_collected_evidence(profile_dir: Path) -> dict | None:
    """Previously collected evidence view for this run, or ``None``."""
    path = Path(profile_dir) / BASELINE_EVIDENCE_NAME
    if not path.is_file():
        return None
    try:
        view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return view if isinstance(view, dict) and view.get("launches") else None


def build_baseline_profiler(
    *,
    profile_dir: Path,
    workspace_root: Path,
    snapshot_root: Path,
    level: int,
    problem_name: str,
    problem_source: str,
    gpu_device: str,
    capture: ProfileCapture | None = None,
    problem_sha256: str | None = None,
):
    """Bind one baseline profile execution for the optimization loop / CLI.

    The returned callable returns the outcome dict of
    :func:`profile_baseline` and is injectable in tests (the loop never
    builds it when offline fakes replace the real timing port)."""

    def run() -> dict:
        return profile_baseline(
            profile_dir=profile_dir,
            workspace_root=workspace_root,
            snapshot_root=snapshot_root,
            level=level,
            problem_name=problem_name,
            problem_source=problem_source,
            gpu_device=gpu_device,
            capture=capture,
            problem_sha256=problem_sha256,
        )

    return run
