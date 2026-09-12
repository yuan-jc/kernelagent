"""Control-plane timing adapter (T07, design §9.3).

Serializes TimingProtocol v1, runs one timing role per case inside the
ADR-0001/0002 container boundary, and reads back the raw batch samples.
All statistics are computed here from raw batches; the container never
reports a summary the parent must trust. Timing results are only valid
with ``source == 'cuda_event'``: profile-derived numbers (T15) are
rejected by :func:`validate_timing_payload`, and a candidate's own
exit/stdout claims never enter the report."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from kernelagent.adapters.evals.kernelbench_eval import (
    EVAL_IMAGE_ID,
    EVAL_IMAGE_REPO,
    EVAL_MEMORY_BYTES,
    EVAL_OUTPUT_LIMIT_BYTES,
    EVAL_TMP_TMPFS_BYTES,
    _sha256_bytes,
)
from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

SOURCE_TAG = "cuda_event"
_TIMING_DRIVER_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "kernelbench" / "timing_driver.py"
)


@dataclass(frozen=True, slots=True)
class TimingProtocol:
    """TimingProtocol v1: every field is part of the run identity."""

    warmup_iters: int = 10
    num_batches: int = 12
    iters_per_batch: int = 20
    input_rotation: str = "fixed"
    stream_policy: str = "current_stream_events_per_batch_device_sync"
    cache_state: str = "warm"
    statistics: str = "raw_batch_samples_no_outlier_removal"
    outlier_rule: str = "none"
    device: int = 0
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "protocol_version": "timing-v1",
            "warmup_iters": self.warmup_iters,
            "num_batches": self.num_batches,
            "iters_per_batch": self.iters_per_batch,
            "input_rotation": self.input_rotation,
            "stream_policy": self.stream_policy,
            "cache_state": self.cache_state,
            "statistics": self.statistics,
            "outlier_rule": self.outlier_rule,
            "device": self.device,
            "seed": self.seed,
        }

    def identity_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return _sha256_bytes(canonical.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class TimingCase:
    case_id: str
    level: int
    problem_id: int
    problem_name: str
    candidate_name: str
    candidate_source: str
    role: str  # eager | compile | candidate
    backend: str = "cuda"
    expect_async_leak: bool = False


@dataclass(frozen=True, slots=True)
class TimingCaseResult:
    case_id: str
    outcome_status: str
    source_ok: bool
    protocol_hash_ok: bool
    correctness_precondition: bool
    batch_samples_ms: tuple[float, ...]
    single_shot_sync_ms: float | None
    async_leak: bool
    async_leak_matches_expectation: bool
    aa_ratio_ci: tuple[float, float] | None
    workdir: str | None
    detail: dict


def validate_timing_payload(payload: dict, expected_protocol_hash: str) -> tuple[bool, str]:
    """Gate for any object claiming to be a formal timing result. Profile
    sources are rejected outright (NCU time must not enter formal
    results); a protocol identity mismatch is a hard failure."""
    if payload.get("source") != SOURCE_TAG:
        return False, f"source must be {SOURCE_TAG!r}; got {payload.get('source')!r}"
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("protocol_version") != "timing-v1":
        return False, "protocol identity missing or wrong version"
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != expected_protocol_hash:
        return False, "protocol hash does not match the requested protocol"
    samples = payload.get("batch_samples_ms")
    if not isinstance(samples, list) or not samples:
        return False, "raw batch samples missing"
    return True, ""


def bootstrap_ratio_ci(
    baseline: list[float],
    candidate: list[float],
    *,
    seed: int = 0,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(baseline)/mean(candidate); the
    resampling unit is the independent batch, never the pooled iteration."""
    if not baseline or not candidate:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = min(len(baseline), len(candidate))
    ratios = []
    for _ in range(iterations):
        b = [baseline[rng.randrange(len(baseline))] for _ in range(n)]
        c = [candidate[rng.randrange(len(candidate))] for _ in range(n)]
        mean_c = sum(c) / n
        if mean_c > 0:
            ratios.append(sum(b) / n / mean_c)
    ratios.sort()
    alpha = (1.0 - confidence) / 2.0
    low = ratios[int(len(ratios) * alpha)]
    high = ratios[min(len(ratios) - 1, int(len(ratios) * (1.0 - alpha)))]
    return (low, high)


def run_timing_case(
    case: TimingCase,
    protocol: TimingProtocol,
    *,
    snapshot_root: Path,
    workspace_root: Path,
    gpu_devices: tuple[str, ...],
    timeout_seconds: float = 1200.0,
    docker_command: tuple[str, ...] = ("docker",),
) -> TimingCaseResult:
    snapshot_root = Path(snapshot_root)
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    driver_source = _TIMING_DRIVER_PATH.read_text(encoding="utf-8")

    problem_file = snapshot_root / f"KernelBench/level{case.level}/{case.problem_name}"
    problem_source = problem_file.read_text(encoding="utf-8")

    case_config = {
        "role": case.role,
        "backend": case.backend,
        "problem_path": f"KernelBench/level{case.level}/{case.problem_name}",
        "device": protocol.device,
        "seed": protocol.seed,
        "protocol": protocol.to_dict(),
        "integrity_check": True,
    }
    inputs = workspace_root / f"timing-inputs-{case.case_id}"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / f"KernelBench/level{case.level}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (evaluator_dir / harness).write_bytes(
            (snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        )
    (problem_dir / case.problem_name).write_bytes(problem_file.read_bytes())
    (inputs / "candidate.py").write_text(case.candidate_source, encoding="utf-8")
    (inputs / "case.json").write_text(json.dumps(case_config, indent=2), encoding="utf-8")
    (inputs / "timing_driver.py").write_text(driver_source, encoding="utf-8")

    request = WorkerRequest(
        request_id=f"timing-{case.case_id}"[:64],
        argv=("python3", "/task/timing_driver.py"),
        timeout_seconds=timeout_seconds,
        workspace_root=workspace_root,
    )
    spec = ContainerSpec(
        image=EVAL_IMAGE_REPO,
        image_id=EVAL_IMAGE_ID,
        memory_bytes=EVAL_MEMORY_BYTES,
        tmp_tmpfs_bytes=EVAL_TMP_TMPFS_BYTES,
        output_limit_bytes=EVAL_OUTPUT_LIMIT_BYTES,
        read_only_mounts=(("/task", inputs),),
        gpu_devices=gpu_devices,
    )
    outcome = execute_container(request, spec, docker_command=docker_command)

    payload = None
    parse_note = ""
    if outcome.workdir is not None:
        result_file = Path(outcome.workdir) / "out" / "timing.json"
        if result_file.is_file():
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parse_note = f"timing.json parse failed: {exc}"
        else:
            parse_note = "timing.json missing from output directory"

    source_ok, source_note = (False, "payload missing")
    protocol_hash_ok = False
    samples: tuple[float, ...] = ()
    precondition = False
    single_shot = None
    async_leak = False
    if isinstance(payload, dict):
        source_ok, source_note = validate_timing_payload(payload, protocol.identity_sha256())
        protocol_hash_ok = source_ok or "protocol" not in source_note
        samples = tuple(payload.get("batch_samples_ms") or ())
        precondition = bool(payload.get("correctness_precondition"))
        single_shot = payload.get("single_shot_sync_ms")
        async_leak = bool(payload.get("async_leak"))
    return TimingCaseResult(
        case_id=case.case_id,
        outcome_status=outcome.status,
        source_ok=source_ok,
        protocol_hash_ok=protocol_hash_ok,
        correctness_precondition=precondition,
        batch_samples_ms=samples,
        single_shot_sync_ms=single_shot,
        async_leak=async_leak,
        async_leak_matches_expectation=async_leak == case.expect_async_leak,
        aa_ratio_ci=None,
        workdir=outcome.workdir,
        detail={
            "source_note": source_note,
            "parse_note": parse_note,
            "role": case.role,
            "backend": case.backend,
            "problem_sha256": _sha256_bytes(problem_source.encode("utf-8")),
            "stderr_tail": outcome.stderr_tail[-1200:],
        },
    )
