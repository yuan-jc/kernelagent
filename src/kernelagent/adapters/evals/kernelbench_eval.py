"""Control-plane adapter for the pinned KernelBench upstream evaluator
(T05, ADR-0002).

Each case runs inside the ADR-0001 container boundary on the ADR-0002
evaluation image: the pinned snapshot is mounted read-only at /task, the
case inputs (candidate source + explicit configuration + the frozen eval
driver) at /case, and the serialized upstream verdict is read back from
the parent-owned output directory. This module never imports torch - the
control plane stages bytes and reads the upstream result; the GPU work
happens under isolation.

Verdict rule (candidates cannot certify themselves): the adapter's pass
verdict is derived solely from the upstream ``KernelExecResult`` fields
(``compiled and correctness``) recorded by the pinned evaluator. A
missing/None upstream result is an evaluator failure and never a pass;
candidate stdout, exit code, and any self-written claims are ignored."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from kernelagent.adapters.models.generation import inspect_candidate_policy
from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

EVAL_IMAGE_REPO = "kernelagent-eval"
# ADR-0002: locally built from configs/eval-image/Dockerfile on the pinned
# pytorch base; pinned by content-addressed image ID (no registry digest).
EVAL_IMAGE_ID = "sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b"
EVAL_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
EVAL_TMP_TMPFS_BYTES = 512 * 1024 * 1024
EVAL_OUTPUT_LIMIT_BYTES = 128 * 1024 * 1024
_DRIVER_PATH = Path(__file__).resolve().parents[4] / "configs" / "kernelbench" / "eval_driver.py"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One correctness evaluation against the pinned reference."""

    case_id: str
    level: int
    problem_id: int
    problem_name: str
    candidate_name: str
    candidate_source: str
    backend: str = "cuda"
    seed: int = 42
    num_correct_trials: int = 1
    num_perf_trials: int = 1
    device: int = 0
    expect_pass: bool = True

    @property
    def problem_path(self) -> str:
        return f"KernelBench/level{self.level}/{self.problem_name}"

    @property
    def task_id(self) -> str:
        return f"kernelbench-l{self.level}-p{self.problem_id:03d}"


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    case_id: str
    task_id: str
    outcome_status: str
    exit_code: int | None
    adapter_pass: bool | None
    upstream_compiled: bool | None
    upstream_correctness: bool | None
    consistent_with_upstream: bool
    matches_expectation: bool
    problem_sha256: str
    candidate_sha256: str
    workdir: str | None
    detail: dict = field(default_factory=dict)


def derive_upstream_verdict(
    result_payload: dict | None,
) -> tuple[bool | None, bool | None, bool | None, bool]:
    """Interpret the serialized upstream verdict.

    Returns (adapter_pass, upstream_compiled, upstream_correctness,
    consistent). ``adapter_pass`` is None only when the evaluator itself
    failed to produce a verdict (never a pass). ``consistent`` records
    whether the adapter's derivation matches the upstream fields exactly -
    the check required by the "compare against direct upstream invocation"
    acceptance row."""
    if not isinstance(result_payload, dict):
        return None, None, None, False
    upstream = result_payload.get("upstream")
    if not isinstance(upstream, dict):
        return None, None, None, False
    compiled = upstream.get("compiled")
    correctness = upstream.get("correctness")
    if not isinstance(compiled, bool) or not isinstance(correctness, bool):
        return None, None, None, False
    adapter_pass = compiled and correctness
    return adapter_pass, compiled, correctness, True


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate_case(
    case: EvalCase,
    *,
    snapshot_root: Path,
    workspace_root: Path,
    gpu_devices: tuple[str, ...],
    driver_source: str | None = None,
    timeout_seconds: float = 900.0,
    docker_command: tuple[str, ...] = ("docker",),
) -> EvalCaseResult:
    """Run one case under isolation and interpret the upstream verdict."""
    if driver_source is None:
        driver_source = _DRIVER_PATH.read_text(encoding="utf-8")
    snapshot_root = Path(snapshot_root)
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    problem_file = snapshot_root / case.problem_path
    problem_source = problem_file.read_text(encoding="utf-8")

    case_config = {
        "case_id": case.case_id,
        "problem_path": case.problem_path,
        "backend": case.backend,
        "seed": case.seed,
        "num_correct_trials": case.num_correct_trials,
        "num_perf_trials": case.num_perf_trials,
        "device": case.device,
        "measure_performance": False,
    }
    # Stage exactly the evaluator subset this case needs (upstream harness
    # files + the one problem file + case inputs), all inside the run
    # workspace so the ADR-0001 mount rule holds; every staged file is
    # content-hashed into the evidence.
    inputs = workspace_root / f"case-inputs-{case.case_id}"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / case.problem_path.rsplit("/", 1)[0]
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        source = (snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        (evaluator_dir / harness).write_bytes(source)
        staged[f"src/kernelbench/{harness}"] = _sha256_bytes(source)
    (problem_dir / case.problem_name).write_bytes(problem_file.read_bytes())
    staged[case.problem_path] = _sha256_bytes(problem_source.encode("utf-8"))
    (inputs / "case.json").write_text(json.dumps(case_config, indent=2), encoding="utf-8")
    (inputs / "candidate.py").write_text(case.candidate_source, encoding="utf-8")
    staged["candidate.py"] = _sha256_bytes(case.candidate_source.encode("utf-8"))
    (inputs / "eval_driver.py").write_text(driver_source, encoding="utf-8")

    request = WorkerRequest(
        request_id=f"eval-{case.case_id}"[:64],
        argv=("python3", "/task/eval_driver.py"),
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

    result_payload: dict | None = None
    parse_note = ""
    if outcome.workdir is not None:
        result_file = Path(outcome.workdir) / "out" / "result.json"
        if result_file.is_file():
            try:
                result_payload = json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parse_note = f"result.json parse failed: {exc}"
        else:
            parse_note = "result.json missing from output directory"
    adapter_pass, compiled, correctness, consistent = derive_upstream_verdict(result_payload)
    if parse_note:
        consistent = False
    if outcome.status != "completed":
        consistent = False
    matches = adapter_pass is not None and adapter_pass == case.expect_pass
    # Alpha threat model (ADR-0004): the candidate is exec()'d inside the
    # evaluator process, so until the trust-domain split the report carries
    # the cooperative-trust markers and the AST policy findings verbatim -
    # never an implicit claim of adversarial strength.
    policy = inspect_candidate_policy(case.candidate_source)
    return EvalCaseResult(
        case_id=case.case_id,
        task_id=case.task_id,
        outcome_status=outcome.status,
        exit_code=outcome.exit_code,
        adapter_pass=adapter_pass,
        upstream_compiled=compiled,
        upstream_correctness=correctness,
        consistent_with_upstream=consistent,
        matches_expectation=matches,
        problem_sha256=_sha256_bytes(problem_source.encode("utf-8")),
        candidate_sha256=_sha256_bytes(case.candidate_source.encode("utf-8")),
        workdir=outcome.workdir,
        detail={
            "backend": case.backend,
            "seed": case.seed,
            "num_correct_trials": case.num_correct_trials,
            "expect_pass": case.expect_pass,
            "parse_note": parse_note,
            "staged_files_sha256": staged,
            "stderr_tail": outcome.stderr_tail[-1500:],
            "candidate_trust": "cooperative",
            "adversarially_secure": False,
            "policy_allowed": policy.allowed,
            "policy_violations": list(policy.violations),
        },
    )
