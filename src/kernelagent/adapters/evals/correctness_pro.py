"""Control-plane adapter for the extended correctness track (T06).

Stages the same frozen staging as T05/T07, runs the correctness-pro
driver inside the ADR-0001 boundary, and interprets the per-check
report. Semantics pinned by tests/test_correctness_pro.py:
- the overall verdict is False unless every core check passed AND the
  sanitizer status is exactly ``pass``;
- ``not_run`` sanitizer is a coverage gap, never a pass;
- the upstream evaluator's verdict (T05) is passed through unchanged -
  the extended track never rewrites it."""

import json
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

SANITIZER_PASS = "pass"
SANITIZER_FAIL = "fail"
SANITIZER_NOT_RUN = "not_run"
_PRO_DRIVER_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "kernelbench" / "correctness_pro_driver.py"
)
_CORE_CHECKS = (
    "baseline",
    "shape_dtype",
    "input_mutation",
    "constant_output",
    "tail_check",
    "state_check",
)


def overall_verdict(pro_result: dict | None) -> tuple[bool, str]:
    """Derive the extended-track verdict; missing payload is never a pass."""
    if not isinstance(pro_result, dict):
        return False, "pro result missing"
    checks = pro_result.get("checks")
    if not isinstance(checks, dict):
        return False, "checks section missing"
    for name in _CORE_CHECKS:
        entry = checks.get(name)
        if not isinstance(entry, dict) or entry.get("passed") is not True:
            return False, f"core check {name!r} did not pass"
    sanitizer = checks.get("sanitizer")
    if not isinstance(sanitizer, dict):
        return False, "sanitizer section missing"
    status = sanitizer.get("status")
    if status not in (SANITIZER_PASS, SANITIZER_FAIL, SANITIZER_NOT_RUN):
        return False, f"unknown sanitizer status {status!r}"
    if status != SANITIZER_PASS:
        return False, f"sanitizer status {status!r} is a coverage gap or failure, not a pass"
    return True, "all core checks passed and sanitizer passed"


def run_pro_case(
    *,
    case_id: str,
    level: int,
    problem_id: int,
    problem_name: str,
    candidate_name: str,
    candidate_source: str,
    snapshot_root: Path,
    workspace_root: Path,
    gpu_devices: tuple[str, ...],
    backend: str = "cuda",
    sanitizer_binary: str | None = None,
    timeout_seconds: float = 900.0,
    docker_command: tuple[str, ...] = ("docker",),
) -> dict:
    """Stage, run, and interpret one enhanced-correctness case."""
    snapshot_root = Path(snapshot_root)
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    driver_source = _PRO_DRIVER_PATH.read_text(encoding="utf-8")
    problem_rel = f"KernelBench/level{level}/{problem_name}"
    problem_file = snapshot_root / problem_rel

    case_config = {
        "problem_path": problem_rel,
        "device": 0,
        "seed": 42,
        "sanitizer_timeout": 240,
        "backend": backend,
    }
    if sanitizer_binary is not None:
        case_config["sanitizer_binary"] = sanitizer_binary

    inputs = workspace_root / f"pro-inputs-{case_id}"
    evaluator_dir = inputs / "src" / "kernelbench"
    problem_dir = inputs / f"KernelBench/level{level}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (evaluator_dir / harness).write_bytes(
            (snapshot_root / "src" / "kernelbench" / harness).read_bytes()
        )
    (problem_dir / problem_name).write_bytes(problem_file.read_bytes())
    (inputs / "candidate.py").write_text(candidate_source, encoding="utf-8")
    (inputs / "case.json").write_text(json.dumps(case_config, indent=2), encoding="utf-8")
    (inputs / "pro_driver.py").write_text(driver_source, encoding="utf-8")

    request = WorkerRequest(
        request_id=f"pro-{case_id}"[:64],
        argv=("python3", "/task/pro_driver.py"),
        timeout_seconds=timeout_seconds,
        workspace_root=workspace_root,
    )
    spec = ContainerSpec(
        image=EVAL_IMAGE_REPO,
        image_id=EVAL_IMAGE_ID,
        memory_bytes=EVAL_MEMORY_BYTES,
        # compute-sanitized torch processes need far more threads (CUDA
        # context + BLAS pools) than the default worker budget.
        pids_limit=256,
        tmp_tmpfs_bytes=EVAL_TMP_TMPFS_BYTES,
        output_limit_bytes=EVAL_OUTPUT_LIMIT_BYTES,
        read_only_mounts=(("/task", inputs),),
        gpu_devices=gpu_devices,
    )
    outcome = execute_container(request, spec, docker_command=docker_command)

    payload = None
    parse_note = ""
    if outcome.workdir is not None:
        result_file = Path(outcome.workdir) / "out" / "pro_result.json"
        if result_file.is_file():
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parse_note = f"pro_result.json parse failed: {exc}"
        else:
            parse_note = "pro_result.json missing from output directory"
    verdict, reason = overall_verdict(payload)
    if outcome.status != "completed":
        verdict, reason = False, f"container outcome {outcome.status!r}"
    elif parse_note:
        verdict, reason = False, parse_note
    return {
        "case_id": case_id,
        "task_id": f"kernelbench-l{level}-p{problem_id:03d}",
        "outcome_status": outcome.status,
        "verdict": verdict,
        "reason": reason,
        "checks": (payload or {}).get("checks", {}),
        "candidate_sha256": _sha256_bytes(candidate_source.encode("utf-8")),
        "workdir": outcome.workdir,
        "stderr_tail": outcome.stderr_tail[-1200:],
    }
