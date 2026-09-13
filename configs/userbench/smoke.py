"""user-bench GPU smoke: both example problems, correctness + timing each.

Parent-side runner (trusted): loads the frozen suite with the torch-free
loader, stages problem.yaml + reference.py + an example candidate into a
read-only /task mount and runs configs/userbench/userbench_driver.py inside
the pinned evaluation container. The candidate never judges itself: the
verdict is computed by the trusted driver against the reference output.

Track: userbench-correctness-v1 + userbench-timing-v1 (suite examples_v1).
Usage: .venv/bin/python configs/userbench/smoke.py [--workload w1_base] \
          [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from kernelagent.adapters.benchmarks.userbench import load_suite
from kernelagent.adapters.evals.kernelbench_eval import EVAL_IMAGE_ID, EVAL_IMAGE_REPO
from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

REPO = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO / "configs/userbench/suites/examples_v1"
DRIVER = REPO / "configs/userbench/userbench_driver.py"

EVAL_CONSTANTS = {
    "memory_bytes": 8 * 1024**3,
    "tmp_tmpfs_bytes": 4 * 1024**3,
    "output_limit_bytes": 8 * 1024**3,
}


def _run_problem(problem, suite, workload_id: str, gpu_device: str, timeout: float) -> dict:
    workload = next((w for w in problem.workloads if w.workload_id == workload_id), None)
    if workload is None:
        return {"status": "infra_error", "notes": [f"unknown workload {workload_id!r}"]}
    workspace = Path(tempfile.mkdtemp(prefix="userbench-smoke-"))
    task_dir = workspace / "task-in"
    task_dir.mkdir(parents=True)
    (task_dir / "problem.yaml").write_text(
        (suite.root / problem.directory / "problem.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (task_dir / "reference.py").write_text(
        (suite.root / problem.directory / "reference.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (task_dir / "candidate.py").write_text(
        (suite.root / problem.directory / "examples/candidate.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (task_dir / "userbench_driver.py").write_text(
        DRIVER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    identity = suite.protocol_identity()
    case = {
        "task_id": problem.task_id,
        "reference_entry": problem.reference_entry,
        "workload": {
            "workload_id": workload.workload_id,
            "shapes": [list(shape) for shape in workload.shapes],
            "dtypes": list(workload.dtypes),
            "seed": workload.seed,
        },
        "tolerance": {
            "rtol": problem.tolerance.rtol,
            "atol": problem.tolerance.atol,
            "equal_nan": problem.tolerance.equal_nan,
            "policy": problem.tolerance.policy,
        },
        "input_generation": {
            "distribution": problem.input_generation.distribution,
            "scale": problem.input_generation.scale,
            "offset": problem.input_generation.offset,
        },
        "timing_params": suite.protocol.get("timing_params", {}),
        "protocol": identity,
    }
    (task_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

    request = WorkerRequest(
        request_id=f"userbench-smoke-{problem.task_id}"[:64],
        argv=("python3", "/task/userbench_driver.py"),
        timeout_seconds=timeout,
        workspace_root=workspace,
    )
    spec = ContainerSpec(
        image=EVAL_IMAGE_REPO,
        image_id=EVAL_IMAGE_ID,
        memory_bytes=EVAL_CONSTANTS["memory_bytes"],
        tmp_tmpfs_bytes=EVAL_CONSTANTS["tmp_tmpfs_bytes"],
        output_limit_bytes=EVAL_CONSTANTS["output_limit_bytes"],
        read_only_mounts=(("/task", task_dir),),
        gpu_devices=(gpu_device,),
    )
    started = time.monotonic()
    outcome = execute_container(request, spec)
    elapsed = time.monotonic() - started
    payload = None
    result_file = Path(outcome.workdir) / "out" / "result.json" if outcome.workdir else None
    if result_file and result_file.is_file():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    return {
        "container_status": outcome.status,
        "wall_seconds": round(elapsed, 1),
        "result": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="w1_base")
    parser.add_argument("--gpu-device", default="nvidia.com/gpu=0")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    suite = load_suite(SUITE_ROOT)
    identity = suite.protocol_identity()
    started = time.monotonic()
    results = {}
    for problem in suite.problems:
        results[problem.task_id] = _run_problem(
            problem, suite, args.workload, args.gpu_device, args.timeout
        )
    elapsed = time.monotonic() - started
    passed = all(
        (entry.get("result") or {}).get("status") == "passed" for entry in results.values()
    )
    summary = {
        "track": {"correctness": identity["correctness"], "timing": identity["timing"]},
        "protocol_identity": identity["identity_sha256"],
        "workload": args.workload,
        "wall_seconds": round(elapsed, 1),
        "problems": results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(
        f"userbench smoke: {'PASS' if passed else 'NOT-PASSED'} in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
