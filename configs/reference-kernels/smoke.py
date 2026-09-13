"""GPU-MODE reference-kernels GPU smoke: one problem, test + benchmark modes.

Parent-side runner (trusted): stages the pinned snapshot bytes for one
problem set into a read-only /task mount, runs the pinned upstream eval.py
verbatim via configs/reference-kernels/rk_driver.py inside the pinned
evaluation container and prints an honest summary.

Track: reference-kernels-local-replica-v1 - a local replication of the
upstream protocol on hardware the official leaderboard does not list.
Results are never comparable with official leaderboard numbers. An
infrastructure failure is reported as such, never as a candidate failure.

Usage: .venv/bin/python configs/reference-kernels/smoke.py \
          [--set pmpp_v2 --name vectorsum_v2] [--benchmark-specs 1] \
          [--candidate path/to/submission.py] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import tempfile
import time
from pathlib import Path

from kernelagent.adapters.benchmarks.reference_kernels import (
    ReferenceKernelsSnapshot,
    load_dev_manifest,
    load_files_manifest,
    problem_from_manifest,
    workload_specs,
)
from kernelagent.adapters.evals.kernelbench_eval import EVAL_IMAGE_ID, EVAL_IMAGE_REPO
from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = REPO / "research/sources/gpu-mode__reference-kernels"
FILES_MANIFEST = REPO / "configs/reference-kernels/snapshot-files.manifest.json"
DEV_MANIFEST = REPO / "configs/reference-kernels/dev-manifest.json"
DRIVER = REPO / "configs/reference-kernels/rk_driver.py"

EVAL_CONSTANTS = {
    "memory_bytes": 8 * 1024**3,
    "tmp_tmpfs_bytes": 4 * 1024**3,
    "output_limit_bytes": 8 * 1024**3,
    # the pinned submissions may use torch.compile: inductor compilation
    # subprocesses + CUDA threads exceed the default pids limit
    "pids_limit": 1024,
}


def _stage(
    snapshot: ReferenceKernelsSnapshot,
    problem,
    case: dict,
    driver_text: str,
    submission_override: str | None = None,
) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="rk-smoke-"))
    task_dir = workspace / "task-in"
    (task_dir / "work").mkdir(parents=True)
    for rel in ("LICENSE", f"problems/{problem.set_name}.yaml"):
        target = task_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(snapshot.read_member(rel), encoding="utf-8")
    for rel in problem.members:
        target = task_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(snapshot.read_member(rel), encoding="utf-8")
    # Materialize the pinned task.yml `files` mapping flat into the runtime
    # directory, exactly as the upstream harness assembles a submission:
    # "@SUBMISSION@" becomes the candidate submission.py; other sources are
    # resolved relative to the problem directory (e.g. "../utils.py").
    runtime = task_dir / "runtime"
    runtime.mkdir(parents=True)
    for entry in problem.task_yml["files"]:
        name, source = entry["name"], entry["source"]
        if source == "@SUBMISSION@":
            pinned = submission_override or f"problems/{problem.directory}/submission.py"
            content = snapshot.read_member(pinned)
        else:
            source_rel = posixpath.normpath(f"problems/{problem.directory}/{source}")
            content = snapshot.read_member(source_rel)
        (runtime / name).write_text(content, encoding="utf-8")
    (task_dir / "rk_driver.py").write_text(driver_text, encoding="utf-8")
    (task_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return workspace, task_dir


def _run(task_dir: Path, gpu_device: str, timeout: float) -> tuple[dict | None, str]:
    workspace = task_dir.parent
    request = WorkerRequest(
        request_id=f"rk-smoke-{time.monotonic_ns()}"[:64],
        argv=("python3", "/task/rk_driver.py"),
        timeout_seconds=timeout,
        workspace_root=workspace,
    )
    spec = ContainerSpec(
        image=EVAL_IMAGE_REPO,
        image_id=EVAL_IMAGE_ID,
        memory_bytes=EVAL_CONSTANTS["memory_bytes"],
        tmp_tmpfs_bytes=EVAL_CONSTANTS["tmp_tmpfs_bytes"],
        output_limit_bytes=EVAL_CONSTANTS["output_limit_bytes"],
        pids_limit=EVAL_CONSTANTS["pids_limit"],
        read_only_mounts=(("/task", task_dir),),
        gpu_devices=(gpu_device,),
    )
    outcome = execute_container(request, spec)
    payload = None
    result_file = Path(outcome.workdir) / "out" / "result.json" if outcome.workdir else None
    if result_file and result_file.is_file():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    return payload, outcome.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="pmpp_v2")
    parser.add_argument("--name", default="vectorsum_v2")
    parser.add_argument(
        "--benchmark-specs",
        type=int,
        default=1,
        help="how many benchmark entries to time (protocol keeps the upstream loop)",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help=(
            "snapshot-relative path of a pinned solution to stage as submission.py "
            "(default: the problem's own submission.py)"
        ),
    )
    parser.add_argument("--gpu-device", default="nvidia.com/gpu=0")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = load_files_manifest(FILES_MANIFEST)
    dev = load_dev_manifest(DEV_MANIFEST)
    snapshot = ReferenceKernelsSnapshot(SNAPSHOT_ROOT, files, verify=True)
    wanted_task_id = f"rk-{args.set_name}-{args.name}"
    dev_problem = next((p for p in dev.problems if p.task_id == wanted_task_id), None)
    if dev_problem is None:
        available = [p.task_id for p in dev.problems]
        print(json.dumps({"status": "FAIL", "reason": "unknown problem", "available": available}))
        return 1
    problem = problem_from_manifest(files, dev_problem, set_yamls=snapshot._set_yamls)
    specs = workload_specs(problem)
    test_specs = [s.spec for s in specs if s.kind == "test"]
    benchmark_specs = [s.spec for s in specs if s.kind == "benchmark"][: args.benchmark_specs]

    driver_text = DRIVER.read_text(encoding="utf-8")
    protocol_block = {
        "name": dev.protocol["name"],
        "identity_sha256": dev.protocol["identity_sha256"],
        "scope": "local replication; official leaderboard numbers are not comparable",
    }

    started = time.monotonic()
    results = {}
    for mode, specs_for_mode in (("test", test_specs), ("benchmark", benchmark_specs)):
        case = {
            "task_id": problem.task_id,
            "mode": mode,
            "set": problem.set_name,
            "specs": specs_for_mode,
            "protocol": protocol_block,
        }
        workspace, task_dir = _stage(snapshot, problem, case, driver_text, args.submission)
        payload, container_status = _run(task_dir, args.gpu_device, args.timeout)
        results[mode] = {"container_status": container_status, "result": payload}

    elapsed = time.monotonic() - started
    test_ok = results["test"]["result"] and results["test"]["result"].get("status") == "passed"
    bench_ok = results["benchmark"]["result"] and (
        results["benchmark"]["result"].get("status") == "passed"
    )
    summary = {
        "track": "reference-kernels-local-replica-v1",
        "task_id": problem.task_id,
        "protocol_identity": dev.protocol["identity_sha256"],
        "official_gpus": problem.official_gpus,
        "note": "local replication; numbers must never be compared with the official leaderboard",
        "wall_seconds": round(elapsed, 1),
        **results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    passed = bool(test_ok and bench_ok)
    print(
        f"rk smoke {problem.task_id}: {'PASS' if passed else 'NOT-PASSED'} in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
