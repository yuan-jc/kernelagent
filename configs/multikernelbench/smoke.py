"""MultiKernelBench GPU smoke: one problem, real correctness + timing.

Parent-side runner (trusted): stages the pinned snapshot bytes into a
read-only /task mount, runs configs/multikernelbench/mkb_driver.py inside
the pinned evaluation container and prints an honest summary. The verdict
comes from the pinned upstream utils/correctness.py (mkb-upstream-cuda-v1);
a container/infrastructure failure is reported as such and never as a
candidate failure.

Usage: .venv/bin/python configs/multikernelbench/smoke.py \
          [--category broadcast --op add_bias_broadcast] \
          [--candidate path/to/ModelNew.py] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kernelagent.adapters.benchmarks.multikernelbench import (
    load_dev_manifest,
    load_files_manifest,
)
from kernelagent.adapters.evals.kernelbench_eval import EVAL_IMAGE_ID, EVAL_IMAGE_REPO
from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = REPO / "research/sources/wzzll123__MultiKernelBench"
FILES_MANIFEST = REPO / "configs/multikernelbench/snapshot-files.manifest.json"
DEV_MANIFEST = REPO / "configs/multikernelbench/dev-manifest.json"
DRIVER = REPO / "configs/multikernelbench/mkb_driver.py"

# Trivially correct candidate (torch eager) - enough to exercise the whole
# pinned protocol path end to end.
DEFAULT_CANDIDATE = (
    "import torch\nimport torch.nn as nn\n\n\n"
    "class ModelNew(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "\n"
    "    def forward(self, x, bias):\n"
    "        return torch.add(x, bias)\n"
)

EVAL_CONSTANTS = {
    "memory_bytes": 8 * 1024**3,
    "tmp_tmpfs_bytes": 4 * 1024**3,
    "output_limit_bytes": 8 * 1024**3,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="broadcast")
    parser.add_argument("--op", default="add_bias_broadcast")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--gpu-device", default="nvidia.com/gpu=0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = load_files_manifest(FILES_MANIFEST)
    dev = load_dev_manifest(DEV_MANIFEST)
    from kernelagent.adapters.benchmarks.multikernelbench import MkbSnapshot

    snapshot = MkbSnapshot(SNAPSHOT_ROOT, files, verify=True)
    problem = snapshot.get_problem(args.category, args.op)
    candidate_source = (
        args.candidate.read_text(encoding="utf-8") if args.candidate else DEFAULT_CANDIDATE
    )

    started = time.monotonic()
    workspace = Path(__import__("tempfile").mkdtemp(prefix="mkb-smoke-"))
    task_dir = workspace / "task-in"
    (task_dir / "protocol" / "utils").mkdir(parents=True)
    (task_dir / "reference" / args.category).mkdir(parents=True)
    for reference in dev.protocol.get("references", []):
        target = task_dir / "protocol" / reference["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SNAPSHOT_ROOT / reference["path"]).read_bytes())
    (task_dir / "reference" / args.category / f"{args.op}.py").write_text(
        problem.code, encoding="utf-8"
    )
    (task_dir / "candidate.py").write_text(candidate_source, encoding="utf-8")
    (task_dir / "mkb_driver.py").write_text(DRIVER.read_text(encoding="utf-8"), encoding="utf-8")
    case = {
        "task_id": problem.ref.task_id,
        "category": args.category,
        "op": args.op,
        "device": args.device_index,
        "protocol": {
            "name": dev.protocol["name"],
            "identity_sha256": dev.protocol["identity_sha256"],
            "problem_sha256": problem.ref.sha256,
        },
    }
    (task_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

    request = WorkerRequest(
        request_id=f"mkb-smoke-{problem.ref.task_id}"[:64],
        argv=("python3", "/task/mkb_driver.py"),
        timeout_seconds=args.timeout,
        workspace_root=workspace,
    )
    spec = ContainerSpec(
        image=EVAL_IMAGE_REPO,
        image_id=EVAL_IMAGE_ID,
        memory_bytes=EVAL_CONSTANTS["memory_bytes"],
        tmp_tmpfs_bytes=EVAL_CONSTANTS["tmp_tmpfs_bytes"],
        output_limit_bytes=EVAL_CONSTANTS["output_limit_bytes"],
        read_only_mounts=(("/task", task_dir),),
        gpu_devices=(args.gpu_device,),
    )
    outcome = execute_container(request, spec)

    payload = None
    result_file = Path(outcome.workdir) / "out" / "result.json" if outcome.workdir else None
    if result_file and result_file.is_file():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    elapsed = time.monotonic() - started
    summary = {
        "track": "mkb-upstream-cuda-v1",
        "task_id": problem.ref.task_id,
        "container_status": outcome.status,
        "protocol_identity": dev.protocol["identity_sha256"],
        "result": payload,
        "wall_seconds": round(elapsed, 1),
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    passed = bool(payload) and payload.get("status") == "passed"
    print(
        f"MKB smoke {problem.ref.task_id}: "
        f"{'PASS' if passed else 'NOT-PASSED'} in {elapsed:.1f}s "
        f"(container {outcome.status})",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
