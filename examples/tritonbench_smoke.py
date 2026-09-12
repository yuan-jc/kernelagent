"""T18 acceptance smoke: Meta tritonbench vector_add through the T04
container boundary, with a project-authored backend registered via the
upstream registration API (no upstream file modified - manifest verified)
and a same-protocol (T07 CUDA-event) comparison. Report:
artifacts/t18/tritonbench-report.json."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    execute_container,
)

SNAPSHOT_DEFAULT = Path("research/sources/tritonbench")
MANIFEST_DEFAULT = Path("configs/tritonbench/manifest.json")
PLUGIN_DEFAULT = Path("configs/tritonbench/ka_plugin.py")
DRIVER_DEFAULT = Path("configs/tritonbench/tb_ka_driver.py")
EVAL_IMAGE_ID = "sha256:5ce0c61e8d6fc0c053d137e9ad730fe0d9d96810fad7eba24b03ba1f4b2b893e"


def _default_gpu_device() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return "nvidia.com/gpu=0"
    first = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    return f"nvidia.com/gpu={first}" if first else "nvidia.com/gpu=0"


def _verify_manifest(clone: Path, manifest: dict) -> bool:
    return all(
        hashlib.sha256((clone / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        for entry in manifest["files"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/t18"))
    parser.add_argument("--clone-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    manifest = json.loads(MANIFEST_DEFAULT.read_text(encoding="utf-8"))
    workspace_clone = args.output / "workspace" / "tritonbench"
    if not workspace_clone.exists():
        shutil.copytree(
            args.clone_root,
            workspace_clone,
            ignore=shutil.ignore_patterns(".git", "submodules"),
        )
    clone_ok = _verify_manifest(workspace_clone, manifest)
    print(f"upstream files byte-identical to pinned commit: {clone_ok}")

    inputs = args.output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "case.json").write_text(
        json.dumps(
            {
                "device": 0,
                "size": 1 << 20,
                "protocol": {"warmup_iters": 3, "num_batches": 5, "iters_per_batch": 10},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (inputs / "tb_ka_driver.py").write_text(DRIVER_DEFAULT.read_text(encoding="utf-8"))
    (inputs / "ka_plugin.py").write_text(PLUGIN_DEFAULT.read_text(encoding="utf-8"))

    request = WorkerRequest(
        request_id="tb-vector-add",
        argv=("python3", "/task/tb_ka_driver.py"),
        timeout_seconds=900.0,
        workspace_root=args.output,
        output_tail_chars=60000,
        env_extra=(("PYTHONPATH", "/task:/tritonbench"),),
    )
    spec = ContainerSpec(
        image="kernelagent-eval",
        image_id=EVAL_IMAGE_ID,
        memory_bytes=8 * 1024 * 1024 * 1024,
        pids_limit=256,
        read_only_mounts=(
            ("/tritonbench", workspace_clone),
            ("/task", inputs),
        ),
        gpu_devices=(gpu_device,),
    )
    outcome = execute_container(request, spec)
    result_file = Path(outcome.workdir) / "out" / "tb_ka.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))

    registered = payload["registered_before_run"] is True
    table_has_backend = payload["upstream_table_contains_ka_backend"] is True
    entries = payload["entries"]
    same_protocol_ok = len(entries) == 3 and all(
        len(e["batch_samples_ms"]) == 5 for e in entries
    )
    accepted = (
        outcome.status == "completed"
        and clone_ok
        and registered
        and table_has_backend
        and same_protocol_ok
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report = {
        "protocol": "t18-tritonbench-v1",
        "commit": commit,
        "upstream_commit": manifest["upstream"]["commit"],
        "gpu_device": gpu_device,
        "manifest_files": len(manifest["files"]),
        "clone_byte_identical": clone_ok,
        "registered_before_run": registered,
        "upstream_table_contains_ka_backend": table_has_backend,
        "upstream_table_tail": payload["upstream_table_tail"],
        "same_protocol_entries": entries,
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "tritonbench-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"clone_ok={clone_ok} registered={registered} table_has_backend={table_has_backend} "
        f"same_protocol_ok={same_protocol_ok}"
    )
    print(f"report={report_path} accepted={accepted}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
