"""T19 acceptance smoke: a pinned FlashInfer-Bench trace (definition +
workload from the upstream commit) executed through the container
boundary with project-authored sm_89 solutions.

- return-style and DPS solutions both verify against the definition's
  reference and get timed (raw batches);
- the upstream B200-only example solution is refused by the hardware
  guard on this sm_89 device (guard actually checks);
- trace revision, file hashes and tensor hashes are recorded.
Report: artifacts/t19/fib-report.json."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.benchmarks.flashinfer_bench import (
    hardware_guard,
    load_definition,
    load_solution,
    load_workload,
)
from kernelagent.worker import (
    ContainerSpec,
    WorkerRequest,
    execute_container,
)

SNAPSHOT_DEFAULT = Path("research/sources/flashinfer-bench/trace")
SOLUTIONS_DEFAULT = Path("configs/flashinfer-bench/solutions")
MANIFEST_DEFAULT = Path("configs/flashinfer-bench/trace-manifest.json")
DRIVER = Path("configs/kernelbench/fib_driver.py")
CURRENT_ARCH = "sm89"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/t19"))
    parser.add_argument("--trace-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    manifest = json.loads(MANIFEST_DEFAULT.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        target = args.trace_root / entry["path"]
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"pinned trace drift: {entry['path']}"

    definition = load_definition(args.trace_root / "definitions/gemm_n4096_k4096.json")
    workload = load_workload(args.trace_root / "workloads/gemm_n4096_k4096.jsonl")
    first_entry = workload[0]
    driver_source = DRIVER.read_text(encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    # Guard: the upstream B200-only solution must be refused on sm_89.
    b200_solution = load_solution(args.trace_root / "solutions/example_agent_solution.json")
    guard_allowed, guard_reason = hardware_guard(b200_solution, CURRENT_ARCH)
    guard_refused = not guard_allowed
    print(f"guard(B200 solution on sm89): refused={guard_refused} reason={guard_reason}")

    report_cases = []
    all_ok = guard_refused
    for solution_file in ("torch_sm89_return.json", "torch_sm89_dps.json"):
        solution = load_solution(SOLUTIONS_DEFAULT / solution_file)
        allowed, reason = hardware_guard(solution, CURRENT_ARCH)
        assert allowed, f"our own solution must pass the guard: {reason}"
        modes = ["dps"] if "dps" in solution.name else ["return"]
        inputs = args.output / f"inputs-{solution.name}"
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / "definition.json").write_bytes(
            (args.trace_root / "definitions/gemm_n4096_k4096.json").read_bytes()
        )
        (inputs / "solution.json").write_text(
            json.dumps(json.loads((SOLUTIONS_DEFAULT / solution_file).read_text()), indent=2),
            encoding="utf-8",
        )
        (inputs / "case.json").write_text(
            json.dumps(
                {
                    "workload": {
                        "uuid": first_entry.uuid,
                        "axes": first_entry.axes,
                    },
                    "device": 0,
                    "seed": 42,
                    "entry_function": "run",
                    "modes": modes,
                    "trace_revision": manifest["upstream"]["commit"],
                    "definition_sha256": definition.source_sha256,
                    "solution_sha256": solution.source_sha256,
                    "protocol": {"warmup_iters": 3, "num_batches": 5, "iters_per_batch": 10},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (inputs / "fib_driver.py").write_text(driver_source, encoding="utf-8")
        request = WorkerRequest(
            request_id=f"fib-{solution.name}",
            argv=("python3", "/task/fib_driver.py"),
            timeout_seconds=600.0,
            workspace_root=args.output,
        )
        spec = ContainerSpec(
            image="kernelagent-eval",
            image_id="sha256:cb7a9f4cfab8943aa29bdbdf531db4617daf736e8388e97eb77d62998ebd973b",
            memory_bytes=8 * 1024 * 1024 * 1024,
            read_only_mounts=(("/task", inputs),),
            gpu_devices=(gpu_device,),
        )
        outcome = execute_container(request, spec)
        result_file = Path(outcome.workdir) / "out" / "fib_result.json"
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        verified = all(m["correct"] for m in payload["verification"].values())
        timed = all(len(m["batch_samples_ms"]) == 5 for m in payload["timings"])
        ok = outcome.status == "completed" and verified and timed
        all_ok = all_ok and ok
        report_cases.append(
            {
                "solution": solution.name,
                "modes": modes,
                "outcome_status": outcome.status,
                "verified": verified,
                "timed": timed,
                "workload_uuid": payload["workload_uuid"],
                "input_hashes": payload["input_hashes"],
                "trace_revision": payload["trace_revision"],
            }
        )
        print(f"{solution.name}: verified={verified} timed={timed} ok={ok}")

    payload = {
        "protocol": "t19-flashinfer-bench-v1",
        "commit": commit,
        "trace_revision": manifest["upstream"]["commit"],
        "gpu_device": gpu_device,
        "guard_cases": [
            {
                "solution": "example_agent_solution (upstream)",
                "target_hardware": ["B200"],
                "refused": guard_refused,
                "reason": guard_reason,
            }
        ],
        "cases": report_cases,
        "accepted": all_ok,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "fib-report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"report={report_path} accepted={all_ok}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
