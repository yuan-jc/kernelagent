"""MultiKernelBench worker driver (mkb-upstream-cuda-v1 track).

Runs INSIDE the worker boundary; the trusted parent stages a read-only
mount at /task with the pinned snapshot bytes:

  /task/protocol/config.py               pinned upstream protocol files
  /task/protocol/utils/correctness.py    (sha256 verified by the parent
  /task/protocol/utils/performance.py     against the dev manifest)
  /task/reference/<category>/<op>.py     the pinned problem (Model is truth)
  /task/candidate.py                     the candidate ModelNew source
  /task/case.json                        problem id + device + protocol identity
  /task/mkb_driver.py                    this driver

The driver adds nothing to the verdict: correctness is whatever the pinned
upstream ``utils/correctness.py::execute_template`` returns (allclose
atol=rtol=1e-4, bool exact, 5 trials, seed 1024), and timing is whatever
the pinned ``utils/performance.py::time_execution_event_template`` returns
(CUDA events, warmup 3, 100 trials, no L2 flush). Numbers from this track
are only comparable inside the mkb-upstream-cuda-v1 track.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

TASK_ROOT = "/task"


def _exec_module(name: str, path: str) -> dict:
    source = open(path, encoding="utf-8").read()
    namespace: dict = {"__name__": name, "__file__": path}
    exec(compile(source, path, "exec"), namespace)  # noqa: S102 - worker-side untrusted code by design
    return namespace


def main() -> int:
    import torch

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    os.environ["HOME"] = "/tmp"

    case = json.load(open(f"{TASK_ROOT}/case.json"))
    payload = {
        "protocol": case.get("protocol", {}),
        "task_id": case.get("task_id"),
        "category": case.get("category"),
        "op": case.get("op"),
        "status": "inconclusive",
        "correct": False,
        "correctness_information": "",
        "elapsed_times_ms": [],
        "notes": [],
    }
    try:
        torch.cuda.init()
        payload["device"] = torch.cuda.get_device_name(0)
        device = f"cuda:{int(case.get('device', 0))}"

        problem_ns = _exec_module(
            "mkb_problem", f"{TASK_ROOT}/reference/{case['category']}/{case['op']}.py"
        )
        candidate_ns = _exec_module("mkb_candidate", f"{TASK_ROOT}/candidate.py")
        if "ModelNew" not in candidate_ns:
            payload["status"] = "build_failed"
            payload["notes"].append("candidate source does not define ModelNew")
        else:
            context = {
                "Model": problem_ns["Model"],
                "ModelNew": candidate_ns["ModelNew"],
                "get_init_inputs": problem_ns["get_init_inputs"],
                "get_inputs": problem_ns["get_inputs"],
            }
            if "get_input_groups" in problem_ns:
                context["get_input_groups"] = problem_ns["get_input_groups"]

            sys.path.insert(0, f"{TASK_ROOT}/protocol")
            from utils.correctness import execute_template
            from utils.performance import time_execution_event_template

            correct, information = execute_template(
                synchronize=torch.cuda.synchronize, device=device, context=context
            )
            payload["correct"] = bool(correct)
            payload["correctness_information"] = str(information)
            payload["status"] = "passed" if correct else "incorrect"
            if correct:
                payload["elapsed_times_ms"] = time_execution_event_template(
                    context=context,
                    device=device,
                    synchronize=torch.cuda.synchronize,
                    event_class=torch.cuda.Event,
                    eval_target="ModelNew",
                )
    except Exception as exc:  # noqa: BLE001 - honest terminal states, never a pass
        if payload["status"] == "inconclusive":
            payload["status"] = (
                "build_failed"
                if isinstance(exc, (ImportError, KeyError, SyntaxError))
                else "inconclusive"
            )
        if "out of memory" in str(exc).lower():
            payload["status"] = "resource_exceeded"
        payload["notes"].append(f"{type(exc).__name__}: {exc}")
        payload["traceback_tail"] = traceback.format_exc()[-2000:]
    finally:
        os.makedirs("/out", exist_ok=True)
        with open("/out/result.json", "w") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
