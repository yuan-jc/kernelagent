"""KernelBench upstream evaluation driver (T05, ADR-0002).

Runs INSIDE the ADR-0001 container boundary; the trusted parent stages:
  /task/...                   pinned snapshot root (src/kernelbench/*,
                              KernelBench/levelN/*.py), read-only
  /case/candidate.py          the candidate whose correctness is judged
  /case/case.json             explicit run configuration
  /case/eval_driver.py        this driver
and expects the serialized upstream verdict at /out/result.json.

The driver adds nothing to the verdict: it calls the pinned
``eval_kernel_against_ref`` with measure_performance=False (correctness
only; formal timing is T07) and records the KernelExecResult verbatim.
Build/cache directories point into the container-private tmpfs so the
read-only rootfs is never written."""

import json
import os
import sys

TASK_ROOT = "/task"
CASE_ROOT = "/case"


def main() -> int:
    case = json.load(open(f"{CASE_ROOT}/case.json"))
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_cache"
    os.environ["HOME"] = "/tmp"
    # Upstream layout: the package lives at /task/src/kernelbench, so the
    # namespace-package parent is /task/src.
    sys.path.insert(0, f"{TASK_ROOT}/src")

    from kernelbench.eval import eval_kernel_against_ref

    with open(f"{TASK_ROOT}/{case['problem_path']}") as handle:
        original_model_src = handle.read()
    with open(f"{CASE_ROOT}/candidate.py") as handle:
        candidate_src = handle.read()

    result = eval_kernel_against_ref(
        original_model_src=original_model_src,
        custom_model_src=candidate_src,
        seed_num=case["seed"],
        num_correct_trials=case["num_correct_trials"],
        num_perf_trials=case["num_perf_trials"],
        measure_performance=False,
        build_dir="/tmp/torch_ext",
        device=case["device"],
        backend=case["backend"],
    )
    if result is None:
        payload = {"upstream": None, "note": "upstream returned None (lock-file/retry path)"}
    else:
        payload = {"upstream": result.model_dump()}
    with open("/out/result.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
