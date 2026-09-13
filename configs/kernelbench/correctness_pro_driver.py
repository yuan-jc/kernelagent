"""Correctness-pro driver (T06, design §9.1-9.2). Runs INSIDE the ADR-0001
container boundary with the same staging as T05/T07:
  /task/src/kernelbench/*.py   pinned upstream evaluator (loader helpers)
  /task/KernelBench/levelN/..  the pinned problem file
  /task/candidate.py           the candidate under enhanced scrutiny
  /task/case.json              check configuration
and writes /out/pro_result.json.

The extended checks are a SEPARATE track: this driver never rewrites the
upstream evaluator's verdict (that belongs to T05); it only records its
own check results. Every check is reported individually, sanitizer
unavailability is recorded as ``not_run`` (never a pass), and the overall
verdict is False unless every check passed."""

import json
import os
import subprocess
import sys

TASK_ROOT = "/task"


def _sanitizer_snippet(problem_path: str, backend: str) -> str:
    # Triton-family backends need their source in a real file (@triton.jit
    # cannot take code from an exec string), so the snippet must use the
    # same tempfile loader as the in-process check.
    if backend.lower() in ("triton", "tilelang", "cute"):
        load_line = (
            "new = load_custom_model_with_tempfile(candidate_src)[0](*get_init_inputs()).cuda()\n"
        )
    else:
        load_line = (
            "new = load_custom_model(candidate_src, {}, '/tmp/torch_ext')"
            "(*get_init_inputs()).cuda()\n"
        )
    return (
        "import json, sys, torch\n"
        f"sys.path.insert(0, {TASK_ROOT + '/src'!r})\n"
        "from kernelbench.eval import load_original_model_and_inputs, load_custom_model,"
        " load_custom_model_with_tempfile\n"
        f"problem_src = open({problem_path!r}).read()\n"
        f"candidate_src = open('{TASK_ROOT}/candidate.py').read()\n"
        "Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})\n"
        "ref = Model(*get_init_inputs()).cuda()\n"
        f"{load_line}"
        "x = [t.cuda() for t in get_inputs()]\n"
        "out = new(*x)\n"
        "torch.cuda.synchronize()\n"
        "print('sanitized-ok')\n"
    )


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["HOME"] = "/tmp"
    sys.path.insert(0, f"{TASK_ROOT}/src")

    import torch
    from kernelbench.eval import (  # noqa: F401 - sanity: pinned loader present
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
    )

    def set_seed(seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    with open(f"{TASK_ROOT}/{case['problem_path']}") as handle:
        problem_src = handle.read()
    with open(f"{TASK_ROOT}/candidate.py") as handle:
        candidate_src = handle.read()

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {})
    device = torch.device(f"cuda:{case['device']}")
    reference = Model(*get_init_inputs()).to(device)
    if case.get("backend", "cuda").lower() in ("triton", "tilelang", "cute"):
        target, _tempfile_obj = load_custom_model_with_tempfile(candidate_src)
        target = target(*get_init_inputs()).to(device)
    else:
        target = load_custom_model(candidate_src, {}, "/tmp/torch_ext")(*get_init_inputs()).to(
            device
        )

    set_seed(case["seed"])
    inputs_a = [t.to(device) for t in get_inputs()]
    set_seed(case["seed"] + 1)
    inputs_b = [t.to(device) for t in get_inputs()]

    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, **detail) -> None:
        checks[name] = {"passed": bool(passed), **detail}

    with torch.no_grad():
        # Preserve pre-call input bytes for the mutation check.
        snapshots_a = [t.detach().clone() for t in inputs_a]
        snapshots_b = [t.detach().clone() for t in inputs_b]

        out_a1 = target(*inputs_a)
        torch.cuda.synchronize()
        ref_a = reference(*inputs_a)
        out_b = target(*inputs_b)
        torch.cuda.synchronize()
        ref_b = reference(*inputs_b)
        out_a2 = target(*inputs_a)
        torch.cuda.synchronize()

    flat_o, flat_r = out_a1.reshape(-1), ref_a.reshape(-1)
    record(
        "baseline",
        bool(torch.allclose(flat_o, flat_r, rtol=1e-2, atol=1e-2)),
        mismatched=int((flat_o != flat_r).sum().item()),
        elements=int(flat_o.numel()),
    )
    record(
        "shape_dtype",
        out_a1.shape == ref_a.shape and out_a1.dtype == ref_a.dtype,
        candidate_shape=list(out_a1.shape),
        reference_shape=list(ref_a.shape),
        candidate_dtype=str(out_a1.dtype),
        reference_dtype=str(ref_a.dtype),
    )
    mutated = any(not torch.equal(t, s) for t, s in zip(inputs_a, snapshots_a))
    mutated_b = any(not torch.equal(t, s) for t, s in zip(inputs_b, snapshots_b))
    record("input_mutation", not (mutated or mutated_b), mutated_a=mutated, mutated_b=mutated_b)
    constant = bool(torch.allclose(out_a1, out_b))
    distinct_inputs = not all(torch.allclose(a, b) for a, b in zip(inputs_a, inputs_b))
    record(
        "constant_output",
        (not constant) if distinct_inputs else True,
        distinct_inputs=distinct_inputs,
    )
    n = flat_o.numel()
    tail_n = max(1, n // 100)
    body_ok = bool(torch.allclose(flat_o[:-tail_n], flat_r[:-tail_n], rtol=1e-2, atol=1e-2))
    tail_ok = bool(torch.allclose(flat_o[-tail_n:], flat_r[-tail_n:], rtol=1e-2, atol=1e-2))
    record(
        "tail_check",
        body_ok and tail_ok,
        body_ok=body_ok,
        tail_ok=tail_ok,
        tail_elements=tail_n,
    )
    repeat_ok = bool(torch.allclose(out_a1, out_a2)) and bool(
        torch.allclose(out_a2, ref_a, rtol=1e-2, atol=1e-2)
    )
    state_b_ok = bool(torch.allclose(out_b, ref_b, rtol=1e-2, atol=1e-2))
    record(
        "state_check", repeat_ok and state_b_ok, repeat_ok=repeat_ok, distinct_input_ok=state_b_ok
    )

    # Sanitizer: short memcheck invocation in a fresh process. Missing tool
    # is recorded as not_run; a nonzero exit is fail. Never a silent pass.
    sanitizer_bin = case.get("sanitizer_binary", "compute-sanitizer")
    snippet = _sanitizer_snippet(f"{TASK_ROOT}/{case['problem_path']}", case.get("backend", "cuda"))
    resolved = None
    try:
        resolved = subprocess.run(
            ["which", sanitizer_bin], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        resolved = None
    if not resolved:
        checks["sanitizer"] = {
            "status": "not_run",
            "reason": f"{sanitizer_bin!r} not found in container",
        }
    else:
        try:
            completed = subprocess.run(
                [
                    sanitizer_bin,
                    "--tool",
                    "memcheck",
                    "--error-exitcode",
                    "9",
                    sys.executable,
                    "-c",
                    snippet,
                ],
                capture_output=True,
                text=True,
                timeout=case.get("sanitizer_timeout", 240),
            )
            checks["sanitizer"] = {
                "status": "pass" if completed.returncode == 0 else "fail",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-800:],
            }
        except subprocess.TimeoutExpired:
            checks["sanitizer"] = {"status": "not_run", "reason": "sanitizer invocation timed out"}

    core_pass = all(entry["passed"] for name, entry in checks.items() if name != "sanitizer")
    sanitizer_status = checks["sanitizer"]["status"]
    overall = core_pass and sanitizer_status == "pass"
    payload = {
        "protocol": "t06-correctness-pro-v1",
        "checks": checks,
        "core_pass": core_pass,
        "sanitizer_status": sanitizer_status,
        "overall_pass": overall,
    }
    with open("/out/pro_result.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
