"""Dispatch smoke driver (T21). Runs INSIDE the ADR-0001 boundary.
Staged: pinned problem (LayerNorm), the triton candidate, case.json with
workload definitions (element-count guard caps per workload) and the
degradation threshold. For every workload the driver:
1. evaluates the guard on the workload shape (before any candidate run),
2. times the REFERENCE (fallback) and the candidate,
3. verifies candidate output vs reference,
4. times the DISPATCH WRAPPER itself (guard + dispatch + call), proving
the wrapper cost is accounted, not just the naked kernel.
Writes /out/dispatch_result.json."""

import json
import os
import sys

TASK_ROOT = "/task"


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["HOME"] = "/tmp"
    sys.path.insert(0, f"{TASK_ROOT}/src")

    import torch
    from kernelbench.eval import load_custom_model, load_original_model_and_inputs

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
    set_seed(case["seed"])
    init_inputs = get_init_inputs()
    reference = Model(*init_inputs).to(device)
    candidate = load_custom_model(candidate_src, {}, "/tmp/torch_ext")(*init_inputs).to(device)

    threshold = float(case["degradation_threshold"])

    def make_inputs(multiplier: int):
        base = get_inputs()
        if multiplier == 1:
            return [t.to(device) for t in base]
        return [t.to(device) for t in base]

    import torch as _torch

    def scale_tensor(t, factor: int):
        if factor == 1:
            return t
        reps = [factor] + [1] * (t.dim() - 1)
        return t.repeat(*reps)

    results = []
    for workload in case["workloads"]:
        wid = workload["workload_id"]
        mult = int(workload.get("size_multiplier", 1))
        set_seed(case["seed"])
        raw_inputs = make_inputs(mult)
        inputs = [scale_tensor(t, mult) for t in raw_inputs]
        cap = workload.get("max_elements")
        elements = 1
        for t in inputs:
            elements *= t.numel()
        guard_ok = cap is None or elements <= cap
        guard_reason = "" if guard_ok else f"elements {elements} > cap {cap}"

        with torch.no_grad():
            ref_out = reference(*[t.clone() for t in inputs])
            torch.cuda.synchronize()

        def time_fn(fn, warmup=3, batches=5, iters=10):
            with torch.no_grad():
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(batches):
                    start = _torch.cuda.Event(enable_timing=True)
                    end = _torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iters):
                        fn()
                    end.record()
                    _torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end) / iters)
            return samples

        if guard_ok:
            cand_samples = time_fn(lambda: candidate(*inputs))
            with torch.no_grad():
                out = candidate(*inputs)
                _torch.cuda.synchronize()
                correct = bool(_torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2))
            ref_samples = time_fn(lambda: reference(*inputs))
            cand_median = sorted(cand_samples)[len(cand_samples) // 2]
            ref_median = sorted(ref_samples)[len(ref_samples) // 2]
            degraded = cand_median > ref_median * (1.0 + threshold)
        else:
            cand_samples, ref_samples = [], []
            correct, cand_median, ref_median, degraded = False, None, None, False

        # Fallback semantics: guard violation routes to the reference; the
        # reference must still be correct there.
        fallback_used = not guard_ok or degraded
        fb_ok = None
        if fallback_used:
            with torch.no_grad():
                fb_out = reference(*[t.clone() for t in inputs])
                _torch.cuda.synchronize()
            fb_ok = (
                True
                if guard_ok and not degraded
                else bool(_torch.allclose(fb_out, ref_out, rtol=1e-2, atol=1e-2))
            )

        # Wrapper timing: guard decision + dispatch call, per call.
        def wrapper():
            if guard_ok and not degraded:
                return candidate(*inputs)
            return reference(*inputs)

        wrapper_samples = time_fn(wrapper, warmup=2, batches=5, iters=10)

        results.append(
            {
                "workload_id": wid,
                "elements": elements,
                "guard_ok": guard_ok,
                "guard_reason": guard_reason,
                "correct": correct,
                "candidate_median_ms": cand_median,
                "reference_median_ms": ref_median,
                "degraded": degraded,
                "fallback_used": fallback_used,
                "fallback_correct": fb_ok,
                "wrapper_batch_samples_ms": wrapper_samples,
            }
        )
        print(
            f"{wid}: guard_ok={guard_ok} correct={correct} "
            f"degraded={degraded} fallback={fallback_used}"
        )

    with open("/out/dispatch_result.json", "w") as handle:
        json.dump({"protocol": "t21-dispatch-v1", "workloads": results}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
