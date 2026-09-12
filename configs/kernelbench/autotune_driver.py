"""Autotune adapter driver (T14, design §8.4). Runs INSIDE the ADR-0001
container boundary with the standard staging (/task + /out).

Config-sweep contract (the adapter's addition over the upstream
decorator, for per-config accounting):
- case.json carries ``configs`` (Triton meta-parameter dicts with a
  ``block_size`` int) and the fixture contract is
  ``ModelNew(*init_inputs, block_size=N)`` writing its result in place.
- structural prune BEFORE any GPU work: non-power-of-two block sizes and
  sizes above the compile-sanity bound never reach measurement;
- per-config correctness against the pinned reference - an incorrect
  config is recorded and excluded from winner selection;
- in-place fairness: pristine input bytes are restored before every
  batch (restoration events counted), because the fixture writes its
  result into the input storage;
- the winner (correct config with lowest median batch time) is frozen
  and re-verified with a fresh run; a frozen config that fails
  re-verification sets ``frozen_verification_failed`` instead of a
  silent pass."""

import json
import os
import sys

TASK_ROOT = "/task"


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def main() -> int:
    case = json.load(open(f"{TASK_ROOT}/case.json"))
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["HOME"] = "/tmp"
    # Probe subprocesses inherit these: unbounded BLAS thread pools inside
    # a pids-limited container segfault during numpy import.
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    sys.path.insert(0, f"{TASK_ROOT}/src")

    import torch
    from kernelbench.eval import (
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
    set_seed(case["seed"])
    reference = Model(*get_init_inputs()).to(device)
    pristine = [t.to(device).clone() for t in get_inputs()]

    with torch.no_grad():
        ref_out = reference(*[t.clone() for t in pristine])
        torch.cuda.synchronize()

    # Triton JIT requires real source files: use the upstream tempfile
    # loader (identical to the evaluator's triton backend path).
    class_handle, _tempfile = load_custom_model_with_tempfile(candidate_src)
    protocol = case["protocol"]
    restorations = 0

    def verify(model, inputs) -> bool:
        with torch.no_grad():
            out = model(*[t.clone() for t in inputs])
            torch.cuda.synchronize()
            return bool(torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2))

    # Config probes run in isolated subprocesses: a hostile/giant config
    # (e.g. BLOCK=2^20 makes the Triton compiler balloon memory) can then
    # only kill its probe, never the sweep or the container.
    probe_snippet = (
        "import json, sys\n"
        "sys.path.insert(0, {src!r})\n"
        "cfg = json.load(open({case!r}))\n"
        "from kernelbench.eval import load_original_model_and_inputs\n"
        "from kernelbench.eval import load_custom_model_with_tempfile\n"
        "problem_src = open('/task/' + cfg['problem_path']).read()\n"
        "candidate_src = open('/task/candidate.py').read()\n"
        "Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem_src, {{}})\n"
        "import torch\n"
        "device = torch.device('cuda:' + str(cfg['device']))\n"
        "ref = Model(*get_init_inputs()).to(device)\n"
        "inputs = [t.to(device) for t in get_inputs()]\n"
        "block = int(sys.argv[1])\n"
        "new = load_custom_model_with_tempfile(candidate_src)[0]\n"
        "new = new(*get_init_inputs(), block_size=block).to(device)\n"
        "out = new(*inputs)\n"
        "torch.cuda.synchronize()\n"
        "assert torch.allclose(out, ref(*inputs), rtol=1e-2, atol=1e-2)\n"
    ).format(src=f"{TASK_ROOT}/src", case=f"{TASK_ROOT}/case.json")

    records = []
    for config in case["configs"]:
        block = config["block_size"]
        record = {"config": dict(config)}
        if not _is_pow2(block) or block > 1 << 20:
            record["status"] = "pruned"
            record["reason"] = "structurally invalid block size"
            records.append(record)
            continue
        import subprocess as _sp

        probe = _sp.run(
            [sys.executable, "-c", probe_snippet, str(block)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if probe.returncode != 0:
            reason = (
                "probe killed (resource exhaustion)"
                if probe.returncode == -9
                else f"probe failed rc={probe.returncode}: {probe.stderr.strip()[-200:]}"
            )
            record["status"] = "pruned"
            record["reason"] = reason
            records.append(record)
            continue
        model = class_handle(*get_init_inputs(), block_size=block).to(device)
        inputs = [t.clone() for t in pristine]
        with torch.no_grad():
            out = model(*inputs)
            torch.cuda.synchronize()
        correct = bool(torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2))
        record["correct"] = correct
        if not correct:
            record["status"] = "incorrect"
            records.append(record)
            continue
        samples = []
        with torch.no_grad():
            for _ in range(protocol["warmup_iters"]):
                model(*inputs)
            torch.cuda.synchronize()
            for _ in range(protocol["num_batches"]):
                for live, saved in zip(inputs, pristine):
                    live.copy_(saved)
                restorations += 1
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(protocol["iters_per_batch"]):
                    model(*inputs)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / protocol["iters_per_batch"])
        record["status"] = "measured"
        record["batch_samples_ms"] = samples
        record["median_ms"] = sorted(samples)[len(samples) // 2]
        records.append(record)

    measured = [r for r in records if r.get("status") == "measured"]
    winner = None
    frozen_verified = False
    if measured:
        winner = min(measured, key=lambda r: r["median_ms"])["config"]
        block = winner["block_size"]
        model = class_handle(*get_init_inputs(), block_size=block).to(device)
        inputs = [t.clone() for t in pristine]
        with torch.no_grad():
            out = model(*inputs)
            torch.cuda.synchronize()
        frozen_verified = bool(torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2))

    payload = {
        "protocol": "t14-autotune-v1",
        "configs": records,
        "winner": winner,
        "frozen_verified": frozen_verified,
        "input_restore_events": restorations,
        "measured_count": len(measured),
    }
    with open("/out/autotune.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
