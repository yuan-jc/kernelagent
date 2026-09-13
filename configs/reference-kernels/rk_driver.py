"""GPU-MODE reference-kernels worker driver (reference-kernels-local-replica-v1).

Runs INSIDE the worker boundary; the trusted parent stages the pinned
snapshot bytes at their upstream-relative layout plus the candidate:

  /task/problems/<set>/...        pinned set files (eval.py, utils.py, ...)
  /task/problems/<set>.yaml       pinned set metadata
  /task/work/submission.py        the candidate (custom_kernel entry)
  /task/case.json                 mode, set, problem dir, test specs, identity
  /task/rk_driver.py              this driver

The pinned upstream per-set ``eval.py`` is executed verbatim (test or
benchmark mode) through its own Popcorn output protocol; the driver only
captures that output, parses the key/value log lines and records them with
the protocol identity. Correctness verdicts are the upstream
``check_implementation`` verdicts; timing follows the upstream benchmark
loop (clear_l2_cache, CUDA events, ns statistics bounded by
max_repeats/max_time). POPCORN_SEED is intentionally left unset: local
replication runs the public per-spec seeds only, and this must not be
presented as equivalent to the official leaderboard's secret-seed scheme.

Local replication only: official leaderboard numbers come from
B200/H100/A100/L4 hardware and are never comparable with results produced
here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

TASK_ROOT = "/task"


def _write_tests_file(path: str, specs: list[dict]) -> None:
    lines = []
    for spec in specs:
        lines.append(";".join(f"{key}:{value}" for key, value in spec.items()))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _parse_popcorn(text: str) -> dict:
    parsed: dict = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed


def main() -> int:

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache")
    os.environ.setdefault("HOME", "/tmp")

    import torch

    case = json.load(open(f"{TASK_ROOT}/case.json"))
    payload = {
        "protocol": case.get("protocol", {}),
        "task_id": case.get("task_id"),
        "mode": case.get("mode"),
        "status": "inconclusive",
        "correct": False,
        "popcorn": {},
        "notes": [],
    }
    workdir = "/tmp/work"  # container-private tmpfs; /task is read-only
    runtime_dir = case.get("runtime_dir", f"{TASK_ROOT}/runtime")
    fd = None
    try:
        torch.cuda.init()
        payload["device"] = torch.cuda.get_device_name(0)
        os.makedirs(workdir, exist_ok=True)
        _write_tests_file(f"{workdir}/tests.txt", case["specs"])

        # Upstream layout: the pinned task.yml `files` mapping is materialized
        # flat into the runtime directory (eval.py imports utils/reference and
        # submission imports task from the same directory), and the pinned
        # eval.py is executed verbatim as a script - the upstream
        # multiprocessing spawn pool re-imports __main__, which only works
        # with the real script entry point.
        fd = os.open(f"{workdir}/popcorn.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        # PEP 446: descriptors are non-inheritable by default; the child
        # process receives the fd number via POPCORN_FD and needs it open.
        os.set_inheritable(fd, True)
        child_env = dict(os.environ)
        child_env["POPCORN_FD"] = str(fd)
        # the spawn pool plus BLAS pools exceed the container's pids limit
        child_env.setdefault("OPENBLAS_NUM_THREADS", "4")
        child_env.setdefault("OMP_NUM_THREADS", "4")
        child_env.setdefault("MKL_NUM_THREADS", "4")
        completed = subprocess.run(
            [sys.executable, "eval.py", case["mode"], f"{workdir}/tests.txt"],
            cwd=runtime_dir,
            env=child_env,
            pass_fds=(fd,),
            check=False,
        )
        exit_code = completed.returncode
        try:
            os.close(fd)
        except OSError:
            pass
        fd = None
        payload["exit_code"] = exit_code
        payload["popcorn"] = _parse_popcorn(open(f"{workdir}/popcorn.log", encoding="utf-8").read())
        check = payload["popcorn"].get("check")
        if case["mode"] == "test":
            payload["correct"] = check == "pass"
            payload["status"] = "passed" if check == "pass" else "incorrect"
        else:
            benchmarks = {
                key: value
                for key, value in payload["popcorn"].items()
                if key.startswith("benchmark.")
            }
            payload["correct"] = check == "pass"
            payload["status"] = "passed" if check == "pass" else "incorrect"
            payload["benchmarks"] = benchmarks
    except Exception as exc:  # noqa: BLE001 - honest terminal states, never a pass
        if "out of memory" in str(exc).lower():
            payload["status"] = "resource_exceeded"
        payload["notes"].append(f"{type(exc).__name__}: {exc}")
        payload["traceback_tail"] = traceback.format_exc()[-2000:]
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except OSError:
            pass
        os.makedirs("/out", exist_ok=True)
        with open("/out/result.json", "w") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
