"""Private Alpha acceptance driver (launch plan Task 6, T16 U1-U3).

Runs the real optimization loop (real T05/T06/T07/T09 adapters over the
container boundary on the real GPU) in three modes:

  --mode correct  A known-correct fixed Triton candidate must complete the
                  loop and be promoted (U1; no model needed - the fixed
                  responder stands in for the provider, which proves the
                  loop, NOT live generation).
  --mode wrong    A known-wrong fixed candidate must be rejected with
                  usable structured feedback and never promoted (U2).
  --mode live     Real provider generation via MODEL_PROVIDER_API_KEY +
                  --base-url + --model (U3, LIVE_MODEL; never faked).

Exit codes follow the optimization loop mapping (0 success, 1 no
improvement, 2 budget, 3 config, 4 infra)."""

import argparse
import json
from pathlib import Path

from kernelagent.config import default_generator
from kernelagent.optimization import (
    OptimizationConfig,
    OptimizationConfigError,
    optimize,
    parse_exit_code,
)

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")


from kernelagent.webapp.jobs import FixedCandidateClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["correct", "wrong", "live"], required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/alpha/layernorm-001"))
    parser.add_argument("--problem", default="kernelbench:l1:40")
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--model", default="glm-4.5")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--gpu-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--token-budget", type=int, default=200000)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--fixtures-root", type=Path, default=FIXTURES_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output
    if args.mode == "wrong":
        output = output.with_name(output.name + "-wrong")
    if args.mode == "live":
        output = output.with_name(output.name + "-live")

    config = OptimizationConfig(
        problem=args.problem,
        backend=args.backend,
        model_id=args.model,
        base_url=args.base_url,
        max_candidates=args.max_candidates,
        max_repair_rounds=args.max_repair_rounds,
        gpu_budget_seconds=args.gpu_budget_seconds,
        token_budget=args.token_budget,
        output=output,
        resume=args.resume,
        snapshot_root=args.snapshot_root,
        gpu_device=args.gpu_device,
    )

    try:
        if args.mode == "live":
            generator = default_generator(config)
        else:
            fixture = (
                "triton_l1_p40_layernorm.py"
                if args.mode == "correct"
                else "wrong_l1_p40_layernorm.py"
            )
            source = (args.fixtures_root / fixture).read_text(encoding="utf-8")
            generator = _generator_for(config, FixedCandidateClient(source))
        result = optimize(config, generator=generator)
    except OptimizationConfigError as exc:
        print(f"config error: {exc}")
        return 3

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    expected = {"correct": "completed", "wrong": "no_improvement", "live": None}
    print(
        f"mode={args.mode} state={result.state} "
        f"champion={result.champion_sha256 or 'none'} "
        f"settled_tokens={report['budget']['settled_tokens']} "
        f"settled_gpu_seconds={report['budget']['settled_gpu_seconds']:.0f} "
        f"report={result.report_path}"
    )
    code = parse_exit_code(result)
    if expected[args.mode] is not None and result.state != expected[args.mode]:
        print(f"UNEXPECTED: mode {args.mode} must end {expected[args.mode]}")
        return 4
    if args.mode == "correct":
        champion_file = result.champion_path
        if champion_file is None or not champion_file.is_file():
            print("UNEXPECTED: correct mode must write a champion file")
            return 4
    if args.mode == "wrong":
        feedback_ok = any(
            record.get("stage") == "evaluate" and record.get("detail")
            for record in report["candidates"]
        )
        if not feedback_ok:
            print("UNEXPECTED: wrong mode must produce usable failure feedback")
            return 4
    return code


def _generator_for(config: OptimizationConfig, client: object):
    from kernelagent.adapters.models.costing import CostLedger
    from kernelagent.adapters.models.generation import CandidateGenerator

    return CandidateGenerator(client, CostLedger())


if __name__ == "__main__":
    raise SystemExit(main())
