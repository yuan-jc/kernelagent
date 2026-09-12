"""CPU acceptance entry point, with no GPU or model SDK dependencies."""

import argparse
import json
from pathlib import Path

from kernelagent import __version__
from kernelagent.acceptance import run_checks
from kernelagent.adapters.benchmarks.kernelbench import run_verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelagent")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Run CPU pytest checks and record evidence")
    check.add_argument("--tests", type=Path, default=Path("tests"))
    check.add_argument("--output", type=Path, default=Path("artifacts/checks"))
    check.add_argument("--timeout", type=float, default=300.0)
    bench = commands.add_parser("bench", help="Benchmark snapshot tools")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    verify = bench_sub.add_parser(
        "verify", help="Verify the local KernelBench snapshot against committed manifests"
    )
    verify.add_argument(
        "--root", type=Path, default=Path("research/sources/ScalingIntelligence__KernelBench")
    )
    verify.add_argument(
        "--manifest", type=Path, default=Path("configs/kernelbench/snapshot-files.manifest.json")
    )
    verify.add_argument(
        "--dev-manifest", type=Path, default=Path("configs/kernelbench/dev-manifest.json")
    )
    args = parser.parse_args(argv)
    if args.command == "bench":
        summary, exit_code = run_verify(args.root, args.manifest, args.dev_manifest)
        print(json.dumps(summary, indent=2))
        return exit_code
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    report = run_checks(args.tests, args.output, args.timeout)
    print(f"{report['status']}: {report['reason']}; report={report['report_path']}")
    return report["exit_code"]
