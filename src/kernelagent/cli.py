"""CPU acceptance entry point, with no GPU or model SDK dependencies."""

import argparse
from pathlib import Path

from kernelagent import __version__
from kernelagent.acceptance import run_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelagent")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Run CPU pytest checks and record evidence")
    check.add_argument("--tests", type=Path, default=Path("tests"))
    check.add_argument("--output", type=Path, default=Path("artifacts/checks"))
    check.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    report = run_checks(args.tests, args.output, args.timeout)
    print(f"{report['status']}: {report['reason']}; report={report['report_path']}")
    return report["exit_code"]
