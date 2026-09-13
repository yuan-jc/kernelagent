"""``python -m kernelagent.webapp`` - start the local console server."""

from __future__ import annotations

import argparse
from pathlib import Path

from kernelagent.webapp.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kernelagent.webapp", description="KernelAgent local web console"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep local)")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/webui"))
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("research/sources/ScalingIntelligence__KernelBench"),
    )
    parser.add_argument(
        "--fixtures-root", type=Path, default=Path("configs/kernelbench/eval-fixtures")
    )
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        runs_root=args.runs_root,
        snapshot_root=args.snapshot_root,
        fixtures_root=args.fixtures_root,
    )


if __name__ == "__main__":
    main()
