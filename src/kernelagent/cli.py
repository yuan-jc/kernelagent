"""CPU acceptance entry point, with no GPU or model SDK dependencies."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent import __version__
from kernelagent.acceptance import run_checks
from kernelagent.adapters.benchmarks.kernelbench import run_verify


def _run_probe_native() -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "kernelagent.probe"], capture_output=True, text=True
    )
    return completed.returncode, completed.stdout, completed.stderr


_PROBE_MARKER = "KERNELAGENT_PROBE_JSON_BEGIN\n"


def _run_probe_wsl(distro: str | None) -> tuple[int, str, str]:
    probe_source = (Path(__file__).parent / "probe.py").read_text(encoding="utf-8")
    command = ["wsl.exe"]
    if distro:
        command += ["-d", distro]
    # Stage the script as a file first: wsl.exe's stdin relay truncates large
    # payloads fed straight to `python3 -` (observed SIGSEGV), while cat is
    # reliable at consuming the pipe to EOF.
    remote = (
        "cat > /tmp/kernelagent_probe_$$.py"
        " && python3 /tmp/kernelagent_probe_$$.py"
        "; rc=$?; rm -f /tmp/kernelagent_probe_$$.py; exit $rc"
    )
    command += ["-e", "bash", "-c", remote]
    completed = subprocess.run(
        command, input=probe_source, capture_output=True, text=True, timeout=300
    )
    return completed.returncode, completed.stdout, completed.stderr


def _probe_command(args: argparse.Namespace) -> int:
    if args.target == "wsl":
        code, stdout, stderr = _run_probe_wsl(args.wsl_distro)
    else:
        code, stdout, stderr = _run_probe_native()
    if _PROBE_MARKER in stdout:
        stdout = stdout.split(_PROBE_MARKER, 1)[1]
    if not stdout.strip():
        print(f"probe produced no report (exit {code}); stderr: {stderr.strip()[-400:]}")
        return 1
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gpu-probe-report.json"
    report_path.write_text(stdout.lstrip("\r\n"), encoding="utf-8", newline="\n")
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"probe output is not valid JSON (exit {code}); saved to {report_path}")
        return 1
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    line = (
        f"probe protocol={report.get('probe_protocol')} exit={code} "
        f"summary={report.get('summary')} report={report_path} sha256={digest}"
    )
    if args.evidence_root:
        from kernelagent.adapters.storage import EvidenceStore

        with EvidenceStore(Path(args.evidence_root)) as store:
            ref = store.put_artifact(
                report_path.read_bytes(), kind="gpu_probe_report", producer_version=__version__
            )
            line += f" evidence={ref.artifact_sha256[:12]}…"
    print(line)
    return code if code in (0, 1) else 1


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
    probe = commands.add_parser("probe", help="Probe a target environment's GPU/CUDA capabilities")
    probe.add_argument("--target", choices=["native", "wsl"], default="native")
    probe.add_argument(
        "--wsl-distro", default=None, help="WSL distro name (defaults to system default)"
    )
    probe.add_argument(
        "--output", type=Path, default=Path("artifacts/probe"), help="Report output directory"
    )
    probe.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Optionally record the report into an EvidenceStore at this root",
    )
    args = parser.parse_args(argv)
    if args.command == "bench":
        summary, exit_code = run_verify(args.root, args.manifest, args.dev_manifest)
        print(json.dumps(summary, indent=2))
        return exit_code
    if args.command == "probe":
        return _probe_command(args)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    report = run_checks(args.tests, args.output, args.timeout)
    print(f"{report['status']}: {report['reason']}; report={report['report_path']}")
    return report["exit_code"]
