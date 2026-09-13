"""CPU acceptance entry point, with no GPU or model SDK dependencies."""

import argparse
import hashlib
import json
import os
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


def _optimize_arguments(parser: argparse.ArgumentParser, *, resume_default: bool = False) -> None:
    """Add the optimize/resume arguments.

    Resume semantics (RV01): every run-scoped argument defaults to None,
    meaning "restore the original value from the persisted run manifest".
    A value passed on the command line is an explicit override: identity
    fields must still match the manifest (the run is refused otherwise) and
    revisable fields (budgets, candidate allowance) are applied with an
    explicit revision event. This is how "only --output/--base-url given"
    resumes the original problem, model and budgets instead of silently
    falling back to this machine's defaults."""
    if resume_default:
        parser.add_argument(
            "--problem", default=None, help="e.g. kernelbench:l1:40 (default: manifest)"
        )
        parser.add_argument("--backend", default=None, help="candidate backend (default: manifest)")
        parser.add_argument("--model", default=None, help="provider model id (default: manifest)")
        parser.add_argument(
            "--base-url",
            dest="base_url",
            default=None,
            help="OpenAI-compatible base URL (default: manifest)",
        )
        parser.add_argument(
            "--max-candidates",
            type=int,
            default=None,
            help="override the manifest's candidate allowance",
        )
        parser.add_argument(
            "--max-repair-rounds",
            type=int,
            default=None,
            help="override the manifest's repair rounds",
        )
        parser.add_argument(
            "--gpu-budget-seconds",
            type=float,
            default=None,
            help="override the manifest's GPU budget",
        )
        parser.add_argument(
            "--token-budget", type=int, default=None, help="override the manifest's token budget"
        )
        parser.add_argument("--output", type=Path, default=Path("artifacts/alpha/run"))
        parser.add_argument(
            "--snapshot-root", type=Path, default=None, help="snapshot root (default: manifest)"
        )
        parser.add_argument("--gpu-device", default=None, help="CDI device (default: manifest)")
        parser.add_argument("--resume", action="store_true", default=True)
        return
    parser.add_argument("--problem", default="kernelbench:l1:40", help="e.g. kernelbench:l1:40")
    parser.add_argument("--backend", default="triton", help="candidate backend (triton)")
    parser.add_argument("--model", default="glm-4.5", help="provider model id")
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=os.environ.get("MODEL_PROVIDER_BASE_URL", ""),
        help="OpenAI-compatible base URL (or MODEL_PROVIDER_BASE_URL)",
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--gpu-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--token-budget", type=int, default=200000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/alpha/run"))
    parser.add_argument("--snapshot-root", type=Path, default=None)
    parser.add_argument("--gpu-device", default=None, help="CDI device, e.g. nvidia.com/gpu=GPU-…")
    parser.add_argument("--resume", action="store_true", default=resume_default)
    parser.add_argument(
        "--no-profile-baseline",
        dest="profile_baseline",
        action="store_false",
        default=True,
        help="Skip the baseline NCU profiling step (classification stays static_only)",
    )


def _resume_config_from_manifest(args: argparse.Namespace):
    """Build the resume OptimizationConfig: unspecified arguments restore
    the original values from the run manifest; explicit arguments override
    (and are re-validated against the manifest's identity inside optimize)."""
    from kernelagent.domain.errors import DomainError
    from kernelagent.domain.run_manifest import MANIFEST_FILENAME
    from kernelagent.domain.serialization import loads as domain_loads
    from kernelagent.optimization import OptimizationConfig, OptimizationConfigError

    output = Path(args.output)
    manifest_path = output / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise OptimizationConfigError(
            f"no run manifest at {manifest_path}: only runs started by this version "
            "can be resumed (start a new experiment)"
        )
    try:
        manifest = domain_loads(manifest_path.read_text(encoding="utf-8"))
    except DomainError as exc:
        raise OptimizationConfigError(
            f"run manifest at {manifest_path} is unreadable: {exc}"
        ) from exc
    if not hasattr(manifest, "model_id"):
        raise OptimizationConfigError(
            f"run manifest at {manifest_path} holds {type(manifest).__name__}, not RunManifest"
        )
    snapshot_root = args.snapshot_root
    if snapshot_root is None and manifest.snapshot_root:
        snapshot_root = Path(manifest.snapshot_root)
    return OptimizationConfig(
        problem=args.problem if args.problem is not None else manifest.problem,
        backend=args.backend if args.backend is not None else manifest.backend,
        model_id=args.model if args.model is not None else manifest.model_id,
        base_url=args.base_url if args.base_url is not None else manifest.base_url,
        max_candidates=(
            args.max_candidates if args.max_candidates is not None else manifest.max_candidates
        ),
        max_repair_rounds=(
            args.max_repair_rounds
            if args.max_repair_rounds is not None
            else manifest.max_repair_rounds
        ),
        gpu_budget_seconds=(
            args.gpu_budget_seconds
            if args.gpu_budget_seconds is not None
            else manifest.gpu_budget_seconds
        ),
        token_budget=args.token_budget if args.token_budget is not None else manifest.token_budget,
        output=output,
        resume=True,
        snapshot_root=snapshot_root,
        gpu_device=args.gpu_device if args.gpu_device is not None else manifest.gpu_device,
        profile_baseline=getattr(args, "profile_baseline", True),
    )


def _run_optimize(args: argparse.Namespace) -> int:
    from kernelagent.config import API_KEY_ENV
    from kernelagent.optimization import (
        OptimizationConfig,
        OptimizationConfigError,
        optimize,
        parse_exit_code,
        run_status,
    )

    if args.command == "status":
        try:
            print(json.dumps(run_status(args.output), indent=2, sort_keys=True))
            return 0
        except OptimizationConfigError as exc:
            print(f"config error: {exc}")
            return 3
    if args.command == "resume" and not args.resume:
        args.resume = True
    try:
        if args.resume:
            config = _resume_config_from_manifest(args)
            if not config.base_url:
                print(
                    "config error: the run manifest has no base_url and no --base-url "
                    "(or MODEL_PROVIDER_BASE_URL) was given"
                )
                return 3
        else:
            if not args.base_url:
                print(
                    "config error: --base-url (or MODEL_PROVIDER_BASE_URL) is required; "
                    f"the credential goes in {API_KEY_ENV}"
                )
                return 3
            config = OptimizationConfig(
                problem=args.problem,
                backend=args.backend,
                model_id=args.model,
                base_url=args.base_url,
                max_candidates=args.max_candidates,
                max_repair_rounds=args.max_repair_rounds,
                gpu_budget_seconds=args.gpu_budget_seconds,
                token_budget=args.token_budget,
                output=args.output,
                resume=args.resume,
                snapshot_root=args.snapshot_root,
                gpu_device=args.gpu_device,
                profile_baseline=getattr(args, "profile_baseline", True),
            )
        result = optimize(config)
    except OptimizationConfigError as exc:
        print(f"config error: {exc}")
        return 3
    print(
        f"state={result.state} champion={result.champion_sha256 or 'none'} "
        f"report={result.report_path}"
    )
    return parse_exit_code(result)


def _run_profile(args: argparse.Namespace) -> int:
    """`kernelagent profile`: collect (or reuse) baseline NCU evidence for a
    run directory and write it back - independent of the webapp."""
    from kernelagent.adapters.profiling.pipeline import (
        ProfileCapture,
        build_baseline_profiler,
    )
    from kernelagent.optimization import (
        OptimizationConfigError,
        RunLockHeldError,
        _sha256_text,
        acquire_run_lock,
        load_run_manifest,
        parse_problem_spec,
        resolve_problem,
        run_baseline_profile_for_run,
    )

    output = Path(args.output)
    try:
        manifest = load_run_manifest(output / "run_manifest.json")
        spec = parse_problem_spec(manifest.problem)
        snapshot_root = Path(manifest.snapshot_root)
        problem_path, problem_source = resolve_problem(spec, snapshot_root)
    except OptimizationConfigError as exc:
        print(f"config error: {exc}")
        return 3
    problem_sha256 = _sha256_text(problem_source)
    if problem_sha256 != manifest.problem_sha256:
        print(
            "config error: pinned problem changed since the run manifest "
            f"(manifest={manifest.problem_sha256[:12]}… current={problem_sha256[:12]}…); "
            "refusing to attach profile evidence to a different problem identity"
        )
        return 3
    metrics = (
        tuple(m.strip() for m in args.metrics.split(",") if m.strip()) if args.metrics else None
    )
    capture_kwargs = {
        "launch_count": args.launch_count,
        "launch_skip": args.launch_skip,
        "timeout_seconds": args.timeout_seconds,
        "ncu_bin": args.ncu_bin,
    }
    try:
        if metrics is not None:
            capture = ProfileCapture(set_name=None, metrics=metrics, **capture_kwargs)
        elif args.set_name:
            capture = ProfileCapture(set_name=args.set_name, metrics=None, **capture_kwargs)
        else:
            # Neither --set nor --metrics: the module default (explicit
            # Tier-A metric list, deterministic against MetricCatalog).
            capture = ProfileCapture(**capture_kwargs)
    except ValueError as exc:
        print(f"config error: {exc}")
        return 3
    gpu_device = args.gpu_device or manifest.gpu_device
    if not gpu_device:
        print("config error: no GPU device in the run manifest and no --gpu-device given")
        return 3
    try:
        lock = acquire_run_lock(output)
    except RunLockHeldError as exc:
        print(f"config error: {exc}")
        return 3
    try:
        profiler = build_baseline_profiler(
            profile_dir=output / "profile",
            workspace_root=output / "workspace",
            snapshot_root=snapshot_root,
            level=spec.level,
            problem_name=problem_path.name,
            problem_source=problem_source,
            gpu_device=gpu_device,
            capture=capture,
            problem_sha256=problem_sha256,
        )
        outcome = run_baseline_profile_for_run(
            output, profiler=profiler, problem_sha256=problem_sha256
        )
    finally:
        lock.release()
    if outcome is None:
        print(json.dumps({"status": "skipped", "reason": "see journal profile_not_run"}))
        return 1
    payload = {
        "status": outcome.get("status"),
        "reason": outcome.get("reason"),
        "detail": outcome.get("detail"),
        "summary": outcome.get("summary"),
        "gpu_wall_seconds": outcome.get("gpu_wall_seconds"),
        "report": f"profile/{outcome.get('report_path')}" if outcome.get("report_path") else None,
        "evidence": f"profile/{outcome.get('evidence_path')}"
        if outcome.get("evidence_path")
        else None,
        "stderr_tail": str(outcome.get("stderr_tail") or "")[-600:] or None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if outcome.get("status") == "collected" else 1


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
    optimize = commands.add_parser(
        "optimize", help="Optimize one pinned KernelBench problem (generate/eval/time/promote)"
    )
    _optimize_arguments(optimize)
    resume = commands.add_parser(
        "resume", help="Resume an interrupted optimize run from its durable journal"
    )
    _optimize_arguments(resume, resume_default=True)
    status = commands.add_parser("status", help="Show a run's durable state and budget")
    status.add_argument("--output", type=Path, required=True, help="Run output directory")
    profile = commands.add_parser(
        "profile",
        help="Collect baseline NCU evidence for a run directory (ADR-0003 diagnostic lease)",
    )
    profile.add_argument("--output", type=Path, required=True, help="Run output directory")
    profile.add_argument(
        "--kind",
        choices=["baseline"],
        default="baseline",
        help="What to profile; only the baseline reference is allowed (candidates are "
        "never profiled)",
    )
    profile.add_argument(
        "--set",
        dest="set_name",
        default=None,
        help="NCU section set (e.g. basic); default: the explicit Tier-A metric list "
        "mirroring MetricCatalog",
    )
    profile.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated explicit NCU metric list (overrides --set)",
    )
    profile.add_argument(
        "--launch-count",
        type=int,
        default=12,
        help="Cap on profiled kernel launches (default: 12)",
    )
    profile.add_argument(
        "--launch-skip", type=int, default=0, help="Kernel launches to skip first (default: 0)"
    )
    profile.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Hard wall-clock timeout for the profiling container (default: 300)",
    )
    profile.add_argument("--gpu-device", default=None, help="CDI device (default: manifest)")
    profile.add_argument(
        "--ncu-bin", default=None, help="Host ncu binary for --import (default: auto-detect)"
    )
    args = parser.parse_args(argv)
    if args.command == "bench":
        summary, exit_code = run_verify(args.root, args.manifest, args.dev_manifest)
        print(json.dumps(summary, indent=2))
        return exit_code
    if args.command == "probe":
        return _probe_command(args)
    if args.command in ("optimize", "resume", "status"):
        return _run_optimize(args)
    if args.command == "profile":
        return _run_profile(args)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    report = run_checks(args.tests, args.output, args.timeout)
    print(f"{report['status']}: {report['reason']}; report={report['report_path']}")
    return report["exit_code"]
