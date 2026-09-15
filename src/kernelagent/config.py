"""Runtime configuration and real-port wiring for the optimization loop
(launch plan Task 5).

Secrets: the provider credential is read once from the
``MODEL_PROVIDER_API_KEY`` environment variable and stays on the control
plane - it never enters the journal, reports, or any worker container.
Offline loops (tests) inject their own generator and never touch it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kernelagent.adapters.evals import (
    EvalCase,
    TimingCase,
    TimingProtocol,
    evaluate_case,
    run_pro_case,
    run_timing_case,
)
from kernelagent.adapters.evals.correctness_pro import overall_verdict
from kernelagent.adapters.models.costing import CostLedger
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.adapters.models.openai_compat import OpenAICompatModelClient
from kernelagent.optimization import (
    API_KEY_ENV,
    CORRECTNESS_PRO_TIMEOUT_SECONDS,
    EVALUATE_TIMEOUT_SECONDS,
    TIMING_TIMEOUT_SECONDS,
    OptimizationConfig,
    OptimizationConfigError,
    ProblemSpec,
)
from kernelagent.promotion import CandidateFacts, confirm_promotion

# API_KEY_ENV is re-exported from kernelagent.optimization: the credential
# env var name lives with the optimization entry point so the run manifest
# and the runtime agree on it (the value itself is only ever read from the
# environment, never stored).


def default_generator(base_url: str, extra_body: dict | None = None) -> CandidateGenerator:
    """Real generation port: OpenAI-compatible transport + cost ledger.

    Missing credentials are a permanent, user-fixable configuration
    error - never a silent fallback to recorded responses."""
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key.strip():
        raise OptimizationConfigError(
            f"environment variable {API_KEY_ENV} is not set: live generation requires "
            "provider credentials on the control plane"
        )
    client = OpenAICompatModelClient(base_url=base_url, api_key=api_key, extra_body=extra_body)
    return CandidateGenerator(client, CostLedger())


@dataclass(frozen=True, slots=True)
class RealStagePorts:
    """Real T05/T06/T07/T09 adapters bound to one pinned problem."""

    evaluate: object
    correctness_pro: object
    timing: object
    confirm: object
    protocol: TimingProtocol


def _stage_timeout(default_seconds: float, deadline_seconds: float | None) -> float:
    """Effective container timeout for one GPU stage (RV04): the stage's
    configured timeout capped by the remaining-budget deadline when one
    was passed - ``min(own timeout, deadline)``. ``None`` (offline fakes
    and any caller that does not meter) keeps the configured timeout."""
    if deadline_seconds is None:
        return default_seconds
    return min(default_seconds, max(0.0, float(deadline_seconds)))


def build_stage_ports(
    config: OptimizationConfig,
    spec: ProblemSpec,
    problem_path: Path,
    snapshot_root: Path,
    gpu_device: str,
) -> RealStagePorts:
    snapshot_root = Path(snapshot_root)
    workspace = Path(config.output) / "workspace"
    protocol = TimingProtocol()

    def evaluate(
        candidate_source: str, candidate_name: str, *, deadline_seconds: float | None = None
    ) -> dict:
        result = evaluate_case(
            EvalCase(
                case_id=candidate_name,
                level=spec.level,
                problem_id=spec.problem_id,
                problem_name=problem_path.name,
                candidate_name=candidate_name,
                candidate_source=candidate_source,
                backend=config.backend,
            ),
            snapshot_root=snapshot_root,
            workspace_root=workspace,
            gpu_devices=(gpu_device,),
            timeout_seconds=_stage_timeout(EVALUATE_TIMEOUT_SECONDS, deadline_seconds),
        )
        return {
            "adapter_pass": result.adapter_pass,
            "compiled": result.upstream_compiled,
            "correct": result.upstream_correctness,
            "outcome_status": result.outcome_status,
            "stderr_tail": result.detail.get("stderr_tail", ""),
            "upstream_metadata": result.detail.get("upstream_metadata", {}),
        }

    def correctness_pro(
        candidate_source: str, candidate_name: str, *, deadline_seconds: float | None = None
    ) -> tuple[bool, str]:
        pro = run_pro_case(
            case_id=candidate_name,
            level=spec.level,
            problem_id=spec.problem_id,
            problem_name=problem_path.name,
            candidate_name=candidate_name,
            candidate_source=candidate_source,
            snapshot_root=snapshot_root,
            workspace_root=workspace,
            gpu_devices=(gpu_device,),
            backend=config.backend,
            timeout_seconds=_stage_timeout(CORRECTNESS_PRO_TIMEOUT_SECONDS, deadline_seconds),
        )
        return overall_verdict(pro)

    def timing(candidate_source: str, role: str, *, deadline_seconds: float | None = None) -> dict:
        is_candidate = role == "candidate"
        result = run_timing_case(
            TimingCase(
                case_id=f"{role}-timing",
                level=spec.level,
                problem_id=spec.problem_id,
                problem_name=problem_path.name,
                candidate_name=problem_path.stem if role == "eager" else "candidate",
                candidate_source=candidate_source,
                role=role,
                backend=config.backend if is_candidate else "cuda",
            ),
            protocol,
            snapshot_root=snapshot_root,
            workspace_root=workspace,
            gpu_devices=(gpu_device,),
            timeout_seconds=_stage_timeout(TIMING_TIMEOUT_SECONDS, deadline_seconds),
        )
        return {
            "source_ok": result.source_ok,
            "precondition": result.correctness_precondition,
            "async_leak": result.async_leak,
            "batches": list(result.batch_samples_ms),
        }

    def confirm(facts: CandidateFacts, incumbent: list[float], candidate: list[float]):
        return confirm_promotion(
            facts=facts,
            incumbent_batches_ms=incumbent,
            candidate_batches_ms=candidate,
            protocol_sha256=protocol.identity_sha256(),
        )

    return RealStagePorts(
        evaluate=evaluate,
        correctness_pro=correctness_pro,
        timing=timing,
        confirm=confirm,
        protocol=protocol,
    )
