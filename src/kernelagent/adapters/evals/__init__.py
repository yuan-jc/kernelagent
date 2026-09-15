"""Evaluation adapters (control plane): pinned upstream evaluators, the
extended correctness track, and the formal timing protocol, driven inside
the T04 container boundary. See ADR-0002 and design §9.1-9.3."""

from kernelagent.adapters.evals.correctness_pro import (
    overall_verdict,
    run_pro_case,
)
from kernelagent.adapters.evals.kernelbench_eval import (
    EVAL_IMAGE_ID,
    EVAL_IMAGE_REPO,
    EvalCase,
    EvalCaseResult,
    derive_upstream_verdict,
    evaluate_case,
    extract_upstream_diagnostics,
)
from kernelagent.adapters.evals.timing import (
    SOURCE_TAG,
    TimingCase,
    TimingCaseResult,
    TimingProtocol,
    bootstrap_ratio_ci,
    run_timing_case,
    validate_batch_samples,
    validate_timing_payload,
)

__all__ = [
    "EVAL_IMAGE_ID",
    "EVAL_IMAGE_REPO",
    "EvalCase",
    "EvalCaseResult",
    "SOURCE_TAG",
    "TimingCase",
    "TimingCaseResult",
    "TimingProtocol",
    "bootstrap_ratio_ci",
    "derive_upstream_verdict",
    "evaluate_case",
    "extract_upstream_diagnostics",
    "overall_verdict",
    "run_pro_case",
    "run_timing_case",
    "validate_batch_samples",
    "validate_timing_payload",
]
