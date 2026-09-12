"""Evaluation adapters (control plane): pinned upstream evaluators driven
inside the T04 container boundary. See ADR-0002."""

from kernelagent.adapters.evals.kernelbench_eval import (
    EVAL_IMAGE_ID,
    EVAL_IMAGE_REPO,
    EvalCase,
    EvalCaseResult,
    derive_upstream_verdict,
    evaluate_case,
)

__all__ = [
    "EVAL_IMAGE_ID",
    "EVAL_IMAGE_REPO",
    "EvalCase",
    "EvalCaseResult",
    "derive_upstream_verdict",
    "evaluate_case",
]
