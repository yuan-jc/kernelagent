"""Prompt-fragment rendering for method plan items (generator-design §3.4).

A fragment is the single, self-contained instruction block injected into a
candidate-generation request through the existing ``seed_note`` channel of
``adapters/models/generation.py::build_request``. Two rules are structural:

- SINGLE FACTOR: one fragment authorizes exactly ONE target change relative
  to the parent source (meta.one_factor_per_round). Correctness guards ride
  along in the GUARDS block without consuming the "one factor" budget
  (README conclusion 9).
- NO SELF-GRADED PERFORMANCE: the candidate must never measure or report
  its own speed; formal timing belongs to the trusted evaluator
  (tuning.do_bench_internal_protocol, design §9.3).

Rendering is a pure function of the plan item - same input, same bytes.
"""

from __future__ import annotations

import json

GUARD_LINES: tuple[str, ...] = (
    "- GUARD Mask every tl.load/tl.store against block/tail boundaries; pick neutral "
    "'other' values for reductions (numerics.boundary_masking).",
    "- GUARD Keep fp32 accumulators for fp16/bf16 reductions and tl.dot; cast back "
    "only at the final store (numerics.fp32_accumulators).",
    "- GUARD Do not measure or report performance yourself; never call "
    "triton.testing.do_bench inside forward; formal timing is the evaluator's job "
    "(tuning.do_bench_internal_protocol).",
)


def render_plan_fragment(
    *,
    method_id: str,
    statement: str,
    requirements: tuple[str, ...],
    parameter_space: tuple[dict, ...],
    evidence_coverage: str,
    bottleneck_label: str,
) -> str:
    """Render one plan item into the fixed §3.4 fragment shape."""
    lines: list[str] = [
        f"OPTIMIZATION PLAN [{method_id}] "
        f"(bottleneck: {bottleneck_label}; evidence coverage: {evidence_coverage})",
        "TARGET CHANGE (apply ONLY this change relative to the parent source):",
        f"- {statement}",
        "REQUIREMENTS:",
    ]
    lines.extend(f"- {line}" for line in requirements)
    lines.append("GUARDS (mandatory):")
    lines.extend(GUARD_LINES)
    if parameter_space:
        lines.append("AUTOTUNE SPACE (budget-bounded; keep every config inside this list):")
        lines.extend(
            f"- config: {json.dumps(config, sort_keys=True)}" for config in parameter_space
        )
    else:
        lines.append(
            "AUTOTUNE SPACE: none for this change "
            "(structural/index-level edit, not a config search)."
        )
    return "\n".join(lines) + "\n"


def compose_seed_note(feedback: str, fragment: str | None) -> str:
    """Seed-note assembly: persisted failure feedback first, then exactly the
    one plan fragment for this round (or neither - both optional). The result
    feeds build_request(seed_note=...) so it is covered by request_sha256
    identity; the hash computation itself is NOT changed by this module."""
    parts = [part for part in (feedback, fragment) if part]
    return "\n\n".join(parts)
