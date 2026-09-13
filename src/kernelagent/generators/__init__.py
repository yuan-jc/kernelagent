"""Optimization method generator (design §6, generator-design MVP).

Plan layer between evidence and candidate generation: profile evidence,
task/code features and structured attempt history are classified into a
bottleneck, matched against the frozen 38-method catalog
(``research/optimization-methods/methods.yaml`` via the frozen package
copy), ranked with explicit refusals, and rendered into single-factor
prompt fragments that the optimization loop injects through the existing
``seed_note`` channel.

Trust boundary (AGENTS, generator-design §4): the generator only PROPOSES.
Correctness, timing and promotion stay with the trusted evaluator/promotion
path; ``expected_gain`` is an uncalibrated ranking prior, never a claim.
Pure offline code: no torch, no triton, no model SDK, no network.
"""

from kernelagent.generators.catalog import (
    CatalogError,
    MethodCatalog,
    MethodEntry,
    load_default,
)
from kernelagent.generators.classify import (
    BottleneckClass,
    classify_bottleneck,
)
from kernelagent.generators.features import (
    CodeFeatures,
    extract_code_features,
    infer_operator_class,
    task_profile_from_problem,
)
from kernelagent.generators.input import (
    AttemptRecord,
    GeneratorInput,
    HardwareFacts,
    attempt_from_record,
)
from kernelagent.generators.plan import (
    MethodPlan,
    MethodPlanItem,
    MethodPlanner,
    PlanRefusal,
)
from kernelagent.generators.prompt_fragments import compose_seed_note, render_plan_fragment

__all__ = [
    "AttemptRecord",
    "BottleneckClass",
    "CatalogError",
    "CodeFeatures",
    "GeneratorInput",
    "HardwareFacts",
    "MethodCatalog",
    "MethodEntry",
    "MethodPlan",
    "MethodPlanItem",
    "MethodPlanner",
    "PlanRefusal",
    "attempt_from_record",
    "classify_bottleneck",
    "compose_seed_note",
    "extract_code_features",
    "infer_operator_class",
    "load_default",
    "render_plan_fragment",
    "task_profile_from_problem",
]
