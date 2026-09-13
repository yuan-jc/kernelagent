"""Static code features and operator-class inference (generator-design §1.2).

The generator is an evidence consumer. Its cheapest always-available evidence
is the reference ``problem_source`` itself: an AST walk extracts structural
signals (kernel-chain length, host allocations, tensor-core usage, dtype,
triangular masks, autotune presence). No source is executed and nothing is
invented: a source that does not parse degrades to ``parses=False`` with all
features empty - that is honest MISSING, not a zero.

The same module derives the :class:`~kernelagent.methods.TaskProfile` the
optimization loop needs from the pinned problem file name; the category is
an inference, and ``unknown`` is a legal, honest outcome.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from kernelagent.methods import TaskProfile

# Operator classes are the generator's coarse prior vocabulary (design §2.2
# prior table). Ordered by specificity: the first matching class wins.
_OPERATOR_CLASS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("attention", "sdpa", "flash_attn")),
    ("softmax", ("softmax", "log_softmax", "cross_entropy", "crossentropy")),
    ("rmsnorm", ("rmsnorm", "rms_norm")),
    (
        "layernorm",
        ("layernorm", "layer_norm", "groupnorm", "group_norm", "batchnorm", "batch_norm"),
    ),
    ("conv", ("conv",)),
    ("matmul", ("matmul", "gemm", "einsum", "bmm", "linear")),
    ("embedding", ("embedding", "gather", "scatter", "index_select", "one_hot")),
    (
        "reduction",
        ("reduction", "cumsum", "cumprod", "argmax", "argmin", "var", "std", "mean", "sum"),
    ),
    (
        "elementwise",
        (
            "gelu",
            "relu",
            "sigmoid",
            "silu",
            "mish",
            "softplus",
            "tanh",
            "clamp",
            "exp",
            "log",
            "pow",
            "sqrt",
            "activation",
            "elementwise",
        ),
    ),
)

_DTYPE_TOKENS: tuple[str, ...] = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int64",
    "int32",
    "int8",
)

_HOST_FACTORIES = frozenset(
    {"zeros", "ones", "empty", "full", "tensor", "randn", "rand", "randint", "arange"}
)
_SYNC_CALLS = frozenset({"item", "cpu", "nonzero", "numpy", "tolist"})
_MASK_CALLS = frozenset({"triu", "tril", "masked_fill", "where"})
_TRANSCENDENTAL_CALLS = frozenset(
    {"exp", "log", "log2", "exp2", "pow", "sqrt", "rsqrt", "sin", "cos"}
)
_PROBLEM_ID_PREFIX = re.compile(r"^\d+_")


@dataclass(frozen=True, slots=True)
class CodeFeatures:
    """Structural signals extracted from the reference problem source."""

    parses: bool = False
    uses_triton: bool = False
    has_tl_dot: bool = False
    has_autotune: bool = False
    forward_call_count: int = 0
    host_allocations: int = 0
    has_sync_call: bool = False
    has_triangular_mask: bool = False
    uses_transcendental: bool = False
    uses_atomic: bool = False
    dtypes: tuple[str, ...] = ()
    op_hints: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        """True when nothing structural was extractable (unparseable source)."""
        return not self.parses


def _dotted_names(tree: ast.AST) -> set[str]:
    """All dotted attribute/name paths in the module (e.g. ``tl.dot``,
    ``torch.nn.functional.softmax``) - static text evidence, never execution."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts: list[str] = [node.attr]
            cursor: ast.AST = node.value
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
                names.add(".".join(reversed(parts)))
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _call_dotted(node: ast.Call) -> str:
    cursor: ast.AST = node.func
    parts: list[str] = []
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    return ".".join(reversed(parts))


def _forward_node(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "forward":
            return node
    return None


def extract_code_features(problem_source: str) -> CodeFeatures:
    """Static, execution-free feature extraction from the reference source.

    A source that fails to parse yields ``parses=False`` with zeroed
    features: downstream classification must then rely on the task name
    prior and say so (coverage degrades, signals are not invented)."""
    try:
        tree = ast.parse(problem_source)
    except SyntaxError:
        return CodeFeatures(parses=False)

    names = _dotted_names(tree)
    lowered = {name.lower() for name in names}
    joined = " ".join(lowered)

    uses_triton = any("triton" in name for name in lowered)
    has_tl_dot = "tl.dot" in lowered
    has_autotune = "triton.autotune" in lowered
    uses_atomic = "atomic" in joined

    op_hints: set[str] = set()
    for hint, tokens in _OPERATOR_CLASS_RULES:
        if any(token in name for name in lowered for token in tokens):
            op_hints.add(hint)
    if any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult) for node in ast.walk(tree)
    ):
        op_hints.add("matmul")

    dtypes = tuple(token for token in _DTYPE_TOKENS if token in joined)

    forward = _forward_node(tree)
    call_region: ast.AST = forward if forward is not None else tree
    forward_call_count = 0
    host_allocations = 0
    has_sync_call = False
    has_triangular_mask = False
    uses_transcendental = False
    for node in ast.walk(call_region):
        if not isinstance(node, ast.Call):
            continue
        forward_call_count += 1
        dotted = _call_dotted(node)
        leaf = dotted.rsplit(".", 1)[-1].lower()
        if dotted.lower().startswith("torch") and any(
            leaf == factory or leaf.startswith(factory + "_") for factory in _HOST_FACTORIES
        ):
            host_allocations += 1
        if leaf in _SYNC_CALLS:
            has_sync_call = True
        if leaf in _MASK_CALLS:
            has_triangular_mask = True
        if leaf in _TRANSCENDENTAL_CALLS:
            uses_transcendental = True

    return CodeFeatures(
        parses=True,
        uses_triton=uses_triton,
        has_tl_dot=has_tl_dot,
        has_autotune=has_autotune,
        forward_call_count=forward_call_count,
        host_allocations=host_allocations,
        has_sync_call=has_sync_call,
        has_triangular_mask=has_triangular_mask,
        uses_transcendental=uses_transcendental,
        uses_atomic=uses_atomic,
        dtypes=dtypes,
        op_hints=frozenset(op_hints),
    )


def infer_operator_class(features: CodeFeatures, task: TaskProfile) -> str:
    """Coarse operator-class prior from AST hints, then task name/category.

    Priority order resolves mixed problems (``layernorm`` beats the generic
    ``norm``/reduction tokens). ``unknown`` is a legal outcome and keeps the
    planner honest on problems the prior table does not cover."""
    haystack = " ".join(sorted(features.op_hints)) + " " + task.category + " " + task.problem_name
    haystack = haystack.lower().replace("-", "_")
    for cls, tokens in _OPERATOR_CLASS_RULES:
        if any(token in haystack for token in tokens):
            return cls
    return "unknown"


def _category_from_name(problem_name: str) -> str:
    name = _PROBLEM_ID_PREFIX.sub("", Path(problem_name).stem).lower().replace("-", "_")
    for cls, tokens in _OPERATOR_CLASS_RULES:
        if any(token in name for token in tokens):
            return cls
    return "unknown"


def task_profile_from_problem(problem_name: str, level: int, problem_id: int) -> TaskProfile:
    """TaskProfile for the loop: the category is name-inferred and may be
    ``unknown`` - it is an input prior for the generator, never a verdict."""
    return TaskProfile(
        task_id=f"kernelbench-l{level}-p{problem_id:03d}",
        level=level,
        problem_id=problem_id,
        problem_name=problem_name,
        category=_category_from_name(problem_name),
    )
