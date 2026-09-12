"""Method registry and generators (T13, design §6.1-6.3).

A Method bundles a generator with its provenance: what kind of task it
applies to, its license, and the dependencies its candidates may use.
The registry is the only way methods reach the main loop, and it
enforces three gates before any candidate exists:

- applicability: a method whose domain does not match the task is
  refused with an explicit reason (never silently skipped into a
  different task's slot);
- dependency whitelist: a candidate whose source imports dependencies
  outside the policy whitelist is refused;
- license gating: methods whose license is not accepted by the active
  policy are refused.

Generated candidates carry their provenance (method name, license, deps)
and are ALWAYS re-verified by the trusted evaluator afterwards - a
generator's own claim is not correctness (that is T05's job)."""

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """Minimal task descriptor methods match against."""

    task_id: str
    level: int
    problem_id: int
    problem_name: str
    category: str  # e.g. "layernorm", "matmul", "conv2d_biasadd"


@dataclass(frozen=True, slots=True)
class MethodPolicy:
    """Active acceptance policy: licenses and importable dependencies."""

    allowed_licenses: frozenset[str] = frozenset({"MIT", "BSD-3", "Apache-2.0", "project-original"})
    allowed_dependencies: frozenset[str] = frozenset({"torch", "triton", "numpy"})


@dataclass(frozen=True, slots=True)
class Method:
    name: str
    category: str  # task category this method applies to
    license: str
    dependencies: frozenset[str]
    generator: object  # Callable[[TaskProfile, str], str] -> candidate source

    def applies_to(self, task: TaskProfile) -> bool:
        return self.category == task.category


@dataclass(frozen=True, slots=True)
class MethodCandidate:
    method_name: str
    license: str
    dependencies: tuple[str, ...]
    candidate_source: str
    candidate_sha256: str


@dataclass(frozen=True, slots=True)
class MethodRefusal:
    method_name: str
    reason: str


class MethodRegistry:
    def __init__(self, policy: MethodPolicy | None = None):
        self._methods: dict[str, Method] = {}
        self.policy = policy or MethodPolicy()

    def register(self, method: Method) -> None:
        if method.name in self._methods:
            raise ValueError(f"method {method.name!r} already registered")
        if method.license not in self.policy.allowed_licenses:
            raise ValueError(
                f"method {method.name!r} license {method.license!r} is not accepted by policy"
            )
        self._methods[method.name] = method

    def methods(self) -> tuple[Method, ...]:
        return tuple(self._methods[name] for name in sorted(self._methods))

    def select(self, task: TaskProfile) -> tuple[list[Method], list[MethodRefusal]]:
        selected: list[Method] = []
        refusals: list[MethodRefusal] = []
        for method in self.methods():
            if method.applies_to(task):
                selected.append(method)
            else:
                refusals.append(
                    MethodRefusal(
                        method_name=method.name,
                        reason=(
                            f"method domain {method.category!r} does not match task "
                            f"category {task.category!r}"
                        ),
                    )
                )
        return selected, refusals


def imported_dependencies(candidate_source: str) -> frozenset[str]:
    """Top-level module names imported by a candidate source."""
    tree = ast.parse(candidate_source)
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            deps.add(node.module.split(".")[0])
    return frozenset(deps)


def run_method(
    method: Method, task: TaskProfile, problem_source: str, *, policy: MethodPolicy | None = None
) -> tuple[MethodCandidate | None, MethodRefusal | None]:
    """One method -> one candidate (or an explicit refusal). Dependencies
    outside the whitelist refuse the candidate after generation."""
    active = policy or MethodPolicy()
    outside = frozenset(d for d in method.dependencies if d not in active.allowed_dependencies)
    if outside:
        return None, MethodRefusal(
            method_name=method.name,
            reason=f"dependencies outside whitelist: {sorted(outside)}",
        )
    try:
        candidate_source = method.generator(task, problem_source)
    except (LookupError, KeyError) as exc:
        return (
            None,
            MethodRefusal(method_name=method.name, reason=f"generator refused task: {exc}"),
        )
    try:
        ast.parse(candidate_source)
    except SyntaxError as exc:
        return None, MethodRefusal(
            method_name=method.name,
            reason=f"generator produced unparseable source: {exc}",
        )
    deps = imported_dependencies(candidate_source)
    outside_used = frozenset(d for d in deps if d not in active.allowed_dependencies)
    if outside_used:
        return None, MethodRefusal(
            method_name=method.name,
            reason=f"candidate imports dependencies outside whitelist: {sorted(outside_used)}",
        )
    import hashlib

    digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
    return (
        MethodCandidate(
            method_name=method.name,
            license=method.license,
            dependencies=tuple(sorted(deps)),
            candidate_source=candidate_source,
            candidate_sha256=digest,
        ),
        None,
    )


def make_template_method(
    name: str,
    category: str,
    license: str,
    template: str,
) -> Method:
    """Template method: renders a frozen template with task parameters.
    The template must define a ``render(problem_source) -> str`` hook via
    plain str.format with ``problem_source`` available."""

    def generator(task: TaskProfile, problem_source: str) -> str:
        return template.format(problem_source=problem_source, task_id=task.task_id)

    return Method(
        name=name,
        category=category,
        license=license,
        dependencies=frozenset({"torch"}),
        generator=generator,
    )


def make_reuse_method(
    name: str,
    category: str,
    license: str,
    library: dict[str, str],
) -> Method:
    """Reuse method: selects a known-good candidate from a frozen library
    keyed by task id; reuse never skips re-verification."""

    def generator(task: TaskProfile, problem_source: str) -> str:
        if task.task_id not in library:
            raise LookupError(f"reuse library has no candidate for {task.task_id!r}")
        return library[task.task_id]  # reuse is self-contained

    return Method(
        name=name,
        category=category,
        license=license,
        dependencies=frozenset({"torch"}),
        generator=generator,
    )
