"""Method registry and generators (T13): pure control-plane semantics.
Candidates produced here are re-verified by the T05 evaluator on GPU in
examples/methods_smoke.py; these tests pin the gates."""

import pytest

from kernelagent.methods import (
    Method,
    MethodPolicy,
    MethodRegistry,
    TaskProfile,
    imported_dependencies,
    make_reuse_method,
    make_template_method,
    run_method,
)

LAYERNORM_TASK = TaskProfile(
    task_id="kernelbench-l1-p040",
    level=1,
    problem_id=40,
    problem_name="40_LayerNorm.py",
    category="layernorm",
)

TEMPLATE = (
    "import torch\nimport torch.nn as nn\n"
    "class ModelNew(nn.Module):\n"
    "    def __init__(self, normalized_shape: tuple):\n"
    "        super(ModelNew, self).__init__()\n"
    "        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)\n"
    "    def forward(self, x):\n"
    "        return self.ln(x)\n"
)


def _constant_template(category: str, license: str = "MIT"):
    def generator(task, problem_source):
        return TEMPLATE

    return Method(
        name=f"template-{category}",
        category=category,
        license=license,
        dependencies=frozenset({"torch"}),
        generator=generator,
    )


def test_registry_rejects_unknown_license():
    registry = MethodRegistry()
    with pytest.raises(ValueError, match="not accepted by policy"):
        registry.register(
            Method(
                name="nvidia-eula-method",
                category="layernorm",
                license="nvidia-eula",
                dependencies=frozenset({"torch"}),
                generator=lambda task, src: "",
            )
        )


def test_registry_rejects_duplicate_names():
    registry = MethodRegistry()
    registry.register(_constant_template("layernorm"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_constant_template("layernorm"))


def test_domain_mismatch_refuses_with_reason():
    registry = MethodRegistry()
    registry.register(_constant_template("layernorm"))
    conv_task = TaskProfile(
        task_id="kernelbench-l2-p001",
        level=2,
        problem_id=1,
        problem_name="1_Conv2D_ReLU_BiasAdd.py",
        category="conv2d_biasadd",
    )
    selected, refusals = registry.select(conv_task)
    assert selected == []
    assert len(refusals) == 1
    assert "layernorm" in refusals[0].reason


def test_dependency_whitelist_blocks_candidate_imports():
    policy = MethodPolicy(allowed_dependencies=frozenset({"torch"}))
    registry = MethodRegistry(policy)
    flashinfer_method = Method(
        name="flashinfer-template",
        category="layernorm",
        license="MIT",
        dependencies=frozenset({"flashinfer"}),
        generator=lambda task, src: TEMPLATE,
    )
    registry.register(flashinfer_method)
    task = TaskProfile("kernelbench-l1-p040", 1, 40, "40_LayerNorm.py", "layernorm")
    candidate, refusal = run_method(registry.methods()[0], task, "source", policy=policy)
    assert candidate is None
    assert refusal is not None and "outside whitelist" in refusal.reason


def test_candidate_importing_outside_whitelist_refused_after_generation():
    policy = MethodPolicy(allowed_dependencies=frozenset({"torch"}))
    sneaky = Method(
        name="sneaky",
        category="layernorm",
        license="MIT",
        dependencies=frozenset({"torch"}),
        generator=lambda task, src: "import flashinfer\nimport torch\n",
    )
    candidate, refusal = run_method(sneaky, LAYERNORM_TASK, "source", policy=policy)
    assert candidate is None
    assert "flashinfer" in refusal.reason


def test_reuse_method_requires_exact_task_entry():
    library = {"kernelbench-l1-p040": TEMPLATE}
    method = make_reuse_method("reuse-ln", "layernorm", "project-original", library)
    candidate, refusal = run_method(method, LAYERNORM_TASK, "src")
    assert candidate is not None and "ModelNew" in candidate.candidate_source
    other = TaskProfile("kernelbench-l1-p023", 1, 23, "23_Softmax.py", "layernorm")
    candidate, refusal = run_method(method, other, "src")
    assert candidate is None
    assert "no candidate" in refusal.reason


def test_template_method_renders_problem_identity():
    method = make_template_method("tmpl-ln", "layernorm", "MIT", TEMPLATE + "\n# {task_id}\n")
    candidate, refusal = run_method(method, LAYERNORM_TASK, "src")
    assert refusal is None
    assert "kernelbench-l1-p040" in candidate.candidate_source
    assert candidate.candidate_sha256 != candidate.candidate_source


def test_imported_dependencies_top_level_modules():
    assert imported_dependencies("import torch\nfrom triton import jit\nimport a.b.c\n") == {
        "torch",
        "triton",
        "a",
    }
