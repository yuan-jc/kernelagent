"""Domain dependency boundary: no torch/Triton/model SDK/adapter imports.

The scanner is exercised against a synthetic violating module so a vacuous
checker cannot pass this file by accident.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

DOMAIN_DIR = Path(__file__).resolve().parents[1] / "src" / "kernelagent" / "domain"

FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "transformers",
        "openai",
        "anthropic",
        "google",
        "zhipuai",
        "dashscope",
        "mistralai",
        "cohere",
    }
)
# Domain may only import from its own package plus the standard library.
FORBIDDEN_INTERNAL_PREFIX = "kernelagent."
ALLOWED_INTERNAL_PREFIX = "kernelagent.domain"


def imported_top_level_modules(source: str) -> set[str]:
    """Return top-level module names imported by ``source`` (AST, not execution)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def imported_internal_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("kernelagent"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("kernelagent"):
                names.add(node.module)
    return names


def test_scanner_flags_synthetic_violation():
    """Negative control: the scanner itself must detect a known violation."""
    bad = "import torch\nfrom triton import autotune\nfrom kernelagent.adapters import x\n"
    top_level = imported_top_level_modules(bad)
    assert {"torch", "triton"} <= top_level
    assert imported_internal_modules(bad) == {"kernelagent.adapters"}


def test_domain_files_exist():
    files = sorted(path.name for path in DOMAIN_DIR.glob("*.py"))
    assert "__init__.py" in files
    assert len(files) >= 8


def test_domain_imports_no_forbidden_top_level_modules():
    violations: dict[str, set[str]] = {}
    for path in sorted(DOMAIN_DIR.glob("*.py")):
        found = imported_top_level_modules(path.read_text(encoding="utf-8")) & FORBIDDEN_TOP_LEVEL
        if found:
            violations[path.name] = found
    assert not violations, f"domain imports forbidden modules: {violations}"


def test_domain_imports_no_layers_outside_domain():
    violations: dict[str, set[str]] = {}
    for path in sorted(DOMAIN_DIR.glob("*.py")):
        found = {
            module
            for module in imported_internal_modules(path.read_text(encoding="utf-8"))
            if module.startswith(FORBIDDEN_INTERNAL_PREFIX)
            and not module.startswith(ALLOWED_INTERNAL_PREFIX)
        }
        if found:
            violations[path.name] = found
    assert not violations, f"domain imports outside kernelagent.domain: {violations}"


def test_domain_import_leaves_forbidden_modules_unloaded():
    code = (
        "import sys, kernelagent.domain; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=DOMAIN_DIR.parents[2]
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(
    "source",
    ["import torch", "from torch import nn", "import triton.language as tl"],
)
def test_parametrized_forbidden_imports_flagged(source):
    assert imported_top_level_modules(source) & FORBIDDEN_TOP_LEVEL
