"""Evaluator tamper resistance (launch plan Task 1, ADR-0004).

The pinned upstream evaluator executes candidate source with ``exec()``
inside the evaluator's own Python process, so a candidate can rewrite the
very functions the verdict depends on (``torch.allclose``, CUDA events,
``json.dump``) or reach into the parent-owned ``/out`` channel. Docker
protects the host, not the evaluator's process. Until the trusted-MVP
trust-domain split lands, this surface is gated by
:func:`inspect_candidate_policy` - an AST misuse-prevention layer that is
explicitly NOT a security boundary: alpha runs declare
``candidate_trust=cooperative`` and ``adversarially_secure=false`` and
champions get human review.

Every attack below reproduces the fully-exec'd candidate source the
policy layer must flag before GPU time is spent on it."""

import json
from pathlib import Path

import pytest

from kernelagent.adapters.evals import EvalCase
from kernelagent.adapters.evals.kernelbench_eval import evaluate_case
from kernelagent.adapters.models.generation import (
    CandidatePolicyResult,
    inspect_candidate_policy,
)
from kernelagent.worker import WorkerOutcome

ALLCLOSE_PATCH = """import torch
import torch.nn as nn

torch.allclose = lambda *args, **kwargs: True

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.zeros_like(torch.matmul(A, B))
"""

CUDA_EVENT_PATCH = """import torch
import torch.nn as nn

torch.cuda.Event = lambda *args, **kwargs: None

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""

OUT_RESULT_WRITE = """import json

class ModelNew:
    def forward(self, x):
        with open("/out/result.json", "w") as handle:
            json.dump({"compiled": True, "correctness": True}, handle)
        return x
"""

JSON_DUMP_PATCH = """import json

json.dump = lambda *args, **kwargs: None

class ModelNew:
    def forward(self, x):
        return x
"""

DYNAMIC_EVAL = """code = "class ModelNew:\\n    pass\\n"
exec(code)

class ModelNew:
    def forward(self, x):
        return x
"""

RESTRICTED_IMPORTS = """import os
import subprocess

class ModelNew:
    def forward(self, x):
        subprocess.run(["true"])
        return x
"""

SETATTR_PATCH = """import torch

setattr(torch, "allclose", lambda *a, **k: True)

class ModelNew:
    def forward(self, x):
        return x
"""

BENIGN_CANDIDATE = """import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
"""


def _policy(source: str) -> CandidatePolicyResult:
    result = inspect_candidate_policy(source)
    assert isinstance(result, CandidatePolicyResult)
    assert isinstance(result.allowed, bool)
    assert isinstance(result.violations, tuple)
    assert all(isinstance(item, str) and item for item in result.violations)
    return result


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("torch.allclose monkeypatch", ALLCLOSE_PATCH),
        ("torch.cuda.Event monkeypatch", CUDA_EVENT_PATCH),
        ("parent /out write", OUT_RESULT_WRITE),
        ("json.dump monkeypatch", JSON_DUMP_PATCH),
        ("exec of dynamic code", DYNAMIC_EVAL),
        ("restricted module import", RESTRICTED_IMPORTS),
        ("setattr monkeypatch", SETATTR_PATCH),
    ],
)
def test_attack_sources_are_rejected_with_structured_reason(label, source):
    result = _policy(source)
    assert result.allowed is False, f"policy must reject: {label}"
    assert result.violations, f"policy must state why it rejected: {label}"


def test_benign_candidate_is_allowed():
    result = _policy(BENIGN_CANDIDATE)
    assert result.allowed is True
    assert result.violations == ()


def test_unparseable_candidate_is_rejected():
    result = _policy("class ModelNew(:\n  pass")
    assert result.allowed is False
    assert any("parse" in violation for violation in result.violations)


def _stage_case(tmp_path: Path, candidate_source: str) -> tuple[Path, Path]:
    snapshot = tmp_path / "snapshot"
    (snapshot / "src" / "kernelbench").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    for harness in ("dataset.py", "eval.py", "timing.py", "utils.py"):
        (snapshot / "src" / "kernelbench" / harness).write_text("# stub\n", encoding="utf-8")
    problem = snapshot / "KernelBench" / "level1" / "40_LayerNorm.py"
    problem.write_text("# problem stub\n", encoding="utf-8")
    return snapshot, problem


def test_eval_report_records_candidate_trust_fields(tmp_path, monkeypatch):
    """The evaluation report must carry the alpha threat-model markers:
    cooperative trust, the recorded policy violations, and an explicit
    adversarially_secure=false - never an implicit claim of strength."""
    snapshot, _ = _stage_case(tmp_path, ALLCLOSE_PATCH)
    out_dir = tmp_path / "run-out"
    (out_dir / "out").mkdir(parents=True)
    (out_dir / "out" / "result.json").write_text(
        json.dumps({"upstream": {"compiled": True, "correctness": True}}), encoding="utf-8"
    )
    outcome = WorkerOutcome(
        request_id="eval-tamper",
        status="completed",
        exit_code=0,
        duration_seconds=0.1,
        stdout_tail="",
        stderr_tail="",
        killed=False,
        workdir=str(out_dir),
    )
    seen = {}

    def fake_execute(request, spec, *, docker_command):
        seen["request_id"] = request.request_id
        return outcome

    monkeypatch.setattr(
        "kernelagent.adapters.evals.kernelbench_eval.execute_container", fake_execute
    )
    result = evaluate_case(
        EvalCase(
            case_id="tamper",
            level=1,
            problem_id=40,
            problem_name="40_LayerNorm.py",
            candidate_name="attacker",
            candidate_source=ALLCLOSE_PATCH,
        ),
        snapshot_root=snapshot,
        workspace_root=tmp_path / "ws",
        gpu_devices=("nvidia.com/gpu=GPU-fake",),
    )
    assert seen["request_id"].startswith("eval-")
    assert result.detail["candidate_trust"] == "cooperative"
    assert result.detail["adversarially_secure"] is False
    assert (
        tuple(result.detail["policy_violations"])
        == inspect_candidate_policy(ALLCLOSE_PATCH).violations
    )


def test_clean_candidate_report_still_marks_alpha_threat_model(tmp_path, monkeypatch):
    """adversarially_secure=false is a property of the evaluation chain, not
    of the individual candidate: clean candidates get the same marker."""
    snapshot, _ = _stage_case(tmp_path, BENIGN_CANDIDATE)
    out_dir = tmp_path / "run-out"
    (out_dir / "out").mkdir(parents=True)
    (out_dir / "out" / "result.json").write_text(
        json.dumps({"upstream": {"compiled": True, "correctness": True}}), encoding="utf-8"
    )
    outcome = WorkerOutcome(
        request_id="eval-clean",
        status="completed",
        exit_code=0,
        duration_seconds=0.1,
        stdout_tail="",
        stderr_tail="",
        killed=False,
        workdir=str(out_dir),
    )
    monkeypatch.setattr(
        "kernelagent.adapters.evals.kernelbench_eval.execute_container",
        lambda request, spec, *, docker_command: outcome,
    )
    result = evaluate_case(
        EvalCase(
            case_id="clean",
            level=1,
            problem_id=40,
            problem_name="40_LayerNorm.py",
            candidate_name="benign",
            candidate_source=BENIGN_CANDIDATE,
        ),
        snapshot_root=snapshot,
        workspace_root=tmp_path / "ws",
        gpu_devices=("nvidia.com/gpu=GPU-fake",),
    )
    assert result.detail["candidate_trust"] == "cooperative"
    assert tuple(result.detail["policy_violations"]) == ()
    assert result.detail["adversarially_secure"] is False
