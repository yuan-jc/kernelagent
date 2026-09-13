"""Minimal optimization loop (launch plan Task 5, T16).

Offline, fully-injected vertical-loop tests: generation is driven by a
scripted responder through the REAL CandidateGenerator, evaluation /
timing are fake ports, and promotion is the REAL confirm_promotion. The
real GPU wiring is exercised by examples (Task 6); a unit test cannot
substitute for it."""

import json
from pathlib import Path

import pytest

from kernelagent.adapters.models.client import ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.optimization import (
    EXIT_BUDGET_EXHAUSTED,
    EXIT_CONFIG_ERROR,
    EXIT_INFRA_ERROR,
    EXIT_NO_IMPROVEMENT,
    EXIT_SUCCESS,
    OptimizationConfig,
    OptimizationConfigError,
    optimize,
    parse_exit_code,
)


class ScriptedResponder:
    """Order-based model double that builds each response against the
    actual request (so hashes validate) and records every request seen."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        content = self._contents.pop(0)
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=content,
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20),
        )


class RecordingLedger:
    def __init__(self):
        self._total_tokens = 0

    def record(self, response):
        self._total_tokens += response.usage.total_tokens

    def total_tokens(self) -> int:
        return self._total_tokens


def candidate_module(body: str) -> str:
    return json.dumps({"code": f"class ModelNew:\n    {body}\n"})


GOOD_CANDIDATE = candidate_module("def forward(self, x):\n        return x")
BAD_CANDIDATE = candidate_module("def forward(self, x):\n        return 0")


def _stage_problem(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text(
        "# pinned problem\n", encoding="utf-8"
    )
    return snapshot


def _passing_evaluate(candidate_source: str, candidate_name: str) -> dict:
    passed = "return x" in candidate_source
    return {
        "adapter_pass": passed,
        "compiled": passed,
        "correct": passed,
        "outcome_status": "completed",
        "stderr_tail": "" if passed else "assert torch.allclose failed",
        "upstream_metadata": "" if passed else "max abs diff 1.0",
    }


def _fast_candidate_timing(candidate_source: str, role: str) -> dict:
    if role == "eager":
        return {
            "source_ok": True,
            "precondition": True,
            "async_leak": False,
            "batches": [10.0] * 12,
        }
    return {
        "source_ok": True,
        "precondition": True,
        "async_leak": False,
        "batches": [5.0] * 12,
    }


def _config(tmp_path: Path, snapshot: Path, **overrides) -> OptimizationConfig:
    defaults = dict(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=3,
        max_repair_rounds=2,
        gpu_budget_seconds=3600.0,
        token_budget=100000,
        output=tmp_path / "run",
        resume=False,
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _run(config, responder, *, evaluate=_passing_evaluate, timing=_fast_candidate_timing):
    ledger = RecordingLedger()
    generator = CandidateGenerator(responder, ledger=ledger)
    return optimize(
        config,
        generator=generator,
        evaluate=evaluate,
        timing=timing,
        correctness_pro=None,
    )


def test_wrong_candidate_feeds_back_then_correct_is_promoted(tmp_path):
    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([BAD_CANDIDATE, GOOD_CANDIDATE])
    config = _config(tmp_path, snapshot, max_candidates=3, max_repair_rounds=2)
    result = _run(config, responder)
    assert responder.calls == 2, "the loop must retry after a structured failure"
    assert "failed" in responder.requests[1].messages[-1][1], (
        "the second request must carry the first attempt's failure evidence"
    )
    assert result.state == "completed"
    assert result.champion_path is not None and result.champion_path.is_file()
    assert result.champion_sha256
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["champion"]["candidate_sha256"] == result.champion_sha256
    assert len(report["candidates"]) == 2
    assert report["candidates"][0]["stage"] == "evaluate"
    assert report["candidates"][1]["status"] == "promoted"
    assert report["candidate_trust"] == "cooperative"
    assert report["adversarially_secure"] is False
    assert parse_exit_code(result) == EXIT_SUCCESS


def test_policy_violating_candidate_never_reaches_gpu(tmp_path):
    snapshot = _stage_problem(tmp_path)
    tamper = json.dumps(
        {
            "code": "import torch\ntorch.allclose = lambda *a, **k: True\n"
            "class ModelNew:\n    def forward(self, x):\n        return x"
        }
    )
    responder = ScriptedResponder([tamper, GOOD_CANDIDATE])

    def evaluate(candidate_source, candidate_name):
        assert "allclose" not in candidate_source, (
            "policy-violating candidate must not be evaluated"
        )
        return _passing_evaluate(candidate_source, candidate_name)

    def timing(candidate_source, role):
        assert "allclose" not in candidate_source, "policy-violating candidate must not be timed"
        return _fast_candidate_timing(candidate_source, role)

    config = _config(tmp_path, snapshot, max_candidates=2, max_repair_rounds=2)
    result = _run(config, responder, evaluate=evaluate, timing=timing)
    assert result.state == "completed"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["candidates"][0]["stage"] == "policy"
    assert report["candidates"][0]["status"] == "rejected"


def test_budget_exhaustion_stops_model_and_gpu_calls(tmp_path):
    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([GOOD_CANDIDATE])
    evaluate_calls = {"count": 0}

    def evaluate(candidate_source, candidate_name):
        evaluate_calls["count"] += 1
        return _passing_evaluate(candidate_source, candidate_name)

    config = _config(tmp_path, snapshot, token_budget=10, max_candidates=2)
    result = _run(config, responder, evaluate=evaluate)
    assert responder.calls == 0, "no model call may happen once the budget cannot cover it"
    assert evaluate_calls["count"] == 0, "no GPU evaluation may happen without budget"
    assert result.state == "budget_exhausted"
    assert parse_exit_code(result) == EXIT_BUDGET_EXHAUSTED


def test_resume_skips_finished_candidates_without_rebilling(tmp_path):
    """Phase 1 ends fruitless (candidate rejected, budget for one); the
    resumed run skips the finished attempt, generates a new candidate for
    attempt 2 only, and the journal prevents double billing."""
    snapshot = _stage_problem(tmp_path)

    responder1 = ScriptedResponder([BAD_CANDIDATE])
    config1 = _config(tmp_path, snapshot, max_candidates=1, output=tmp_path / "run")
    result1 = _run(config1, responder1)
    assert result1.state == "no_improvement"
    settled1 = json.loads(result1.report_path.read_text(encoding="utf-8"))["budget"][
        "settled_tokens"
    ]

    responder2 = ScriptedResponder([GOOD_CANDIDATE])
    config2 = _config(tmp_path, snapshot, max_candidates=2, output=tmp_path / "run", resume=True)
    result2 = _run(config2, responder2)
    assert responder2.calls == 1, "finished candidates must not generate again"
    assert result2.state == "completed"
    report2 = json.loads(result2.report_path.read_text(encoding="utf-8"))
    assert report2["budget"]["settled_tokens"] > settled1, (
        "new work is billed, and journal accounting prevents double billing"
    )
    assert report2["budget"]["settled_tokens"] == settled1 + 30


def test_no_improvement_is_a_legal_terminal(tmp_path):
    snapshot = _stage_problem(tmp_path)
    slow_candidate = candidate_module("def forward(self, x):\n        return x + 0")

    def slower_timing(candidate_source: str, role: str) -> dict:
        if role == "eager":
            return {
                "source_ok": True,
                "precondition": True,
                "async_leak": False,
                "batches": [5.0] * 12,
            }
        return {
            "source_ok": True,
            "precondition": True,
            "async_leak": False,
            "batches": [10.0] * 12,
        }

    responder = ScriptedResponder([slow_candidate])
    config = _config(tmp_path, snapshot, max_candidates=1)
    result = _run(config, responder, timing=slower_timing)
    assert result.state == "no_improvement"
    assert result.champion_path is None
    assert parse_exit_code(result) == EXIT_NO_IMPROVEMENT


def test_exit_code_mapping_covers_terminal_states():
    from kernelagent.optimization import exit_code_for

    assert EXIT_CONFIG_ERROR == 3
    assert exit_code_for("completed") == EXIT_SUCCESS
    assert exit_code_for("no_improvement") == EXIT_NO_IMPROVEMENT
    assert exit_code_for("budget_exhausted") == EXIT_BUDGET_EXHAUSTED
    assert exit_code_for("infra_error") == EXIT_INFRA_ERROR
    with pytest.raises(OptimizationConfigError):
        exit_code_for("not-a-state")


def test_config_rejects_unknown_problem_spec():
    from kernelagent.optimization import parse_problem_spec

    with pytest.raises(OptimizationConfigError):
        parse_problem_spec("kernelbench-40")
    assert parse_problem_spec("kernelbench:l1:40").problem_id == 40
