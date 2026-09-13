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


# --- RV01: persisted run identity, resume guard, and legal terminals -------


def _stage_problem_with_extra(tmp_path: Path, extra_ids: tuple[int, ...]) -> Path:
    snapshot = _stage_problem(tmp_path)
    for problem_id in extra_ids:
        (snapshot / "KernelBench" / "level1" / f"{problem_id}_OtherOp.py").write_text(
            f"# pinned problem {problem_id}\n", encoding="utf-8"
        )
    return snapshot


def _completed_run(tmp_path: Path, **overrides):
    snapshot = _stage_problem_with_extra(tmp_path, (41,))
    responder = ScriptedResponder([GOOD_CANDIDATE])
    config = _config(
        tmp_path,
        snapshot,
        max_candidates=1,
        max_repair_rounds=0,
        problem=overrides.pop("problem", "kernelbench:l1:40"),
        model_id=overrides.pop("model_id", "test-model"),
        gpu_budget_seconds=overrides.pop("gpu_budget_seconds", 3600.0),
        **overrides,
    )
    result = _run(config, responder)
    assert result.state == "completed"
    return result


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        ({"gpu_device": "nvidia.com/gpu=GPU-swapped"}, "gpu_device"),
        ({"backend": "cuda"}, "backend"),
        ({"model_id": "other-model"}, "model_id"),
    ],
)
def test_resume_rejects_changed_run_identity(tmp_path, changed, field):
    """Swapping GPU identity, backend, or model provider configuration must
    refuse to reuse the old results - never re-attribute old scores."""
    _completed_run(tmp_path)
    responder = ScriptedResponder([])
    config = _config(tmp_path, tmp_path / "snapshot", max_candidates=1, resume=True, **changed)
    with pytest.raises(OptimizationConfigError, match=field):
        _run(config, responder)
    assert responder.calls == 0, "a refused resume must not make any model call"


def test_resume_rejects_changed_problem(tmp_path):
    _completed_run(tmp_path)
    responder = ScriptedResponder([])
    config = _config(
        tmp_path,
        tmp_path / "snapshot",
        max_candidates=1,
        resume=True,
        problem="kernelbench:l1:41",
    )
    with pytest.raises(OptimizationConfigError, match="problem"):
        _run(config, responder)
    assert responder.calls == 0


def test_resume_rejects_changed_container_image(tmp_path, monkeypatch):
    from kernelagent.adapters.evals import kernelbench_eval

    _completed_run(tmp_path)
    monkeypatch.setattr(kernelbench_eval, "EVAL_IMAGE_ID", "sha256:" + "b" * 64)
    responder = ScriptedResponder([])
    config = _config(tmp_path, tmp_path / "snapshot", max_candidates=1, resume=True)
    with pytest.raises(OptimizationConfigError, match="image_id"):
        _run(config, responder)
    assert responder.calls == 0


def test_resume_rejects_changed_timing_protocol(tmp_path, monkeypatch):
    from kernelagent.adapters.evals import timing as timing_adapter

    _completed_run(tmp_path)

    class _ShiftedProtocol:
        def identity_sha256(self) -> str:
            return "e" * 64

    monkeypatch.setattr(timing_adapter, "TimingProtocol", _ShiftedProtocol)
    responder = ScriptedResponder([])
    config = _config(tmp_path, tmp_path / "snapshot", max_candidates=1, resume=True)
    with pytest.raises(OptimizationConfigError, match="timing_protocol_sha256"):
        _run(config, responder)
    assert responder.calls == 0


def test_resume_with_same_identity_does_not_repeat_finished_work(tmp_path):
    """Resuming a finished run with identical identity must execute nothing,
    keep the champion, and add no billing."""
    result1 = _completed_run(tmp_path)
    settled1 = json.loads(result1.report_path.read_text(encoding="utf-8"))["budget"][
        "settled_tokens"
    ]

    responder = ScriptedResponder([])
    config = _config(tmp_path, tmp_path / "snapshot", max_candidates=1, resume=True)
    result2 = _run(config, responder)
    assert responder.calls == 0, "identical-identity resume must not regenerate"
    assert result2.state == "completed"
    assert result2.champion_sha256 == result1.champion_sha256
    report2 = json.loads(result2.report_path.read_text(encoding="utf-8"))
    assert report2["budget"]["settled_tokens"] == settled1, "finished work is never billed twice"
    assert report2["candidates"][0]["status"] == "promoted"


def test_resume_replays_manifest_for_unspecified_config(tmp_path):
    """The loop API mirrors the CLI contract: manifest values keep identity,
    so a resume that only overrides the output location restores the
    original problem/model/budget from the manifest, not local defaults."""
    result1 = _completed_run(
        tmp_path,
        problem="kernelbench:l1:41",
        model_id="original-model",
        gpu_budget_seconds=1234.5,
    )
    responder = ScriptedResponder([])
    restored_config = _config(
        tmp_path,
        tmp_path / "snapshot",
        max_candidates=1,
        resume=True,
        problem="kernelbench:l1:41",
        model_id="original-model",
        gpu_budget_seconds=1234.5,
    )
    result2 = _run(restored_config, responder)
    assert responder.calls == 0
    assert result2.state == "completed"
    report2 = json.loads(result2.report_path.read_text(encoding="utf-8"))
    report1 = json.loads(result1.report_path.read_text(encoding="utf-8"))
    assert report2["config"]["problem"] == "kernelbench:l1:41"
    assert report2["config"]["model_id"] == "original-model"
    assert report2["budget"]["gpu_seconds_limit"] == 1234.5
    assert report2["budget"]["settled_tokens"] == report1["budget"]["settled_tokens"]


def test_resume_budget_override_records_explicit_event(tmp_path):
    """Budget/allowance changes are allowed on resume but must leave an
    explicit, auditable journal event - never a silent change."""
    snapshot = _stage_problem(tmp_path)
    responder1 = ScriptedResponder([BAD_CANDIDATE])
    config1 = _config(
        tmp_path, snapshot, max_candidates=1, max_repair_rounds=0, output=tmp_path / "run"
    )
    assert _run(config1, responder1).state == "no_improvement"

    responder2 = ScriptedResponder([GOOD_CANDIDATE])
    config2 = _config(
        tmp_path,
        snapshot,
        max_candidates=2,
        max_repair_rounds=1,
        gpu_budget_seconds=7200.0,
        output=tmp_path / "run",
        resume=True,
    )
    result = _run(config2, responder2)
    assert result.state == "completed"
    assert responder2.calls == 1
    entries = [
        json.loads(line)
        for line in (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    revised = {
        entry["field"]: (entry["previous_value"], entry["new_value"])
        for entry in entries
        if entry["kind"] == "manifest_revised"
    }
    assert revised == {
        "max_candidates": (1, 2),
        "max_repair_rounds": (0, 1),
        "gpu_budget_seconds": (3600.0, 7200.0),
    }


def test_fresh_run_rejects_existing_output_directory(tmp_path):
    """A non-resume run into an existing output directory is a config error
    raised before any model or GPU call (RV01)."""
    output = tmp_path / "run"
    output.mkdir()
    (output / "stale.txt").write_text("leftover", encoding="utf-8")
    responder = ScriptedResponder([])
    config = _config(tmp_path, _stage_problem(tmp_path), output=output)
    with pytest.raises(OptimizationConfigError, match="already exists"):
        _run(config, responder)
    assert responder.calls == 0


def test_resume_without_manifest_is_refused(tmp_path):
    """A journal without its manifest cannot prove identity: refuse instead
    of defaulting to 'matches' (legacy runs must start a new experiment)."""
    _completed_run(tmp_path)
    manifest_path = tmp_path / "run" / "run_manifest.json"
    assert manifest_path.is_file()
    manifest_path.unlink()
    responder = ScriptedResponder([])
    config = _config(tmp_path, tmp_path / "snapshot", max_candidates=1, resume=True)
    with pytest.raises(OptimizationConfigError, match="manifest missing"):
        _run(config, responder)
    assert responder.calls == 0


@pytest.mark.parametrize("bad_candidates", [0, -2])
def test_invalid_max_candidates_rejected_before_any_call(tmp_path, bad_candidates):
    responder = ScriptedResponder([])
    with pytest.raises(OptimizationConfigError, match="max_candidates"):
        _config(tmp_path, _stage_problem(tmp_path), max_candidates=bad_candidates)
    assert responder.calls == 0, "invalid resource counts must cost zero model calls"


@pytest.mark.parametrize(
    "budget_field",
    ["gpu_budget_seconds", "token_budget"],
)
@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), 0, -5],
)
def test_invalid_budgets_rejected_before_any_call(tmp_path, budget_field, bad_value):
    responder = ScriptedResponder([])
    with pytest.raises(OptimizationConfigError, match=budget_field):
        _config(tmp_path, _stage_problem(tmp_path), **{budget_field: bad_value})
    assert responder.calls == 0
    assert not (tmp_path / "run" / "journal.jsonl").exists(), "nothing may be scheduled"


def test_completed_terminal_requires_a_champion():
    from kernelagent.optimization import validate_terminal

    validate_terminal("completed", "a" * 64)  # legal
    validate_terminal("no_improvement", None)  # legal: honest empty terminal
    with pytest.raises(OptimizationConfigError, match="champion"):
        validate_terminal("completed", None)
    with pytest.raises(OptimizationConfigError, match="champion"):
        validate_terminal("completed", "")


def test_journal_reconstructs_candidate_stage_timeline(tmp_path):
    """An external reader can rebuild each candidate's pipeline stage
    sequence and terminal outcome from journal events alone."""
    from kernelagent.domain import PIPELINE_STAGE_ORDER

    snapshot = _stage_problem(tmp_path)
    responder = ScriptedResponder([BAD_CANDIDATE, GOOD_CANDIDATE])
    config = _config(tmp_path, snapshot, max_candidates=2, max_repair_rounds=2)
    result = _run(config, responder)
    assert result.state == "completed"

    entries = [
        json.loads(line)
        for line in (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    timeline: dict[str, list[str]] = {}
    terminals: dict[str, str] = {}
    for entry in entries:
        if entry["kind"] == "stage_started":
            timeline.setdefault(entry["action_id"], []).append(entry["stage"])
        elif entry["kind"] == "action_finished" and entry["action_id"].startswith("candidate-"):
            terminals[entry["action_id"]] = str(entry.get("result_ref"))

    assert timeline["baseline-eager"] == ["timing"]
    assert timeline["candidate-000"] == ["generate", "policy", "evaluate"]
    assert terminals["candidate-000"] == "candidate-000:eval-failed"
    assert timeline["candidate-001"] == [stage.value for stage in PIPELINE_STAGE_ORDER]
    assert terminals["candidate-001"] == "candidate-001:promoted"
