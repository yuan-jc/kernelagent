"""RV03 acceptance (REVIEW.md R5/RV03): resumed-run request parity.

For every candidate-failure feedback class - parse failure, policy
rejection, correctness failure, performance no-improvement - the next
generation request after an interrupt+resume must be byte-identical to
the request the uninterrupted path would have sent.

Two paths drive the same scripted responder sequence (failing candidate
first, repairing candidate second):

- path A: one continuous ``optimize`` run over both candidates;
- path B: the first run ends after the failing candidate, and a second
  ``optimize(resume=True)`` call continues (the process died in between).

The request is fully determined by fixed, persisted elements, so parity
is provable without persisting the request body itself (RV03: no
spurious storage): the problem source and model identity are pinned by
the run manifest (RV01 identity guard), the prompt template and sampling
parameters are frozen constants of ``build_request``, and the seed note
is ``compose_seed_note(restored feedback, plan fragment)`` where the
feedback text and ``method_id`` are persisted on the candidate record.
Each record therefore only needs to carry the generation's
``request_sha256`` as durable request-identity evidence; the tests
assert it matches the observed requests on both paths.

Billing: a completed generation is settled exactly once - the resumed
run must not re-generate or re-bill candidate-000.

Everything is offline: a scripted responder, fake evaluator/timing
ports, no GPU, no network, no secrets in any request or record.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from kernelagent.adapters.models.client import ModelResponse, ModelUsage
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.optimization import OptimizationConfig, optimize

PROBLEM_SOURCE = "# pinned problem\n"

# The four feedback classes (RV03) as model responses:
PARSE_FAILURE = json.dumps({"note": "no code key here"})
POLICY_CANDIDATE = json.dumps(
    {
        "code": "import torch\nimport os\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return x"
    }
)
CORRECTNESS_CANDIDATE = json.dumps(
    {"code": "class ModelNew:\n    def forward(self, x):\n        return 0"}
)
SLOW_CANDIDATE = json.dumps(
    {"code": "class ModelNew:\n    def forward(self, x):\n        return x + 0"}
)
GOOD_CANDIDATE = json.dumps(
    {"code": "class ModelNew:\n    def forward(self, x):\n        return x"}
)

# case -> (first response, substrings that MUST appear in the second
# request if the failure feedback was genuinely restored - guards the
# parity assertion against trivially matching two feedback-less requests)
FEEDBACK_CASES = {
    "parse_failure": (
        PARSE_FAILURE,
        ("failed at stage 'generation'", "generation payload must be an object"),
    ),
    "policy_rejection": (
        POLICY_CANDIDATE,
        ("failed at stage 'policy'", "restricted-import: os"),
    ),
    "correctness_failure": (
        CORRECTNESS_CANDIDATE,
        ("failed at stage 'evaluate'", "upstream evaluator verdict"),
    ),
    "no_improvement": (
        SLOW_CANDIDATE,
        ("failed at stage 'confirm'", "does not establish speedup"),
    ),
}

COMPILATION_ERROR = {
    "compilation_error_name": "CompilationError",
    "compilation_error": "invalid operands to binary expression",
}


class ScriptedResponder:
    """Order-based model double that records every request it sees."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=self._contents.pop(0),
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20),
        )


class RecordingLedger:
    def __init__(self):
        self._total = 0

    def record(self, response):
        self._total += response.usage.total_tokens

    def total_tokens(self) -> int:
        return self._total


def _stage_snapshot(root: Path) -> Path:
    """Identical pinned snapshot bytes for both paths (the problem source
    is part of the request, so both paths must pin the same bytes)."""
    snapshot = root / "snapshot"
    (snapshot / "KernelBench" / "level1").mkdir(parents=True)
    (snapshot / "KernelBench" / "level1" / "40_LayerNorm.py").write_text(
        PROBLEM_SOURCE, encoding="utf-8"
    )
    return snapshot


def _config(
    root: Path,
    snapshot: Path,
    *,
    resume: bool,
    max_candidates: int,
    max_repair_rounds: int,
) -> OptimizationConfig:
    return OptimizationConfig(
        problem="kernelbench:l1:40",
        backend="triton",
        model_id="test-model",
        base_url="http://127.0.0.1:9/v1",
        max_candidates=max_candidates,
        max_repair_rounds=max_repair_rounds,
        gpu_budget_seconds=3600.0,
        token_budget=100000,
        output=root / "run",
        resume=resume,
        snapshot_root=snapshot,
        gpu_device="nvidia.com/gpu=GPU-fake",
    )


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


def _compilation_then_passing_evaluate(candidate_source: str, candidate_name: str) -> dict:
    if candidate_name == "candidate-000":
        return {
            "adapter_pass": False,
            "compiled": False,
            "correct": False,
            "outcome_status": "completed",
            "stderr_tail": "nvcc failed",
            "upstream_metadata": COMPILATION_ERROR,
        }
    return _passing_evaluate(candidate_source, candidate_name)


def _source_aware_timing(candidate_source: str, role: str) -> dict:
    """Eager baseline 10ms; a genuinely-changed candidate 5ms (promoted),
    the no-improvement candidate 10ms (retained at confirm)."""
    if role == "eager":
        batches = [10.0] * 12
    else:
        faster = candidate_source.rstrip().endswith("return x")
        batches = ([5.0] if faster else [10.0]) * 12
    return {"source_ok": True, "precondition": True, "async_leak": False, "batches": batches}


def _drive(config: OptimizationConfig, contents: list[str], *, evaluate=_passing_evaluate):
    responder = ScriptedResponder(contents)
    result = optimize(
        config,
        generator=CandidateGenerator(responder, ledger=RecordingLedger()),
        evaluate=evaluate,
        timing=_source_aware_timing,
        correctness_pro=None,
    )
    return result, responder


def test_compilation_feedback_is_identical_after_resume(tmp_path):
    snapshot_a = _stage_snapshot(tmp_path / "a")
    result_a, responder_a = _drive(
        _config(tmp_path / "a", snapshot_a, resume=False, max_candidates=2, max_repair_rounds=2),
        [CORRECTNESS_CANDIDATE, GOOD_CANDIDATE],
        evaluate=_compilation_then_passing_evaluate,
    )

    snapshot_b = _stage_snapshot(tmp_path / "b")
    _drive(
        _config(tmp_path / "b", snapshot_b, resume=False, max_candidates=1, max_repair_rounds=0),
        [CORRECTNESS_CANDIDATE],
        evaluate=_compilation_then_passing_evaluate,
    )
    result_b, responder_b = _drive(
        _config(tmp_path / "b", snapshot_b, resume=True, max_candidates=2, max_repair_rounds=2),
        [GOOD_CANDIDATE],
        evaluate=_compilation_then_passing_evaluate,
    )

    assert result_a.state == result_b.state == "completed"
    assert responder_a.requests[1] == responder_b.requests[0]
    feedback = responder_a.requests[1].messages[-1][1]
    assert "CompilationError" in feedback
    assert "invalid operands to binary expression" in feedback


def _journal(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_bytes(output: Path, name: str) -> bytes:
    return (output / "records" / f"{name}.json").read_bytes()


@pytest.mark.parametrize("case_id", sorted(FEEDBACK_CASES))
def test_resume_request_parity_and_single_billing(tmp_path, case_id):
    bad, markers = FEEDBACK_CASES[case_id]

    # Path A: one continuous run (failing candidate -> repaired candidate).
    snapshot_a = _stage_snapshot(tmp_path / "a")
    result_a, responder_a = _drive(
        _config(tmp_path / "a", snapshot_a, resume=False, max_candidates=2, max_repair_rounds=2),
        [bad, GOOD_CANDIDATE],
    )

    # Path B: the process dies after the failing candidate; a resume
    # continues in a fresh process state (new optimize call).
    snapshot_b = _stage_snapshot(tmp_path / "b")
    result_b1, responder_b1 = _drive(
        _config(tmp_path / "b", snapshot_b, resume=False, max_candidates=1, max_repair_rounds=0),
        [bad],
    )
    # Snapshot run 1's report now: the resume rewrites <output>/report.json
    # with the cumulative ledger, and result paths are read lazily.
    report_b1_payload = json.loads(result_b1.report_path.read_text(encoding="utf-8"))
    assert result_b1.state == "no_improvement"
    result_b2, responder_b2 = _drive(
        _config(tmp_path / "b", snapshot_b, resume=True, max_candidates=2, max_repair_rounds=2),
        [GOOD_CANDIDATE],
    )

    assert result_a.state == "completed"
    assert result_b2.state == "completed"

    # Completed generations are never repeated: exactly one call per
    # candidate on every path.
    assert responder_a.calls == 2
    assert responder_b1.calls == 1
    assert responder_b2.calls == 1, "resume must not regenerate candidate-000"

    # The first request is identical on both paths...
    assert responder_a.requests[0] == responder_b1.requests[0]

    # ...and the second request - the point of RV03 - is byte-identical
    # between the continuous and the resumed path: full frozen-dataclass
    # equality (model, sampling parameters, messages) plus equal content
    # hash.
    second_a = responder_a.requests[1]
    second_b = responder_b2.requests[0]
    assert second_a == second_b, (
        "the resumed path must resend exactly the request the continuous "
        f"path would have sent (case {case_id})"
    )
    assert second_a.request_sha256 == second_b.request_sha256
    assert second_a.messages[-1][1] == second_b.messages[-1][1]

    # The restored failure feedback is genuinely present (never a silent
    # empty seed note on both sides).
    text = second_a.messages[-1][1]
    for marker in markers:
        assert marker in text, f"restored feedback must contain {marker!r} (case {case_id})"

    # Billing: 30 tokens per generation, settled exactly once each. The
    # resumed run adds only candidate-001's settlement - candidate-000 is
    # never generated or billed twice.
    report_a = json.loads(result_a.report_path.read_text(encoding="utf-8"))
    report_b2 = json.loads(result_b2.report_path.read_text(encoding="utf-8"))
    assert report_a["budget"]["settled_tokens"] == 60
    assert report_b1_payload["budget"]["settled_tokens"] == 30
    assert report_b2["budget"]["settled_tokens"] == 60

    settled = [e for e in _journal(tmp_path / "b" / "run") if e["kind"] == "budget_settled"]
    settled_per_action = Counter(e["action_id"] for e in settled)
    assert settled_per_action["candidate-000"] == 1, (
        "a finished candidate must never be settled twice across the resume"
    )
    assert settled_per_action["candidate-001"] == 1
    finished_per_action = Counter(
        e["action_id"] for e in _journal(tmp_path / "b" / "run") if e["kind"] == "action_finished"
    )
    assert finished_per_action["candidate-000"] == 1


@pytest.mark.parametrize("case_id", sorted(FEEDBACK_CASES))
def test_candidate_record_persists_request_identity(tmp_path, case_id):
    """Durable request-identity evidence (RV03): every candidate record
    carries the sha256 of the generation request it consumed, and the
    failing candidate's whole record is byte-identical across the two
    paths - so the parity claim is verifiable from disk alone, without
    trusting the responder doubles."""
    bad, _ = FEEDBACK_CASES[case_id]

    snapshot_a = _stage_snapshot(tmp_path / "a")
    result_a, responder_a = _drive(
        _config(tmp_path / "a", snapshot_a, resume=False, max_candidates=2, max_repair_rounds=2),
        [bad, GOOD_CANDIDATE],
    )
    snapshot_b = _stage_snapshot(tmp_path / "b")
    _, responder_b1 = _drive(
        _config(tmp_path / "b", snapshot_b, resume=False, max_candidates=1, max_repair_rounds=0),
        [bad],
    )
    _, responder_b2 = _drive(
        _config(tmp_path / "b", snapshot_b, resume=True, max_candidates=2, max_repair_rounds=2),
        [GOOD_CANDIDATE],
    )
    assert result_a.state == "completed"

    output_a = tmp_path / "a" / "run"
    output_b = tmp_path / "b" / "run"

    # candidate-000 record: persisted request hash matches the observed
    # request on both paths, and the record bytes are identical.
    record_a = json.loads(_record_bytes(output_a, "candidate-000"))
    record_b = json.loads(_record_bytes(output_b, "candidate-000"))
    assert record_a["request_sha256"] == responder_a.requests[0].request_sha256
    assert record_b["request_sha256"] == responder_b1.requests[0].request_sha256
    assert (
        hashlib.sha256(_record_bytes(output_a, "candidate-000")).hexdigest()
        == hashlib.sha256(_record_bytes(output_b, "candidate-000")).hexdigest()
    ), "the failing candidate's durable record must be identical on both paths"

    # candidate-001 record: same request identity on both paths (content
    # equality was already asserted in the parity test; champion_path
    # differs because the runs live in different directories).
    record_a1 = json.loads(_record_bytes(output_a, "candidate-001"))
    record_b1 = json.loads(_record_bytes(output_b, "candidate-001"))
    assert record_a1["request_sha256"] == responder_a.requests[1].request_sha256
    assert record_b1["request_sha256"] == responder_b2.requests[0].request_sha256
    assert record_a1["request_sha256"] == record_b1["request_sha256"]
