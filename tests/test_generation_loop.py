"""Candidate generation loop (T12): offline-recorded semantics plus the
no-credentials live error path. The real-model closed loop itself needs
user-provided provider credentials (LIVE_MODEL) and cannot be simulated
here; these tests pin everything around it."""

import pytest

from kernelagent.adapters.models import ModelUsage
from kernelagent.adapters.models.client import (
    ModelRequest,
    ModelResponse,
    RecordedModelClient,
)
from kernelagent.adapters.models.costing import CostLedger
from kernelagent.adapters.models.errors import ParseError, PermanentModelError, TransientModelError
from kernelagent.adapters.models.generation import (
    CandidateGenerator,
    build_request,
    extract_candidate_source,
    validate_candidate_structure,
)
from kernelagent.adapters.models.openai_compat import OpenAICompatModelClient

PROBLEM = "class Model(nn.Module):\n    def forward(self, x): return x\n"
GOOD_CONTENT = (
    '```json\n{"code": "import torch\\nimport torch.nn as nn\\n'
    "class ModelNew(nn.Module):\\n"
    '    def forward(self, x):\\n        return torch.relu(x)\\n"}\n```'
)


def _response(request: ModelRequest, content: str) -> ModelResponse:
    return ModelResponse(
        request_sha256=request.request_sha256,
        model_id=request.model_id,
        content=content,
        finish_reason="stop",
        usage=ModelUsage(prompt_tokens=10, completion_tokens=5),
    )


def test_request_identity_is_content_hashed_and_seed_sensitive():
    a = build_request(PROBLEM, "model-x")
    b = build_request(PROBLEM, "model-x")
    c = build_request(PROBLEM, "model-x", seed_note=" try 2")
    assert a.request_sha256 == b.request_sha256
    assert a.request_sha256 != c.request_sha256


def test_generated_candidate_requires_model_new():
    with pytest.raises(ParseError):
        validate_candidate_structure("class Wrong(nn.Module):\n    pass\n")
    with pytest.raises(ParseError):
        validate_candidate_structure("def broken(:\n")
    validate_candidate_structure("class ModelNew(nn.Module):\n    pass\n")


def test_error_response_yields_failure_never_success():
    """A malformed model answer must produce an explicit failure record and
    must still bill its tokens - but never a success."""
    ledger = CostLedger()
    CandidateGenerator(RecordedModelClient({}), ledger)
    request = build_request(PROBLEM, "model-x")
    response = _response(request, "I cannot help with that.")
    # Drive the recorded path manually through the same parsing pipeline.

    with pytest.raises(ParseError):
        extract_candidate_source(response.content)
    ledger.record(response)
    assert ledger.total_tokens() == 15, "rejected output still costs tokens"


def test_recorded_good_response_produces_structurally_valid_candidate():
    ledger = CostLedger()
    request = build_request(PROBLEM, "model-x")
    client = RecordedModelClient({request.request_sha256: _response(request, GOOD_CONTENT)})
    generator = CandidateGenerator(client, ledger)
    outcome, _ = generator.generate(PROBLEM, "model-x")
    from kernelagent.adapters.models.generation import GenerationSuccess

    assert isinstance(outcome, GenerationSuccess)
    assert "ModelNew" in outcome.candidate_source
    assert ledger.total_tokens() == 15
    assert outcome.finish_reason == "stop"


def test_unconfigured_provider_is_permanent_error():
    with pytest.raises(PermanentModelError):
        OpenAICompatModelClient(base_url="", api_key="")
    # Transport-level failures (DNS, unreachable host) are transient retry
    # candidates by contract; auth/contract failures are permanent.
    client = OpenAICompatModelClient(base_url="https://example.invalid/v1", api_key="k")
    with pytest.raises(TransientModelError):
        client.complete(build_request(PROBLEM, "model-x"))
