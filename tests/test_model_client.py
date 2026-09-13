"""Model client offline layer: contracts, replay, retry, budget, parsing."""

import json
import subprocess
import sys

import pytest

from kernelagent.adapters.models import (
    BudgetExceededError,
    CostLedger,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ParseError,
    PermanentModelError,
    RecordedModelClient,
    RecordingMissError,
    RetryingModelClient,
    ScriptedModelClient,
    TokenBudget,
    TransientModelError,
    extract_json_payload,
    parse_structured,
)


def make_request(**overrides) -> ModelRequest:
    values = {
        "model_id": "glm-5.3",
        "messages": (
            ("system", "You are a kernel optimizer."),
            ("user", "Propose one optimization."),
        ),
        "temperature": 0.2,
        "max_tokens": 1024,
        "purpose": "hypothesis",
    }
    values.update(overrides)
    return ModelRequest(**values)


def make_response(
    request: ModelRequest, content='```json\n{"claim": "fuse it"}\n```'
) -> ModelResponse:
    return ModelResponse(
        request_sha256=request.request_sha256,
        model_id=request.model_id,
        content=content,
        finish_reason="stop",
        usage=ModelUsage(prompt_tokens=120, completion_tokens=40),
    )


# -- request/response contracts ---------------------------------------------


def test_request_hash_stable_and_content_sensitive():
    assert make_request().request_sha256 == make_request().request_sha256
    assert make_request().request_sha256 != make_request(temperature=0.5).request_sha256
    assert (
        make_request().request_sha256
        != make_request(messages=(("user", "different prompt"),)).request_sha256
    )
    assert make_request().request_sha256 != make_request(max_tokens=512).request_sha256
    assert make_request().request_sha256 == make_request(purpose="other").request_sha256


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"messages": ()},
        {"messages": (("wizard", "hi"),)},
        {"messages": (("user", ""),)},
        {"messages": (["user", "hi"],)},
        {"messages": [("user", "hi")]},
        {"temperature": True},
        {"temperature": "hot"},
        {"temperature": 2.5},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"purpose": []},
    ],
    ids=[
        "model",
        "empty-messages",
        "bad-role",
        "empty-text",
        "list-message",
        "list-messages",
        "bool-temp",
        "str-temp",
        "temp-range",
        "zero-tokens",
        "bool-tokens",
        "mutable-purpose",
    ],
)
def test_invalid_requests_rejected(overrides):
    with pytest.raises(ValueError):
        make_request(**overrides)


def test_usage_totals_and_rejects_bad_values():
    usage = ModelUsage(prompt_tokens=10, completion_tokens=5)
    assert usage.total_tokens == 15
    assert usage.plus(ModelUsage(1, 2)) == ModelUsage(11, 7)
    with pytest.raises(ValueError):
        ModelUsage(prompt_tokens=-1, completion_tokens=0)
    with pytest.raises(ValueError):
        ModelUsage(prompt_tokens=True, completion_tokens=0)


def test_response_requires_content_and_finish_reason():
    request = make_request()
    with pytest.raises(ValueError):
        make_response(request, content="")
    with pytest.raises(ValueError):
        response = make_response(request)
        ModelResponse(
            request_sha256=response.request_sha256,
            model_id=response.model_id,
            content="x",
            finish_reason="",
            usage=response.usage,
        )
    with pytest.raises(ValueError, match="request_sha256"):
        ModelResponse("bad", request.model_id, "x", "stop", ModelUsage(1, 1))
    with pytest.raises(ValueError, match="usage"):
        ModelResponse(request.request_sha256, request.model_id, "x", "stop", [1, 1])


# -- offline clients ---------------------------------------------------------


def test_scripted_client_returns_in_order_and_raises_scripted_errors():
    request = make_request()
    good = make_response(request)
    client = ScriptedModelClient(
        [TransientModelError("rate limited"), PermanentModelError("auth"), good]
    )
    with pytest.raises(TransientModelError):
        client.complete(request)
    with pytest.raises(PermanentModelError):
        client.complete(request)
    assert client.complete(request) == good
    with pytest.raises(RecordingMissError):
        client.complete(request)


def test_scripted_client_rejects_foreign_responses():
    client = ScriptedModelClient([make_response(make_request(max_tokens=512))])
    with pytest.raises(ValueError):
        client.complete(make_request())


def test_recorded_client_hits_by_request_hash_and_misses_loudly():
    request = make_request()
    response = make_response(request)
    client = RecordedModelClient({request.request_sha256: response})
    assert client.complete(request) == response
    assert client.complete(request) == response
    assert client.hits == 2
    with pytest.raises(RecordingMissError, match="no recorded response"):
        client.complete(make_request(max_tokens=512))


def test_recorded_client_rejects_miskeyed_or_wrong_model_response():
    request = make_request()
    other = make_request(max_tokens=512)
    with pytest.raises(ValueError, match="recording key"):
        RecordedModelClient({request.request_sha256: make_response(other)})

    wrong_model = ModelResponse(
        request_sha256=request.request_sha256,
        model_id="model-b",
        content="{}",
        finish_reason="stop",
        usage=ModelUsage(1, 1),
    )
    client = RecordedModelClient({request.request_sha256: wrong_model})
    with pytest.raises(ValueError, match="does not match request model"):
        client.complete(request)
    assert client.hits == 0


def test_retrying_client_recovers_from_transient_errors():
    request = make_request()
    good = make_response(request)
    sleeps: list[float] = []
    inner = ScriptedModelClient(
        [TransientModelError("rate limited"), TransientModelError("timeout"), good]
    )
    client = RetryingModelClient(inner, max_attempts=4, sleep=sleeps.append, backoff_seconds=0.5)
    assert client.complete(request) == good
    assert sleeps == [0.5, 1.0]


def test_retrying_client_raises_last_transient_after_budget():
    request = make_request()
    inner = ScriptedModelClient(
        [TransientModelError("e1"), TransientModelError("e2"), TransientModelError("e3")]
    )
    client = RetryingModelClient(inner, max_attempts=3, sleep=lambda s: None)
    with pytest.raises(TransientModelError, match="e3"):
        client.complete(request)
    assert inner.calls == 3


def test_retrying_client_does_not_retry_permanent_errors():
    request = make_request()
    inner = ScriptedModelClient([PermanentModelError("invalid api key")])
    client = RetryingModelClient(inner, max_attempts=5, sleep=lambda s: None)
    with pytest.raises(PermanentModelError):
        client.complete(request)
    assert inner.calls == 1


def test_retrying_client_rejects_zero_attempts():
    with pytest.raises(ValueError):
        RetryingModelClient(ScriptedModelClient([]), max_attempts=0)


# -- cost accounting ---------------------------------------------------------


def test_ledger_aggregates_per_model():
    request_a = make_request()
    request_b = make_request(model_id="other-model")
    ledger = CostLedger()
    ledger.record(make_response(request_a))
    ledger.record(make_response(make_request()))
    ledger.record(make_response(request_b))
    totals = ledger.totals()
    assert totals["glm-5.3"].total_tokens == 320
    assert totals["glm-5.3"].requests == 2
    assert totals["other-model"].total_tokens == 160
    assert ledger.total_tokens() == 480
    assert ledger.total_requests() == 3
    assert ledger.totals(model_id="missing") == {}


def test_budget_blocks_when_projected_usage_exceeds_cap():
    ledger = CostLedger()
    ledger.record(make_response(make_request()))
    budget = TokenBudget(max_total_tokens=200)
    budget.ensure_allowed(ledger, estimated_next_tokens=30)
    with pytest.raises(BudgetExceededError, match="exceeds budget"):
        budget.ensure_allowed(ledger, estimated_next_tokens=100)


def test_budget_rejects_invalid_configuration():
    ledger = CostLedger()
    with pytest.raises(ValueError):
        TokenBudget(max_total_tokens=0)
    budget = TokenBudget(max_total_tokens=10)
    with pytest.raises(ValueError):
        budget.ensure_allowed(ledger, estimated_next_tokens=-1)
    for invalid in (True, 1.5, float("nan")):
        with pytest.raises(ValueError, match="integer"):
            budget.ensure_allowed(ledger, estimated_next_tokens=invalid)


# -- structured parsing ------------------------------------------------------


def test_parse_bare_and_fenced_json():
    assert parse_structured('{"claim": "x"}') == {"claim": "x"}
    fenced = 'Here is my proposal:\n```json\n{"claim": "fuse", "risk": "regs"}\n```\nDone.'
    assert parse_structured(fenced) == {"claim": "fuse", "risk": "regs"}
    plain_fence = '```\n{"claim": "y"}\n```'
    assert extract_json_payload(plain_fence) == '{"claim": "y"}'


def test_parse_enforces_required_keys_and_types():
    with pytest.raises(ParseError, match="missing required key"):
        parse_structured('{"other": 1}', required={"claim": "str"})
    with pytest.raises(ParseError, match="must be str"):
        parse_structured('{"claim": 3}', required={"claim": "str"})
    with pytest.raises(ParseError, match="must be int"):
        parse_structured('{"count": true}', required={"count": "int"})
    with pytest.raises(ParseError, match="unknown expected type"):
        parse_structured("{}", required={"a": "tensor"})


@pytest.mark.parametrize(
    "text",
    ["", "   ", "just prose, no json", "[1, 2, 3]", "{broken"],
    ids=["empty", "blank", "prose", "non-object", "broken"],
)
def test_parse_failures_are_explicit(text):
    with pytest.raises(ParseError):
        parse_structured(text)


# -- evidence integration and import boundary --------------------------------


def test_exchange_records_into_evidence_store(tmp_path):

    from kernelagent.adapters.storage import EvidenceStore

    request = make_request()
    response = make_response(request)
    with EvidenceStore(tmp_path / "evidence") as store:
        request_ref = store.put_artifact(
            json.dumps(
                {"request_sha256": request.request_sha256, "model_id": request.model_id},
                sort_keys=True,
            ).encode("utf-8"),
            kind="model_request",
            producer_version="t12a",
        )
        response_ref = store.put_artifact(
            json.dumps({"content": response.content, "usage": [120, 40]}, sort_keys=True).encode(
                "utf-8"
            ),
            kind="model_response",
            producer_version="t12a",
        )
        assert store.audit().healthy
    assert request_ref.kind == "model_request"
    assert response_ref.kind == "model_response"


def test_models_import_no_provider_sdks_or_gpu_stack():
    code = (
        "import sys; import kernelagent.adapters.models; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


# --- Provider URL normalization and self-explanatory HTTP errors ---------


def test_normalize_base_url_trims_user_paste():
    from kernelagent.adapters.models.openai_compat import normalize_base_url

    assert normalize_base_url(" https://api.deepseek.com ") == "https://api.deepseek.com"
    assert (
        normalize_base_url("https://api.deepseek.com/v1/chat/completions")
        == "https://api.deepseek.com/v1"
    )
    assert normalize_base_url('"https://x/v1/"') == "https://x/v1"
    assert normalize_base_url("https://x/v1") == "https://x/v1"


def test_list_models_surfaces_provider_reason(monkeypatch):
    import io
    import urllib.error

    from kernelagent.adapters.models.errors import PermanentModelError
    from kernelagent.adapters.models.openai_compat import list_models

    def raise_http_error(request, timeout):
        raise urllib.error.HTTPError(
            url="https://api.deepseek.com/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Your api key is invalid"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    with pytest.raises(PermanentModelError, match="401.*api key is invalid"):
        list_models("https://api.deepseek.com", "sk-wrong")
