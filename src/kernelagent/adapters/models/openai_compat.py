"""OpenAI-compatible provider client (T12).

Real transport for the model client protocol: POST {base_url}
/chat/completions with a bearer credential. The credential exists only on
the control plane and is never logged, hashed into identities, or passed
to worker containers. Missing configuration is an explicit permanent
error - there is no silent fallback to recorded responses.

Error mapping: 429/5xx/timeouts are transient (retry candidates);
401/403/400 and contract violations are permanent."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from kernelagent.adapters.models.client import ModelRequest, ModelResponse, ModelUsage
from kernelagent.adapters.models.errors import PermanentModelError, TransientModelError

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        sleep=time.sleep,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise PermanentModelError("provider base_url is not configured")
        if not isinstance(api_key, str) or not api_key.strip():
            raise PermanentModelError("provider api_key is not configured")
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._sleep = sleep

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": request.model_id,
            "messages": [{"role": role, "content": text} for role, text in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        body = json.dumps(payload).encode("utf-8")
        attempts = 0
        while True:
            attempts += 1
            try:
                request_obj = urllib.request.Request(
                    f"{self._base_url}/chat/completions",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request_obj, timeout=self._timeout) as handle:
                    response_payload = json.loads(handle.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in TRANSIENT_STATUS:
                    if attempts >= 3:
                        raise TransientModelError(
                            f"provider HTTP {exc.code} after retries"
                        ) from exc
                    self._sleep(0.5 * (2 ** (attempts - 1)))
                    continue
                raise PermanentModelError(f"provider HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise TransientModelError(f"provider transport failed: {exc}") from exc

        try:
            model_id = response_payload["model"]
            choice = response_payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"] or "stop"
            usage_payload = response_payload["usage"]
            usage = ModelUsage(
                prompt_tokens=usage_payload["prompt_tokens"],
                completion_tokens=usage_payload["completion_tokens"],
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentModelError(f"provider response malformed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise PermanentModelError("provider returned empty content")
        response = ModelResponse(
            request_sha256=request.request_sha256,
            model_id=model_id,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )
        if response.model_id != request.model_id:
            raise PermanentModelError(
                f"response model {response.model_id!r} does not match request {request.model_id!r}"
            )
        return response


def list_models(base_url: str, api_key: str, timeout_seconds: float = 30.0) -> list[dict]:
    """List the models a key can use via the OpenAI-compatible
    ``GET {base_url}/models`` endpoint. Same credential rules as
    :class:`OpenAICompatModelClient`: the key exists only on the control
    plane for this one call and is never logged or persisted."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise PermanentModelError("provider base_url is not configured")
    if not isinstance(api_key, str) or not api_key.strip():
        raise PermanentModelError("provider api_key is not configured")
    url = f"{base_url.strip().rstrip('/')}/models"
    request_obj = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in TRANSIENT_STATUS:
            raise TransientModelError(f"provider HTTP {exc.code}") from exc
        raise PermanentModelError(f"provider HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransientModelError(f"provider transport failed: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    models: list[dict] = []
    for entry in data if isinstance(data, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            owned_by = entry.get("owned_by")
            models.append(
                {"id": entry["id"], "owned_by": owned_by if isinstance(owned_by, str) else None}
            )
    return models
