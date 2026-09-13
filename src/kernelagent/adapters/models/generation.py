"""Candidate generation loop (T12): spec in, evaluated candidate out.

The generator builds a replayable ModelRequest from a frozen problem
description, calls the injected ModelClient (real provider or recorded
replay - the loop cannot tell and must not care), parses the candidate
source out of the response, validates it structurally, and hands it to
the caller for GPU evaluation through the T04 boundary.

Honesty rules (design §12): every completed response is billed to the
cost ledger even when the candidate is rejected; a response that yields
no valid candidate produces an explicit ``GenerationFailure`` - never a
success record; a live-model attempt without configured credentials is a
permanent, reported error rather than a silent replay."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from kernelagent.adapters.models.client import ModelClient, ModelRequest, ModelResponse
from kernelagent.adapters.models.costing import CostLedger
from kernelagent.adapters.models.errors import ParseError
from kernelagent.adapters.models.parsing import extract_json_payload

GENERATION_PROTOCOL = "t12-generation-v1"
CANDIDATE_POLICY_PROTOCOL = "candidate-policy-v1"

# Restricted-module roots (ADR-0004): a candidate that imports these can
# reach the evaluator's process, host channels, or the /out verdict file.
_RESTRICTED_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "shutil",
        "signal",
        "ctypes",
        "importlib",
        "threading",
        "asyncio",
        "pickle",
        "builtins",
        "sys",
    }
)
# Calls that execute or rewire code/namespace at evaluator runtime.
_DYNAMIC_CALLS = frozenset({"eval", "exec", "compile", "globals", "locals", "vars"})
_DUNDER_ATTRIBUTES = frozenset({"__dict__", "__globals__", "__builtins__"})


@dataclass(frozen=True, slots=True)
class CandidatePolicyResult:
    allowed: bool
    violations: tuple[str, ...]


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def inspect_candidate_policy(candidate_source: str) -> CandidatePolicyResult:
    """AST misuse-prevention gate over candidate source (ADR-0004).

    Flags the documented evaluator-tampering patterns with structured
    violations: attribute assignment on imported/restricted modules
    (``torch.allclose = ...``), parent ``/out`` access, dynamic execution
    (``eval``/``exec``/``compile``/namespace rewiring), restricted imports
    (``os``/``subprocess``/``socket``/...), ``setattr`` monkeypatching and
    dunder global access.

    This is a best-effort line of defense against accidental and naive
    tampering, NOT a security boundary: exec() can do more than the AST
    surface shows (``getattr`` by computed name, ``__import__``, module
    aliasing). Until the trusted-MVP trust-domain split, evaluation runs
    declare ``candidate_trust=cooperative`` and
    ``adversarially_secure=false``; champions require human review."""
    try:
        tree = ast.parse(candidate_source)
    except SyntaxError as exc:
        return CandidatePolicyResult(allowed=False, violations=(f"parse-error: {exc.msg}",))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names if alias.name != "*")

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                f"restricted-import: {alias.name}"
                for alias in node.names
                if (alias.asname or alias.name.split(".")[0]) in _RESTRICTED_IMPORTS
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _RESTRICTED_IMPORTS:
                violations.append(f"restricted-import: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr in _DUNDER_ATTRIBUTES:
            violations.append(f"dunder-attribute-access: {node.attr}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Attribute):
                    continue
                root = _root_name(target)
                if root and (root in imported or root in _RESTRICTED_IMPORTS):
                    violations.append(f"external-attribute-assignment: {_dotted(target)}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _DYNAMIC_CALLS:
                violations.append(f"dynamic-call: {name}")
            elif name == "setattr":
                violations.append("setattr-call: setattr")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "/out" in node.value:
                violations.append(f"output-path-access: {node.value[:80]!r}")

    deduped = tuple(dict.fromkeys(violations))
    return CandidatePolicyResult(allowed=not deduped, violations=deduped)


PROMPT_TEMPLATE = (
    "You are a CUDA/Triton kernel engineer. Optimize the PyTorch operator below.\n"
    "Problem source:\n{problem_source}\n"
    'Return EXACTLY one JSON object with key "code" whose value is the complete\n'
    "candidate module source defining class ModelNew(nn.Module) with the same\n"
    "__init__ signature and forward semantics as the reference Model.\n"
    "Do not change the operator's math. No prose outside the JSON."
)


@dataclass(frozen=True, slots=True)
class GenerationSuccess:
    candidate_source: str
    candidate_sha256: str
    request_sha256: str
    finish_reason: str


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    stage: str  # parse | structure
    reason: str
    request_sha256: str


def build_request(problem_source: str, model_id: str, seed_note: str = "") -> ModelRequest:
    """Frozen prompt assembly; the request hash is the generation identity."""
    return ModelRequest(
        model_id=model_id,
        messages=(("user", PROMPT_TEMPLATE.format(problem_source=problem_source) + seed_note),),
        temperature=0.0,
        max_tokens=2048,
        purpose="candidate-generation",
    )


def extract_candidate_source(response_text: str) -> str:
    """Pull the candidate module source out of the model response: a JSON
    object with key ``code``, optionally wrapped in a fenced block."""
    payload = extract_json_payload(response_text)
    import json as _json

    try:
        parsed = _json.loads(payload)
    except _json.JSONDecodeError as exc:
        raise ParseError(f"generation payload is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("code"), str):
        raise ParseError('generation payload must be an object with string key "code"')
    return parsed["code"]


def validate_candidate_structure(candidate_source: str) -> None:
    """Structural gate before GPU work: syntactic module + ModelNew class."""
    try:
        tree = ast.parse(candidate_source)
    except SyntaxError as exc:
        raise ParseError(f"candidate source does not parse: {exc}") from exc
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    if "ModelNew" not in class_names:
        raise ParseError("candidate source does not define class ModelNew")


class CandidateGenerator:
    def __init__(self, client: ModelClient, ledger: CostLedger):
        self._client = client
        self.ledger = ledger

    def generate(
        self, problem_source: str, model_id: str, seed_note: str = ""
    ) -> tuple[GenerationSuccess | GenerationFailure, ModelRequest]:
        request = build_request(problem_source, model_id, seed_note)
        try:
            response: ModelResponse = self._client.complete(request)
        except Exception as exc:  # noqa: BLE001 - transport errors are failures to record
            return (
                GenerationFailure(
                    stage="transport", reason=str(exc)[:300], request_sha256=request.request_sha256
                ),
                request,
            )
        # Every completed response is billed, accepted or not.
        self.ledger.record(response)
        try:
            candidate_source = extract_candidate_source(response.content)
            validate_candidate_structure(candidate_source)
        except ParseError as exc:
            return (
                GenerationFailure(
                    stage="structure" if "ModelNew" in str(exc) else "parse",
                    reason=str(exc)[:300],
                    request_sha256=request.request_sha256,
                ),
                request,
            )
        return (
            GenerationSuccess(
                candidate_source=candidate_source,
                candidate_sha256=hashlib.sha256(candidate_source.encode("utf-8")).hexdigest(),
                request_sha256=request.request_sha256,
                finish_reason=response.finish_reason,
            ),
            request,
        )
