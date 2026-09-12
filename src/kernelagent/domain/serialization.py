"""Versioned JSON serialization and canonical content hashing.

Envelope: ``{"schema_version": "1.0", "type": <class name>, "data": {...}}``.
Reconstruction always re-runs ``__post_init__`` validation, so a payload with
tampered fields is rejected at load time, not silently accepted.

``content_sha256`` hashes the canonical ``{"type", "data"}`` payload only:
it is stable across schema versions and independent of JSON key order, so it
measures object content, not envelope layout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable

from kernelagent.domain.errors import DomainError, SchemaVersionError
from kernelagent.domain.evidence import EvidenceRef
from kernelagent.domain.implementation import BuildSpec, Implementation, SourceFile
from kernelagent.domain.method import Hypothesis, MethodProposal
from kernelagent.domain.operator import IOPort, OperatorSpec
from kernelagent.domain.result import EvaluationResult, TaskResult
from kernelagent.domain.task import OptimizationTask
from kernelagent.domain.workload import Workload

SCHEMA_VERSION = "1.0"


def _reject_unknown_fields(data: dict[str, Any], cls: type) -> None:
    allowed = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise DomainError(f"unknown field(s) for {cls.__name__}: {', '.join(unknown)}")


def _json_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DomainError(f"{field} must be a JSON array")
    return value


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DomainError(f"{field} must be a JSON object")
    return value


def _evidence_ref(data: dict[str, Any]) -> EvidenceRef:
    _reject_unknown_fields(data, EvidenceRef)
    return EvidenceRef(
        artifact_sha256=data["artifact_sha256"],
        kind=data["kind"],
        producer_version=data["producer_version"],
    )


def _io_port(data: dict[str, Any]) -> IOPort:
    _reject_unknown_fields(data, IOPort)
    return IOPort(
        name=data["name"],
        dtype=data["dtype"],
        shape=tuple(_json_list(data["shape"], "IOPort.shape")),
    )


def _operator_spec(data: dict[str, Any]) -> OperatorSpec:
    _reject_unknown_fields(data, OperatorSpec)
    return OperatorSpec(
        operator_id=data["operator_id"],
        granularity=data["granularity"],
        inputs=tuple(
            _io_port(_json_object(item, "OperatorSpec.inputs item"))
            for item in _json_list(data["inputs"], "OperatorSpec.inputs")
        ),
        outputs=tuple(
            _io_port(_json_object(item, "OperatorSpec.outputs item"))
            for item in _json_list(data["outputs"], "OperatorSpec.outputs")
        ),
        notes=data.get("notes", ""),
    )


def _workload(data: dict[str, Any]) -> Workload:
    _reject_unknown_fields(data, Workload)
    return Workload(
        workload_id=data["workload_id"],
        operator_id=data["operator_id"],
        shapes=tuple(
            tuple(_json_list(shape, "Workload.shapes item"))
            for shape in _json_list(data["shapes"], "Workload.shapes")
        ),
        dtypes=tuple(_json_list(data["dtypes"], "Workload.dtypes")),
        seed=data.get("seed"),
        weight=data.get("weight", 1.0),
    )


def _optimization_task(data: dict[str, Any]) -> OptimizationTask:
    _reject_unknown_fields(data, OptimizationTask)
    return OptimizationTask(
        task_id=data["task_id"],
        operator=_operator_spec(_json_object(data["operator"], "OptimizationTask.operator")),
        workloads=tuple(
            _workload(_json_object(item, "OptimizationTask.workloads item"))
            for item in _json_list(data["workloads"], "OptimizationTask.workloads")
        ),
        track=data.get("track", "upstream_compatible"),
    )


def _source_file(data: dict[str, Any]) -> SourceFile:
    _reject_unknown_fields(data, SourceFile)
    return SourceFile(path=data["path"], content=data["content"])


def _build_spec(data: dict[str, Any]) -> BuildSpec:
    _reject_unknown_fields(data, BuildSpec)
    return BuildSpec(
        toolchain=data["toolchain"],
        flags=tuple(_json_list(data.get("flags", []), "BuildSpec.flags")),
        env=tuple(
            tuple(_json_list(pair, "BuildSpec.env item"))
            for pair in _json_list(data.get("env", []), "BuildSpec.env")
        ),
    )


def _implementation(data: dict[str, Any]) -> Implementation:
    _reject_unknown_fields(data, Implementation)
    build_spec = data.get("build_spec")
    return Implementation(
        operator_id=data["operator_id"],
        backend=data["backend"],
        entry_point=data["entry_point"],
        source_files=tuple(
            _source_file(_json_object(item, "Implementation.source_files item"))
            for item in _json_list(data["source_files"], "Implementation.source_files")
        ),
        build_spec=(
            None
            if build_spec is None
            else _build_spec(_json_object(build_spec, "Implementation.build_spec"))
        ),
        applicability_guard=data.get("applicability_guard", ""),
        parent_ids=tuple(_json_list(data.get("parent_ids", []), "Implementation.parent_ids")),
        origin=data.get("origin", ""),
    )


def _hypothesis(data: dict[str, Any]) -> Hypothesis:
    _reject_unknown_fields(data, Hypothesis)
    return Hypothesis(
        statement=data["statement"],
        supporting_evidence=tuple(
            _evidence_ref(_json_object(item, "Hypothesis.supporting_evidence item"))
            for item in _json_list(
                data.get("supporting_evidence", []), "Hypothesis.supporting_evidence"
            )
        ),
        predicted_observations=tuple(
            _json_list(data.get("predicted_observations", []), "Hypothesis.predicted_observations")
        ),
        falsification_conditions=tuple(
            _json_list(
                data.get("falsification_conditions", []), "Hypothesis.falsification_conditions"
            )
        ),
    )


def _method_proposal(data: dict[str, Any]) -> MethodProposal:
    _reject_unknown_fields(data, MethodProposal)
    return MethodProposal(
        method_id=data["method_id"],
        method_version=data["method_version"],
        hypothesis=_hypothesis(_json_object(data["hypothesis"], "MethodProposal.hypothesis")),
        parameter_space_sha256=data["parameter_space_sha256"],
        estimated_gpu_seconds=data["estimated_gpu_seconds"],
        target_workload_ids=tuple(
            _json_list(data.get("target_workload_ids", []), "MethodProposal.target_workload_ids")
        ),
    )


def _evaluation_result(data: dict[str, Any]) -> EvaluationResult:
    _reject_unknown_fields(data, EvaluationResult)
    return EvaluationResult(
        candidate_id=data["candidate_id"],
        protocol_sha256=data["protocol_sha256"],
        environment_sha256=data["environment_sha256"],
        status=data["status"],
        evidence=tuple(
            _evidence_ref(_json_object(item, "EvaluationResult.evidence item"))
            for item in _json_list(data.get("evidence", []), "EvaluationResult.evidence")
        ),
    )


def _task_result(data: dict[str, Any]) -> TaskResult:
    _reject_unknown_fields(data, TaskResult)
    champion = data.get("champion")
    best_result = data.get("best_result")
    return TaskResult(
        task_id=data["task_id"],
        terminal_state=data["terminal_state"],
        champion=(
            None
            if champion is None
            else _implementation(_json_object(champion, "TaskResult.champion"))
        ),
        best_result=(
            None
            if best_result is None
            else _evaluation_result(_json_object(best_result, "TaskResult.best_result"))
        ),
        evidence=tuple(
            _evidence_ref(_json_object(item, "TaskResult.evidence item"))
            for item in _json_list(data.get("evidence", []), "TaskResult.evidence")
        ),
    )


_CONSTRUCTORS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "EvidenceRef": _evidence_ref,
    "IOPort": _io_port,
    "OperatorSpec": _operator_spec,
    "Workload": _workload,
    "OptimizationTask": _optimization_task,
    "SourceFile": _source_file,
    "BuildSpec": _build_spec,
    "Implementation": _implementation,
    "Hypothesis": _hypothesis,
    "MethodProposal": _method_proposal,
    "EvaluationResult": _evaluation_result,
    "TaskResult": _task_result,
}


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise DomainError(f"cannot encode {type(value).__name__} in domain payloads")


def to_jsonable(obj: Any) -> dict[str, Any]:
    type_name = type(obj).__name__
    if type_name not in _CONSTRUCTORS:
        raise DomainError(f"type {type_name!r} is not a serializable domain object")
    return {"schema_version": SCHEMA_VERSION, "type": type_name, "data": _encode(obj)}


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=2, sort_keys=True) + "\n"


def from_payload(payload: str | bytes | dict[str, Any]) -> Any:
    if isinstance(payload, (str, bytes)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DomainError(f"payload is not valid JSON: {exc}") from exc
    else:
        parsed = payload
    if not isinstance(parsed, dict):
        raise DomainError("payload must be a JSON object envelope")
    unknown_envelope = sorted(set(parsed) - {"schema_version", "type", "data"})
    if unknown_envelope:
        raise DomainError(f"unknown envelope field(s): {', '.join(unknown_envelope)}")
    version = parsed.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"unsupported schema_version {version!r}; this build supports "
            f"{SCHEMA_VERSION!r} only; migrate the payload explicitly"
        )
    type_name = parsed.get("type")
    constructor = _CONSTRUCTORS.get(type_name) if isinstance(type_name, str) else None
    if constructor is None:
        known = ", ".join(sorted(_CONSTRUCTORS))
        raise DomainError(f"unknown domain type {type_name!r}; known types: {known}")
    data = parsed.get("data")
    if not isinstance(data, dict):
        raise DomainError(f"envelope for {type_name!r} must contain a 'data' object")
    try:
        return constructor(data)
    except KeyError as exc:
        raise DomainError(f"missing required field {exc} for {type_name}") from exc


def loads(text: str) -> Any:
    return from_payload(text)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(obj: Any) -> str:
    envelope = to_jsonable(obj)
    return hashlib.sha256(
        _canonical_bytes({"type": envelope["type"], "data": envelope["data"]})
    ).hexdigest()
