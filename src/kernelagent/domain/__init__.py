"""Core domain contracts for the optimization agent.

Pure value objects with explicit invariants; importing this package must never
pull torch, Triton, a model SDK or any adapter (enforced by tests and by the
import-boundary checks introduced in T01).
"""

from kernelagent.domain.errors import ContractError, DomainError, SchemaVersionError
from kernelagent.domain.evidence import EvidenceRef
from kernelagent.domain.implementation import (
    BuildSpec,
    Implementation,
    SourceFile,
    implementation_id,
)
from kernelagent.domain.method import Hypothesis, MethodProposal
from kernelagent.domain.operator import GRANULARITIES, Granularity, IOPort, OperatorSpec
from kernelagent.domain.result import (
    EVALUATION_STATUSES,
    TASK_TERMINAL_STATES,
    EvaluationResult,
    EvaluationStatus,
    TaskResult,
    TaskTerminalState,
)
from kernelagent.domain.serialization import (
    SCHEMA_VERSION,
    content_sha256,
    dumps,
    from_payload,
    loads,
    to_jsonable,
)
from kernelagent.domain.task import TRACKS, OptimizationTask, Track
from kernelagent.domain.workload import Workload

__all__ = [
    "ContractError",
    "DomainError",
    "SchemaVersionError",
    "EvidenceRef",
    "BuildSpec",
    "Implementation",
    "SourceFile",
    "implementation_id",
    "Hypothesis",
    "MethodProposal",
    "GRANULARITIES",
    "Granularity",
    "IOPort",
    "OperatorSpec",
    "EVALUATION_STATUSES",
    "TASK_TERMINAL_STATES",
    "EvaluationResult",
    "EvaluationStatus",
    "TaskResult",
    "TaskTerminalState",
    "SCHEMA_VERSION",
    "content_sha256",
    "dumps",
    "from_payload",
    "loads",
    "to_jsonable",
    "OptimizationTask",
    "TRACKS",
    "Track",
    "Workload",
]
