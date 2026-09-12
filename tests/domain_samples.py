"""Sample builders shared by domain tests. IDs use fixed hex patterns so
hash-format validation passes without calling the hash functions themselves."""

from kernelagent.domain import (
    BuildSpec,
    EvaluationResult,
    EvidenceRef,
    Hypothesis,
    Implementation,
    IOPort,
    MethodProposal,
    OperatorSpec,
    OptimizationTask,
    SourceFile,
    TaskResult,
    Workload,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def make_evidence_ref(**overrides) -> EvidenceRef:
    values = {
        "artifact_sha256": HEX_C,
        "kind": "timing_samples",
        "producer_version": "kernelagent@0.1.0",
    }
    values.update(overrides)
    return EvidenceRef(**values)


def make_io_port(**overrides) -> IOPort:
    values = {"name": "x", "dtype": "float32", "shape": (1024, 1024)}
    values.update(overrides)
    return IOPort(**values)


def make_operator(**overrides) -> OperatorSpec:
    values = {
        "operator_id": "matmul",
        "granularity": "operator",
        "inputs": (make_io_port(name="a"), make_io_port(name="b")),
        "outputs": (make_io_port(name="y"),),
        "notes": "y = a @ b",
    }
    values.update(overrides)
    return OperatorSpec(**values)


def make_workload(**overrides) -> Workload:
    values = {
        "workload_id": "w-square-1k",
        "operator_id": "matmul",
        "shapes": ((1024, 1024), (1024, 1024)),
        "dtypes": ("float32", "float32"),
        "seed": 7,
        "weight": 1.0,
    }
    values.update(overrides)
    return Workload(**values)


def make_task(**overrides) -> OptimizationTask:
    values = {
        "task_id": "matmul-square",
        "operator": make_operator(),
        "workloads": (
            make_workload(),
            make_workload(workload_id="w-rect-2k", shapes=((2048, 512), (512, 2048))),
        ),
        "track": "upstream_compatible",
    }
    values.update(overrides)
    return OptimizationTask(**values)


def make_source_file(path="kernel.py", content="def run(a, b):\n    return a + b\n") -> SourceFile:
    return SourceFile(path=path, content=content)


def make_build_spec(**overrides) -> BuildSpec:
    values = {
        "toolchain": "triton",
        "flags": ("-O3",),
        "env": (("TRITON_PRINT_AUTOTUNING", "1"),),
    }
    values.update(overrides)
    return BuildSpec(**values)


def make_implementation(**overrides) -> Implementation:
    values = {
        "operator_id": "matmul",
        "backend": "triton",
        "entry_point": "kernel:run",
        "source_files": (
            make_source_file(),
            make_source_file(path="wrapper.py", content="from kernel import run\n"),
        ),
        "build_spec": make_build_spec(),
        "applicability_guard": "cuda_compute_capability >= 8.0",
        "parent_ids": (HEX_B,),
        "origin": "method.fusion.epilogue@1",
    }
    values.update(overrides)
    return Implementation(**values)


def make_hypothesis(**overrides) -> Hypothesis:
    values = {
        "statement": "Fusing the epilogue removes one intermediate tensor round trip.",
        "supporting_evidence": (make_evidence_ref(),),
        "predicted_observations": ("fewer kernel launches",),
        "falsification_conditions": ("independent re-measurement shows no gain",),
    }
    values.update(overrides)
    return Hypothesis(**values)


def make_method_proposal(**overrides) -> MethodProposal:
    values = {
        "method_id": "fusion.epilogue",
        "method_version": "1",
        "hypothesis": make_hypothesis(),
        "parameter_space_sha256": HEX_D,
        "estimated_gpu_seconds": 120.0,
        "target_workload_ids": ("w-square-1k",),
    }
    values.update(overrides)
    return MethodProposal(**values)


def make_evaluation_result(**overrides) -> EvaluationResult:
    values = {
        "candidate_id": "cand-1",
        "protocol_sha256": HEX_E,
        "environment_sha256": HEX_F,
        "status": "passed",
        "evidence": (make_evidence_ref(),),
    }
    values.update(overrides)
    return EvaluationResult(**values)


def make_task_result(**overrides) -> TaskResult:
    values = {
        "task_id": "matmul-square",
        "terminal_state": "completed_improved",
        "champion": make_implementation(),
        "best_result": make_evaluation_result(),
        "evidence": (make_evidence_ref(),),
    }
    values.update(overrides)
    return TaskResult(**values)
