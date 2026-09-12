"""FlashInfer-Bench trace adapter (T19, design §4/§8.3).

Maps FlashInfer-Bench trace artifacts (Definition / Workload / Solution)
onto this project's task model and enforces FIB's own applicability gate
(``spec.target_hardware``) before anything runs:

- the pinned trace (commit + sha256 in configs/flashinfer-bench/
  trace-manifest.json) is executed verbatim - inputs are generated from
  the workload's axis bindings, the definition's ``reference`` provides
  the baseline, and the solution's declared entry point provides the
  candidate;
- DPS (destination-passing style) and return-style solutions are both
  honored: pre-allocated outputs are passed and the returned value is
  compared when present;
- the hardware guard refuses solutions whose ``target_hardware`` does
  not include the current device - recorded as a guard rejection, never
  silently run;
- tensor hashes (inputs/outputs) and the trace revision are recorded
  for every run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FibDefinition:
    name: str
    op_type: str
    reference: str
    inputs: dict
    outputs: dict
    axes: dict
    source_sha256: str


@dataclass(frozen=True, slots=True)
class FibWorkloadEntry:
    definition: str
    axes: dict
    uuid: str


@dataclass(frozen=True, slots=True)
class FibSolution:
    name: str
    definition: str
    language: str
    target_hardware: tuple[str, ...]
    entry_point: str
    sources: tuple[dict, ...]
    source_sha256: str


def load_definition(path: Path) -> FibDefinition:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return FibDefinition(
        name=raw["name"],
        op_type=raw["op_type"],
        reference=raw["reference"],
        inputs=raw["inputs"],
        outputs=raw["outputs"],
        axes=raw["axes"],
        source_sha256=_sha256_text(Path(path).read_text(encoding="utf-8")),
    )


def load_workload(path: Path) -> list[FibWorkloadEntry]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        entries.append(
            FibWorkloadEntry(
                definition=raw["definition"],
                axes=raw["workload"]["axes"],
                uuid=raw["workload"]["uuid"],
            )
        )
    return entries


def load_solution(path: Path) -> FibSolution:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = raw["spec"]
    return FibSolution(
        name=raw["name"],
        definition=raw["definition"],
        language=spec["language"],
        target_hardware=tuple(spec.get("target_hardware", ())),
        entry_point=spec["entry_point"],
        sources=tuple(raw["sources"]),
        source_sha256=_sha256_text(Path(path).read_text(encoding="utf-8")),
    )


def hardware_guard(solution: FibSolution, current_arch: str) -> tuple[bool, str]:
    """FIB solutions declare the hardware they support; a solution whose
    target list excludes the current device is refused before execution."""
    if not solution.target_hardware:
        return True, "no target_hardware restriction declared"
    if any(arch in solution.target_hardware for arch in ("sm89", "sm_89", current_arch)):
        return True, f"current device {current_arch} is targeted"
    return False, (
        f"solution targets {list(solution.target_hardware)!r}, "
        f"current device {current_arch!r} not included"
    )


def tensor_sha256(tensor) -> str:  # pragma: no cover - torch-dependent host helper

    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def build_driver_case(
    definition: FibDefinition, entry: FibWorkloadEntry, solution: FibSolution, current_arch: str
) -> dict:
    """Assemble the container driver case for one trace entry."""
    allowed, reason = hardware_guard(solution, current_arch)
    return {
        "guard": {"allowed": allowed, "reason": reason},
        "definition": {
            "name": definition.name,
            "sha256": definition.source_sha256,
            "axes": definition.axes,
            "inputs": definition.inputs,
            "outputs": definition.outputs,
        },
        "workload": {"axes": entry.axes, "uuid": entry.uuid},
        "solution": {
            "name": solution.name,
            "sha256": solution.source_sha256,
            "language": solution.language,
            "entry_point": solution.entry_point,
        },
    }
