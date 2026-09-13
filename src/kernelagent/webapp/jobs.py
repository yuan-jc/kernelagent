"""Job plumbing for the web console: run modes, generators, problem listing.

Modes map 1:1 to the alpha acceptance groups:
- ``live``: real provider generation via the user-supplied API key
  (LIVE_MODEL; this is the only mode that can ever count for U3).
- ``demo-correct`` / ``demo-wrong``: frozen fixture candidates that prove
  the loop on real GPU without a provider - never reportable as
  LIVE_MODEL."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kernelagent.adapters.models.client import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from kernelagent.adapters.models.costing import CostLedger
from kernelagent.adapters.models.generation import CandidateGenerator
from kernelagent.adapters.models.openai_compat import OpenAICompatModelClient
from kernelagent.config import default_generator
from kernelagent.optimization import OptimizationConfigError

MODES = ("live", "demo-correct", "demo-wrong")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")
_DEMO_FIXTURES = {
    "demo-correct": "triton_l1_p40_layernorm.py",
    "demo-wrong": "wrong_l1_p40_layernorm.py",
}


class FixedCandidateClient(ModelClient):
    """Offline stand-in for the provider: always returns one frozen
    candidate (launch plan Task 6 U1/U2 semantics)."""

    def __init__(self, source: str):
        self._source = source
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            request_sha256=request.request_sha256,
            model_id=request.model_id,
            content=json.dumps({"code": self._source}),
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1),
        )


def build_generator(
    mode: str,
    base_url: str,
    api_key: str,
    *,
    fixtures_root: Path = FIXTURES_DEFAULT,
) -> CandidateGenerator:
    """Build the CandidateGenerator for one job; credentials stay local."""
    if mode == "live":
        if api_key.strip():
            client = OpenAICompatModelClient(base_url=base_url, api_key=api_key)
            return CandidateGenerator(client, CostLedger())
        # No key in the request: fall back to the control-plane env var,
        # which raises an explicit config error when it is unset too.
        return default_generator(base_url)
    fixture = _DEMO_FIXTURES.get(mode)
    if fixture is None:
        raise OptimizationConfigError(f"unknown run mode {mode!r}")
    source_path = Path(fixtures_root) / fixture
    if not source_path.is_file():
        raise OptimizationConfigError(f"fixture missing: {source_path}")
    candidate = source_path.read_text(encoding="utf-8")
    return CandidateGenerator(FixedCandidateClient(candidate), CostLedger())


def list_problems(snapshot_root: Path) -> dict:
    """Enumerate the pinned KernelBench snapshot as bench/level/problem."""
    root = Path(snapshot_root)
    kb = root / "KernelBench"
    levels = []
    if kb.is_dir():

        def level_key(path: Path) -> tuple:
            match = re.fullmatch(r"level(\d+)", path.name)
            return (int(match.group(1)) if match else 1 << 30, path.name)

        for level_dir in sorted((p for p in kb.iterdir() if p.is_dir()), key=level_key):
            match = re.fullmatch(r"level(\d+)", level_dir.name)
            if not match:
                continue
            problems = []
            entries = []
            for problem_file in level_dir.glob("*.py"):
                problem_match = re.match(r"^(\d+)_(.+)\.py$", problem_file.name)
                if problem_match:
                    entries.append((int(problem_match.group(1)), problem_match.group(2)))
            for pid, stem in sorted(entries):
                problems.append(
                    {
                        "id": pid,
                        "spec": f"kernelbench:l{match.group(1)}:{pid}",
                        "label": f"{pid}_{stem}",
                    }
                )
            levels.append({"level": int(match.group(1)), "problems": problems})
    return {
        "bench": "kernelbench",
        "note": "tritonbench / flashinfer-bench adapters exist but are not wired into optimize",
        "levels": levels,
    }
