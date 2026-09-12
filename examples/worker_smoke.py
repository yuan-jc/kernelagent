"""Harmless process-runner example; no GPU or generated code is executed."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from kernelagent.worker import WorkerRequest, execute

outcome = execute(
    WorkerRequest(
        request_id="cpu-smoke",
        argv=(sys.executable, "-c", "print('hello from the worker')"),
        timeout_seconds=10,
        workspace_root=Path("artifacts/worker-smoke"),
        keep_workdir_on_failure=True,
    )
)
print(json.dumps(asdict(outcome), indent=2))
raise SystemExit(0 if outcome.status == "completed" else 1)
