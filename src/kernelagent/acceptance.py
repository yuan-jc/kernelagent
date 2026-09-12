"""Record real CPU check outcomes. This is not a GPU evaluator or a sandbox.

pytest's exit status alone is insufficient: an all-skipped run exits zero.
We additionally require at least one passing test case and valid JUnit evidence.
Only trusted project tests should be passed to this developer-facing command.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def _counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("Unexpected JUnit root")
    counts = dict(total=0, passed=0, failed=0, errors=0, skipped=0)
    for case in root.iter("testcase"):
        counts["total"] += 1
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def run_checks(tests: Path, output: Path, timeout: float = 300) -> dict:
    """Run pytest in a separate process and return a versioned report.

    Each invocation has its own directory so an old passing report is never
    reused after a crash. Paths and logs are local developer evidence and may
    contain machine paths; publishing them requires a separate review.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    run_id = uuid.uuid4().hex
    run_dir = output.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    junit = run_dir / "junit.xml"
    stdout, stderr = run_dir / "stdout.log", run_dir / "stderr.log"
    command = [sys.executable, "-m", "pytest", str(tests.resolve()), f"--junitxml={junit}"]
    # Stable evidence: unrelated locally-installed plugins/addopts must not change this run.
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process_code = None
    reason = "check execution did not complete"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            completed = subprocess.run(
                command, stdout=out, stderr=err, timeout=timeout, env=env, check=False
            )
            process_code = completed.returncode
            reason = f"pytest exited with {process_code}"
        except subprocess.TimeoutExpired:
            reason = "pytest timed out"
        except OSError as exc:
            reason = f"could not start pytest: {type(exc).__name__}"
    counts = dict(total=0, passed=0, failed=0, errors=0, skipped=0)
    valid_junit = False
    if junit.is_file():
        try:
            counts = _counts(junit)
            valid_junit = True
        except (ET.ParseError, OSError, ValueError):
            reason = "missing or invalid JUnit evidence"
    success = (
        process_code == 0
        and valid_junit
        and counts["passed"] > 0
        and counts["failed"] == 0
        and counts["errors"] == 0
    )
    if success:
        reason = "at least one test passed and no checks failed"
    elif process_code == 0 and counts["passed"] == 0:
        reason = "no passing tests (empty or all skipped); acceptance denied"
    artifacts = [
        {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (stdout, stderr, junit)
        if p.is_file()
    ]
    report = {
        "schema_version": "1.0",
        "kind": "cpu_test_run",
        "run_id": run_id,
        "status": "PASS" if success else "FAIL",
        "reason": reason,
        "exit_code": 0 if success else 1,
        "pytest_exit_code": process_code,
        "started_at": started_at,
        "duration_seconds": time.monotonic() - started,
        "command": command,
        "environment": {"python": sys.version.split()[0], "platform": sys.platform},
        "counts": counts,
        "artifacts": artifacts,
        "report_path": str(run_dir / "report.json"),
    }
    # Directory is unique; atomic replace keeps readers from seeing partial JSON.
    temporary = run_dir / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "report.json")
    return report
