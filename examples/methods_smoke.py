"""T13 acceptance smoke: two generators (template + reuse) flow through
the SAME main loop into the T05 evaluator on real GPU; a domain-mismatched
method and a whitelist-violating method are refused. Report:
artifacts/t13/methods-report.json."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import EvalCase, evaluate_case
from kernelagent.methods import (
    MethodRegistry,
    TaskProfile,
    run_method,
)

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")

TASK = TaskProfile(
    task_id="kernelbench-l1-p040",
    level=1,
    problem_id=40,
    problem_name="40_LayerNorm.py",
    category="layernorm",
)
TEMPLATE_METHOD = (
    "import torch\nimport torch.nn as nn\n\n\n"
    "class ModelNew(nn.Module):\n"
    "    def __init__(self, normalized_shape: tuple):\n"
    "        super(ModelNew, self).__init__()\n"
    "        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)\n\n"
    "    def forward(self, x):\n"
    "        return self.ln(x)\n\n"
    "# rendered for {task_id}\n"
)


def _default_gpu_device() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return "nvidia.com/gpu=0"
    first = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    return f"nvidia.com/gpu={first}" if first else "nvidia.com/gpu=0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/t13"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    (FIXTURES_DEFAULT / "correct_l1_p40_layernorm.py").read_text(encoding="utf-8")
    reuse_source = (FIXTURES_DEFAULT / "correct_l1_p40_layernorm.py").read_text(encoding="utf-8")
    from kernelagent.methods import make_reuse_method, make_template_method

    registry = MethodRegistry()
    registry.register(
        make_template_method("template-layernorm", "layernorm", "MIT", TEMPLATE_METHOD)
    )
    registry.register(
        make_reuse_method(
            "reuse-layernorm",
            "layernorm",
            "project-original",
            {"kernelbench-l1-p040": reuse_source},
        )
    )
    registry.register(
        make_template_method("template-matmul", "matmul", "MIT", "import torch\nx = 1\n")
    )

    problem_source = (args.snapshot_root / "KernelBench/level1/40_LayerNorm.py").read_text(
        encoding="utf-8"
    )
    selected, refusals = registry.select(TASK)

    report_cases = []
    all_ok = len(selected) == 2 and len(refusals) == 1 and "matmul" in refusals[0].reason
    for method in selected:
        candidate, refusal = run_method(method, TASK, problem_source)
        if refusal is not None or candidate is None:
            reason = refusal.reason if refusal else "?"
            report_cases.append({"method": method.name, "refusal": reason})
            all_ok = False
            continue
        eval_result = evaluate_case(
            EvalCase(
                case_id=f"t13-{method.name}",
                level=1,
                problem_id=40,
                problem_name="40_LayerNorm.py",
                candidate_name=method.name,
                candidate_source=candidate.candidate_source,
            ),
            snapshot_root=args.snapshot_root,
            workspace_root=args.output / "workspace",
            gpu_devices=(gpu_device,),
        )
        verified = (
            eval_result.outcome_status == "completed"
            and eval_result.adapter_pass is True
            and eval_result.consistent_with_upstream
        )
        all_ok = all_ok and verified
        report_cases.append(
            {
                "method": method.name,
                "license": candidate.license,
                "dependencies": list(candidate.dependencies),
                "candidate_sha256": candidate.candidate_sha256,
                "outcome_status": eval_result.outcome_status,
                "gpu_verified": verified,
            }
        )
        print(f"{method.name}: gpu_verified={verified}")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "protocol": "t13-methods-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "selected_methods": [m.name for m in selected],
        "refusals": [{"method": r.method_name, "reason": r.reason} for r in refusals],
        "cases": report_cases,
        "accepted": all_ok,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "methods-report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"report={report_path} accepted={all_ok}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
