"""T12 acceptance smoke: the model→code→GPU generation loop.

Uses the frozen smoke problem (LayerNorm) with RECORDED model responses
to prove the full loop machinery: request assembly → client call →
structured parsing → structural validation → GPU evaluation through the
T04 boundary → cost ledger. An error response must produce a failure
record without ever becoming a success.

The LIVE_MODEL row (real provider call) needs user-configured provider
credentials; without them the loop reports live_model="not_run" with the
explicit blocker - it can never be substituted by the recorded replay.

Report: artifacts/t12/generation-loop-report.json; exit 0 requires the
good recorded loop to verify on GPU and the error response to be
rejected."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.adapters.evals import EvalCase, evaluate_case
from kernelagent.adapters.models import ModelUsage
from kernelagent.adapters.models.client import (
    ModelResponse,
    RecordedModelClient,
)
from kernelagent.adapters.models.costing import CostLedger
from kernelagent.adapters.models.generation import CandidateGenerator

SNAPSHOT_DEFAULT = Path("research/sources/ScalingIntelligence__KernelBench")
FIXTURES_DEFAULT = Path("configs/kernelbench/eval-fixtures")
PROBLEM_PATH = Path("KernelBench/level1/40_LayerNorm.py")


def _recorded_good_content() -> str:
    """The recorded 'good' response carries the proven two-pass Triton
    LayerNorm candidate (same source as the T07 fixture), wrapped in the
    JSON envelope the prompt demands."""
    fixture = (FIXTURES_DEFAULT / "triton_l1_p40_layernorm.py").read_text(encoding="utf-8")
    return "```json\n" + json.dumps({"code": fixture}) + "\n```"


GOOD_CONTENT = _recorded_good_content()
BAD_CONTENT = "Sorry, I cannot generate kernels for that request."


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/t12"))
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--gpu-device", default=None)
    parser.add_argument("--live", action="store_true", help="Attempt the real provider call")
    parser.add_argument("--base-url", default="https://api.zhipuai.cn/api/paas/v4/")
    parser.add_argument("--model-id", default="glm-4-flash")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send thinking:{type:disabled} (reasoning-style providers burn the "
        "token budget before emitting content)",
    )
    args = parser.parse_args()
    gpu_device = args.gpu_device or _default_gpu_device()

    problem_source = (args.snapshot_root / PROBLEM_PATH).read_text(encoding="utf-8")

    # Frozen smoke protocol: one good recorded generation + one refusal.
    request = None
    ledger = CostLedger()

    from kernelagent.adapters.models.generation import build_request

    request = build_request(problem_source, args.model_id)
    responses = {
        request.request_sha256: ModelResponse(
            request_sha256=request.request_sha256,
            model_id=args.model_id,
            content=GOOD_CONTENT,
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=180, completion_tokens=310),
        )
    }
    client = RecordedModelClient(responses)
    generator = CandidateGenerator(client, ledger)
    outcome, _ = generator.generate(problem_source, args.model_id)
    generated_ok = hasattr(outcome, "candidate_source")

    verified = None
    if generated_ok:
        eval_case = EvalCase(
            case_id="t12-generated",
            level=1,
            problem_id=40,
            problem_name="40_LayerNorm.py",
            candidate_name="generated-candidate",
            candidate_source=outcome.candidate_source,
            backend="triton",
        )
        eval_result = evaluate_case(
            eval_case,
            snapshot_root=args.snapshot_root,
            workspace_root=args.output / "workspace",
            gpu_devices=(gpu_device,),
        )
        verified = (
            eval_result.outcome_status == "completed"
            and eval_result.adapter_pass is True
            and eval_result.consistent_with_upstream
        )
    else:
        verified = False

    # Refusal case: same problem, different recorded response (seed note
    # changes the request hash, so the recording needs its own entry).
    refusal_request = build_request(problem_source, args.model_id, seed_note=" attempt-2")
    refusal_client = RecordedModelClient(
        {
            refusal_request.request_sha256: ModelResponse(
                request_sha256=refusal_request.request_sha256,
                model_id=args.model_id,
                content=BAD_CONTENT,
                finish_reason="stop",
                usage=ModelUsage(prompt_tokens=180, completion_tokens=8),
            )
        }
    )
    refusal_generator = CandidateGenerator(refusal_client, ledger)
    refusal_outcome, _ = refusal_generator.generate(
        problem_source, args.model_id, seed_note=" attempt-2"
    )
    refusal_rejected = (
        not hasattr(refusal_outcome, "candidate_source") and refusal_outcome.stage == "parse"
    )

    # LIVE_MODEL: only with explicit user-configured credentials.
    live_status = "not_run"
    live_detail = "no provider credentials configured on the control plane (user action)"
    if args.live:
        import os

        api_key = os.environ.get("MODEL_PROVIDER_API_KEY", "")
        if not api_key:
            live_detail = "MODEL_PROVIDER_API_KEY empty; live call refused by policy"
        else:
            from kernelagent.adapters.models.openai_compat import OpenAICompatModelClient

            live_client = OpenAICompatModelClient(
                base_url=args.base_url,
                api_key=api_key,
                extra_body={"thinking": {"type": "disabled"}} if args.disable_thinking else None,
            )
            live_outcome, _ = generator.__class__(live_client, ledger).generate(
                problem_source, args.model_id
            )
            live_status = "ran" if hasattr(live_outcome, "candidate_source") else "failed"
            live_detail = getattr(live_outcome, "reason", "completed")

    accepted = generated_ok and verified is True and refusal_rejected
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "protocol": "t12-generation-loop-v1",
        "commit": commit,
        "gpu_device": gpu_device,
        "generated": generated_ok,
        "generated_candidate_sha256": getattr(outcome, "candidate_sha256", None),
        "gpu_verified": verified,
        "refusal_rejected": refusal_rejected,
        "ledger": {
            "total_tokens": ledger.total_tokens(),
            "total_requests": ledger.total_requests(),
        },
        "live_model": {"status": live_status, "detail": live_detail},
        "accepted": accepted,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "generation-loop-report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(
        f"generated={generated_ok} gpu_verified={verified} refusal_rejected={refusal_rejected} "
        f"live={live_status} accepted={accepted}"
    )
    print(f"report={report_path}")
    print(f"report_sha256={hashlib.sha256(report_path.read_bytes()).hexdigest()}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
