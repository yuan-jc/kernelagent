# Alpha Runbook：`kernelagent optimize` 私人验收与使用

适用：v1 Alpha（`candidate_trust=cooperative`，见 ADR-0004）。champion 晋升后必须人工审查源码。

## 0. 前置

```bash
cd /home/y/kernelagent
export PATH="$PWD/.venv/bin:$PATH"
nvidia-smi                            # 确认 GPU 可用
docker images | grep kernelagent-eval # 确认 ADR-0002 镜像在位
```

## 1. 三组验收命令

```bash
# U1 已知正确固定候选：闭环完成并晋升（不消耗模型；证明的是闭环不是 LIVE_MODEL）
python examples/alpha_run.py --mode correct --output artifacts/alpha/layernorm-001

# U2 已知错误固定候选：不晋升，反馈可用
python examples/alpha_run.py --mode wrong --output artifacts/alpha/layernorm-001 --max-candidates 2

# U3 真实 GLM 从零生成（LIVE_MODEL；需要用户自己的凭据，绝不回放冒充）
export MODEL_PROVIDER_API_KEY='<仅当前 shell 有效>'
python examples/alpha_run.py --mode live \
  --base-url '<OpenAI-compatible base URL>' --model glm-4.5 \
  --max-candidates 5 --max-repair-rounds 2 \
  --output artifacts/alpha/layernorm-001
```

## 2. 断点续跑

任何中断（SSH 断开、kill、崩溃）后：

```bash
python examples/alpha_run.py --mode <同模式> --output <同一目录> --resume
kernelagent status --output <同一目录>   # 查看 durable 状态与预算
```

- journal 是唯一事实：已完成候选不重复执行/计费；被中断候选显式标记 `action_interrupted` 并重新预留。
- 硬杀（SIGKILL）父进程后，在飞容器会自行退出但留下 Exited 记录，需手工清理：
  `docker ps -a --filter label=kernelagent.worker=1 -q | xargs -r docker rm`。

## 3. 预算

- `--gpu-budget-seconds` / `--token-budget`：预留→结算，durable 重放保证恢复后不超支（Task 2）。
- 终态：`completed`(0) / `no_improvement`(1) / `budget_exhausted`(2) / 配置错误(3) / `infra_error`(4)。

## 4. 证据位置

每次运行：`<output>/report.json`（身份：commit、problem sha256、GPU UUID、预算、终态、candidate_trust 标记）、
`<output>/records/*.json`（逐候选阶段记录）、`<output>/champion/`（晋升内核 + champion.json）、
`<output>/journal.jsonl`（hash 链 journal）。

## 5. 停止条件（出现任一立即停止自动运行）

见 `CODEX_REVIEW_AND_LAUNCH_PLAN.md` §5：篡改攻击 PASS、结算超预算、非有限正数计时样本、
候选自报无可信观测、身份字段缺失、容器无法回收、以回放冒充 LIVE_MODEL。

## 6. 已实测记录（2026-09-13，RTX 4060 Laptop）

- U1 correct：`artifacts/alpha/layernorm-003` state=completed，champion `fd0eac1c…`，晋升 CI [1.654, 1.706]。
- U2 wrong：`artifacts/alpha/layernorm-003-wrong` state=no_improvement，evaluate 阶段 `correct=False` 被拒。
- 中断恢复：`artifacts/alpha/layernorm-007` SIGKILL 后 resume：`action_interrupted` 标记、恰好 2 次结算（无重复计费）、无残留容器。
- U3 live：NOT_RUN——等待用户配置 `MODEL_PROVIDER_API_KEY`（诚实记录，不用回放冒充）。
