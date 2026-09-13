# 当前状态与下一步

首用环境：**Ubuntu + NVIDIA GPU**。环境安装见 [Ubuntu 指南](../UBUNTU.md)，状态清单见 [task board](../task-board.json)，有效检查索引见 [current.json](../evidence/current.json)。
本页只保留一份当前快照；历史移交见 Git 历史（及提交 `1b4a306` 前的旧记录）。

## 当前快照（2026-09-13，v1 Alpha 收口）

**主线**：按 `CODEX_REVIEW_AND_LAUNCH_PLAN.md`（launch plan）完成 Task 1–6：
修复全部已复现的可信性缺陷（P0-1 评测篡改、P0-2 恢复预算超支、P1-2 计时证据崩溃、P2-1 guard 误判、P2-2 lint），
交付产品入口 `kernelagent optimize|resume|status`，并在真实 GPU（RTX 4060 Laptop, UUID `GPU-ea248ec5-…`）上
完成固定正确/错误候选与中断恢复三组 Alpha 验收。U3（真实 GLM 从零生成）诚实标记 NOT_RUN，等待用户提供
`MODEL_PROVIDER_API_KEY`。

**LIVE_MODEL 已打通（2026-09-13 晚）**：用户提供 DeepSeek key 后，T12 G5（`generation_loop_smoke --live`，
accepted=true，1598 tokens）与 T16 U3（`alpha_run --mode live`，2 个真实生成候选经 GPU 评测，诚实终态
no_improvement，1952 tokens）双双 PASS；provider=`deepseek-flash`（`thinking:{type:disabled}`）。
修复过程中落地的通用能力：provider URL 归一化、HTTP 错误带 provider 原因、`extra_body`（关闭思考）、
生成 max_tokens 8192、别名/推理耗尽的明确报错。**T12、T16 转 ACCEPTED。**

**能力边界（诚实声明）**：

- Alpha 威胁模型为"合作型候选"：候选在 pinned evaluator 同进程 exec 执行，AST policy 门
  （`inspect_candidate_policy`，ADR-0004）只是 misuse 防线而非安全边界；每份评测/优化报告显式携带
  `candidate_trust=cooperative`、`adversarially_secure=false`；champion 需人工审查。
  可信 MVP 的信任域分离（候选容器只见输入、verdict 由不加载候选的 verifier 产生）按 launch plan Task 7 落在 T23。
- LIVE_MODEL 验收基于 DeepSeek（provider 中立设计，OpenAI 兼容即可）；champion 未产生（模型生成的
  kernel 未过正确性）是记录在案的合法终态，不构成对优化效果的声明。

**任务板**：T00–T10、T12–T15、T12a、T16、T18、T19、T21 **全部 ACCEPTED**（2026-09-13）；
T11/T17/T20/T22–T25 TODO。T20 已预置（镜像未建，CUTLASS tarball 已下载校验，
恢复手册 `docs/handoffs/recovery-runbook.md`）。

## 本轮提交链（launch plan Task 1–6，逐项红-绿）

| 提交 | 内容 |
|---|---|
| `5e9e51f` | Task 1：篡改反例 + `inspect_candidate_policy` AST 门 + ADR-0004 + 报告带 `candidate_trust` 标记 |
| `3d636df` | Task 2：`reconstruct_budget` durable 预算重建；resume 不再超支；中断预留显式保留，反复崩溃收敛 budget_exhausted |
| `3d2db2f` | Task 3：`validate_batch_samples` 严格校验；`confirm_promotion` 对 NaN/0/负样本抛类型化 ValueError 而非 IndexError |
| `db33c6d` | Task 4：dispatch 逐输入逐维 guard（多输入不再相乘）；T20 lint 修复，ruff 全门恢复 |
| `754d8f8` | Task 5：`kernelagent optimize/resume/status` 最小可恢复闭环 + T16 卡片 |
| `69ef32c` | Task 6：Alpha 验收（U1 晋升 CI [1.654,1.706]、U2 拒绝+反馈、SIGKILL→resume 无重复计费）+ pro driver triton tempfile 修复 + [alpha-runbook](../alpha-runbook.md) |

全量回归 468 passed / 3 skipped（Windows 专用），`ruff check src tests examples configs` 干净。

## Alpha 证据（GPU 实测，2026-09-13）

- U1 正确候选：`artifacts/alpha/layernorm-003/report.json` sha256 `b1fec230…`——state=completed，
  champion `fd0eac1c…`（triton LayerNorm 对 eager 晋升 CI [1.654, 1.706]，12 批原始样本）。
- U2 错误候选：`artifacts/alpha/layernorm-003-wrong/report.json` sha256 `313e3494…`——evaluate 阶段
  `correct=False` 被拒，state=no_improvement，反馈含上游 metadata/stderr 尾。
- 中断恢复：`artifacts/alpha/layernorm-007/report.json` sha256 `c6d6bbca…`——SIGKILL 父进程组后
  `--resume`：`action_interrupted` 显式标记、恰好 2 次结算（无重复计费）、孤儿 Exited 容器按 runbook 清理后为 0。

## 遗留与下一步

1. launch plan Task 7（可信 MVP 信任域分离）→ T23 对抗回归——当前主线。
2. 扩展包恢复：T17/T20/T22（T20 预置在案）→ 之后 T24/T25 研究包；T11（外部 KernelAgent 接入）。
3. 暂停中的扩展包：T17/T20/T22（按 launch plan §7 指令暂停；T20 预置见恢复手册）→ 之后 T24/T25 研究包。
4. 一条可选 sudo 配置（T15 深度 profiling 前）：非 root ncu 计数器权限
   （`/etc/modprobe.d/nvidia-profiling.conf` + 重启）。

当前无持续 GPU/模型作业；`docker ps -a --filter label=kernelagent.worker=1` 无残留。
