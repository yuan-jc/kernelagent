# 当前状态与下一步

> 2026-09-13 后续审查：修复跟踪优先级见 [REVIEW.md](../../REVIEW.md) 的 RV01–RV09（2026-09-14 通宵轮已落地 RV01/RV02 核心，见下）。先完成 Windows 可验证修复与 CI，再执行 Ubuntu/GPU 和 U3；下方 Alpha 结果保留为历史快照，不代表这些新发现已修复。

首用环境：**Ubuntu + NVIDIA GPU**。环境安装见 [Ubuntu 指南](../UBUNTU.md)，状态清单见 [task board](../task-board.json)，有效检查索引见 [current.json](../evidence/current.json)。
本页只保留一份当前快照；历史移交见 Git 历史（及提交 `1b4a306` 前的旧记录）。

## 规划增量（2026-09-15，T33 profiling → 方法决策链待办）

- 新增 `docs/work-packages/T33.md`，把当前 baseline NCU → max 聚合 → 单标签规则 →
  方法 signal → 人工先验分数链路，重构为可拆分实施的可信中间层计划。
- 目标方案覆盖 metric alias/单位/范围、launch 关联、时间占比聚合、覆盖和稳定性、可选
  Roofline、多轴 severity/confidence、方法 guards/正反信号、无总分偏序、可选效用分、历史反馈、
  resume 和证据字段，并列出 T33a–T33f 及正反例验收。
- T33 当前为 **TODO**；本文没有改变运行代码、阈值或评分，也没有运行 GPU/LIVE_MODEL 验收。

## 当前增量（2026-09-15，T32 用户友好 CLI 帮助）

- `kernelagent --help` 现说明产品用途、典型工作流、模型环境变量，并引导到子命令帮助；
  `check`、`bench verify`、`probe`、`optimize`、`resume`、`status`、`profile` 均有完整描述、
  参数含义/单位/默认值或 manifest 继承规则、前置条件和可复制示例。
- 明确 `max_repair_rounds` 是失败额度而非额外候选数，GPU budget 是 evaluation/timing/profiling
  的累计租约秒数，profiling 与正式 timing 分离，退出码及凭据边界可在帮助页直接查看。
  `resume --help` 已移除恒真的冗余 `--resume` 参数；实际 resume 行为未改变。
- `README.md` 增加帮助入口及关键语义。CLI 专项 `20 passed`；全量
  `737 passed / 23 skipped / 0 failed`；ruff check 与 format check 均通过。
- 标准 CPU 报告：`artifacts/cli-help-cpu/50fd156e53594190ba1c26e7508ed66e/report.json`
  （sha256 `8ab748bf45872bff69e2a44e1aa6967a3c2c36a8069b28c8c479960feee3c1c2`）。
  本包只改变 CPU CLI 文档层，GPU 与 LIVE_MODEL 均为 `N/A`。

## 当前增量（2026-09-15，T31 编译错误结构化反馈）

- KernelBench 上游 `KernelExecResult.metadata` 现由 control-plane adapter 有界透传；真实 stage port
  不再把 `upstream_metadata` 硬编码为空。metadata 只作诊断证据，候选是否通过仍仅由上游
  `compiled && correctness` 决定。
- evaluate 失败记录新增 `evaluation_diagnostics`：区分 `compilation_error`、`runtime_error`、
  `correctness_error`，保留错误名、核心消息、stderr 尾和有界上游 metadata。编译失败摘要以
  quoted untrusted data 注入下一轮生成；不会进入 correctness/timing/晋升。
- resume 从持久化的 `detail` 重建相同反馈。专项验收
  `uv run --locked pytest tests/test_kernelbench_eval_adapter.py tests/test_optimization_loop.py tests/test_resume_request_parity.py -q`
  为 `56 passed`；全量 `727 passed / 23 skipped / 0 failed`；ruff 全绿。
- 标准 CPU 报告：
  `artifacts/compile-error-feedback-cpu/e3c193f96c8a4e2292491ccbe9225c94/report.json`
  （sha256 `5593af7345e7f7d8bdb4519d036fe43520e9832cacfe9279255108f9a4164314`）。
  本轮未在 Ubuntu/NVIDIA 上触发真实 nvcc/Triton 编译失败，也未调用真实模型，均为 **NOT_RUN**；
  因而 T31 为 `READY_FOR_ACCEPTANCE`，不是 GPU/LIVE_MODEL 已验收。

## 当前增量（2026-09-15，T30 profiling 链路加固）

- 基线：从最新 `origin/main`（`cf5e46c`）创建
  `codex/profiling-integration-hardening`；原本地 `main` 的 8 个未推提交保持不动。
- 优化流程已有的 `baseline timing -> NCU profile -> method plan -> candidate` 链路保留；本轮修复
  stale scratch/report 可被失败重试误用、NCU 非零退出码被 shell `echo` 掩盖、`n/a` 被当成指标值、
  resume 未校验证据内容、失败 profiling 未计入预算等可信性问题。
- profiling 仍与正式 timing 分离；只 profile 父端固定 baseline，不把候选放进 ADR-0003 的
  privileged 诊断租约。真实 profiler 以 hard timeout 作为启动前 GPU 预算预留，完成后按实际 wall time
  结算；失败、权限拒绝与超时同样结算已消耗的租约时间并降级为 static-only。
- 验收：profiling/优化专项 `73 passed`；全量 `722 passed / 23 skipped / 0 failed`；ruff 全绿。
  标准报告：`artifacts/t30-profiling-hardening-cpu/b012f58b97374d76b9c4ad6eb258449a/report.json`
  （sha256 `69d35bf8f9ce6927f15f72227bd010b65e56b3e59dacfb7a3e787bfcc068ed24`）。
- 未验证：本轮没有 Ubuntu/NVIDIA 环境，真实 `.ncu-rep` 与不同 GPU 指标可得性均为 **NOT_RUN**；
  原 T30 的 RTX 4060 GPU PASS 作为历史证据保留，不能替代本轮加固后的 GPU 复验。

## 当前快照（2026-09-14 通宵轮：主循环整合 + 前端 + 生成器 + benchmark 扩展）

**主线**：用户指定的通宵目标轮完成（领导者分派 9 个子 agent，逐包审查提交）。全量测试 **681+ passed / 3 skipped**
（收尾时以 git log 为准），ruff 全绿。提交链（按序）：

| 提交 | 内容 |
|---|---|
| `3870b38` | 调研三件套：前端设计规范（UCAgent 调研+7页IA+API契约）、优化方法目录（38 方法+生成器设计）、benchmark 选型（MKB+rk+user-bench） |
| `197904e` | RV01 核心：版本化 RunManifest + resume 身份守卫（GPU/镜像/协议/problem/model 变更拒绝）+ 配置前置校验（非法 exit 3 零调用）+ PipelineStage/RunState 枚举 + stage_started 事件 |
| `7daf334` / `6a50851` | React+Vite+Tailwind 前端重写：7 页完整实现（暗色主题、阶段时间线、候选泳道、Launch 表单、journal 查看器、Profile 页）；jsdom 冒烟 35/35 真实后端 + 38/38 mock |
| `19d2faa` / `ec20c44` | 修复：runs 摘要 champion null 崩溃；webapp job.json 簿记撞 RV01 输出目录守卫（集成实测发现） |
| `d7a79eb` | webapp API v2：SPA 静态托管+回退、report/journal 游标增量/records（三层脱敏）/workspace 只读端点；穿越攻击反例（含 symlink 逃逸） |
| `849d061` | **优化方法生成器**：瓶颈分类（MISSING 诚实降级）→硬能力守卫（SM90 方法在 SM89 拒绝）→单因素 prompt 注入→method_plan 事件+method_id 归因；resume 重建反馈（**修复 R5**）；planner=None 可回归旧行为 |
| `1e75ecb` | benchmark 三件套：MultiKernelBench（305 文件钉死快照+20 题冻结子集）、GPU-MODE reference-kernels（12 题，上游 eval.py 判定）、自建 user-bench（torch-free loader+examples_v1）；协议身份独立注册+漂移拒绝 |
| `d53753d` | RV02：attempt 身份、结算+完成原子单写、撕裂尾恢复/中部损坏拒绝、单写者锁；5 写入边界故障注入 + 连续双恢复（**修复 R2**，duplicate settlement 反例覆盖） |

**Live 验证（DeepSeek，控制台 UI 发起，GPU 实测）**：6 次真实运行（`artifacts/webui/20260914-{025856,030053,030346,030641,030902,034551,034934}`）：
生成→方法规划→Triton 编译→上游正确性判定→诚实终态全链路真实工作。关键发现：
① provider 将 `deepseek-chat` 别名到 `deepseek-flash`，模型身份校验会正确拒绝（须用 provider 实际模型名，UI 可拉取列表）；
② deepseek-flash 是混合推理模型，必须 `disable_thinking`（UI 勾选框）否则 8192 max_tokens 全被推理吃掉；
③ R3 的固定 300 秒/动作配额确认仍在（600 秒总预算被 baseline+1 候选耗尽）——RV04 修复中；
④ `max_repair_rounds` 实际语义是"容忍的失败候选数"（失败预算），非额外修复尝试数——行为诚实但语义需 ADR 澄清；
⑤ 5 候选×失败容忍 4 的最终实验：方法规划器按序轮转了 **5 种不同方法**（online_streaming→register_pressure_reduction→
two_pass_deterministic→warp_block_staged→autotune_space_design），全部 `compiled=True` 但 `correct=False`，
诚实终态 no_improvement——瓶颈清晰定位在模型生成质量（LayerNorm 数值），而非 agent 链路；与 2026-09-13 会话结论一致。
DeepSeek 消耗合计约 **¥0.08**（余额 19.67）。控制台启动：`set -a; source .env; set +a; .venv/bin/python -m kernelagent.webapp` → http://127.0.0.1:8501。
demo-correct 端到端（034346）：**completed + champion 晋升**，NCU 画像采集 6.18s 如实计费，method_plan 以
`launch_bound / ncu_full` 分类——profile→规划→晋升全证据链在真实 GPU 上闭合。

**已知小问题（前端，待修）**：运行刚结束的轮询间隙 Run 详情可能瞬态显示"未知/预算 NOT_RUN"（刷新即正确）；baseline 泳道行状态徽章在终态误显"运行中"；Dashboard"最佳加速比"瓦片因摘要无 CI 数据恒为 NOT_RUN。

**任务板**：新增 T26（前端控制台）、T27（webapp API v2）、T28（benchmark 三件套）、T29（方法生成器）、T30（profiler 链路），
状态见 task-board；RV01/RV02 记录 REVIEW.md 交付格式节。R5 已修（849d061），R6 已修（197904e 前置校验），
R1 核心/频率已修、R2 已修（均 CPU 层，Ubuntu 复验待 RV07）。

## 遗留与下一步

1. RV03–RV06（REVIEW.md）：反馈与请求持久化核对、分阶段实际用量预算（R3）、独立确认与证据链（R4）、CI 边界测试修复（R7，未动）。
2. T30 profiler 链路收尾（本轮 IN_PROGRESS）：NCU 证据接入方法生成器。
3. RV07 干净 Ubuntu 验收 → RV08 LIVE_MODEL 完整预算跑 → RV09/T23 信任域分离。
4. T11/T17/T20/T22/T24/T25 按原计划；T20 预置在案（`docs/handoffs/recovery-runbook.md`）。
5. 一条可选 sudo 配置（T15 深度 profiling 前）：非 root ncu 计数器权限
   （`/etc/modprobe.d/nvidia-profiling.conf` + 重启）。

## 当前快照（2026-09-13，v1 Alpha 收口，历史）

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
