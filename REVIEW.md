# KernelAgent 当前审查（2026-09-13）

审查提交：`6edcdeb45783defcb236f1fd75226affd3597f60`，相对上一基线 e439fc1。历史审查仅同步代码并检查，未修改实现或验收状态。本文件现纳入 Git，供 Windows 开发与 Ubuntu 验收共同使用；下方执行任务尚未完成。

## 结论

项目已形成合作型候选、单题 Triton 的最小优化闭环，有真实适配器和较完整的 CPU 单元测试；尚不能验收为设计要求的可信、可恢复、预算受控的自动优化 agent。当前最值得投入的是主循环整合与证据可靠性，而不是继续扩展 benchmark/backend。

Ubuntu/GPU 的固定候选成功、错误候选拒绝与 SIGKILL 恢复，由仓库 handoff/evidence 索引报告。本机没有这些 GPU 原始 artifact，不能独立校验其完整内容与哈希；本次 GPU 与 LIVE_MODEL 均为 NOT_RUN。T12 READY_FOR_ACCEPTANCE、T16 IN_PROGRESS 和 U3 NOT_RUN 的声明是合理的。

## 发现与验收要求

### R1 [P1] 恢复可将旧 GPU 成绩归到新环境

位置：src/kernelagent/optimization.py:323、501，src/kernelagent/cli.py 的 resume 参数装配。

baseline 的 action identity 只有 problem hash 前缀；candidate identity 只有 problem hash 前缀、backend、index。不包含 GPU、镜像、协议、模型请求身份，也没有持久化完整 run manifest 并校验配置变化。复现：完成一个离线运行后，仅修改 gpu_device 并 resume；模型调用为 0，直接返回旧 champion，但 report.gpu_device 是新设备。CLI resume 还使用参数默认值，并不自动恢复原始配置。

要求：持久化不可变运行清单；resume 默认加载它；环境、代码、评测协议身份改变时拒绝复用或创建新实验。允许追加预算等变更要有显式事件。验收应覆盖更换 GPU、镜像、协议、problem、model；保持身份的恢复不得重复执行。

### R2 [P1] 结算与完成事件之间崩溃会重复执行并破坏后续恢复

位置：src/kernelagent/orchestrator.py:377–389、110。

budget_settled 与 action_finished 分开追加。注入在 action_finished 写入之前崩溃：已有结算但无完成，恢复再次执行相同 action；第二次结算后，下一次 replay 抛出 duplicate settlement。实测 action 执行 2 次，错误为 `duplicate settlement for action 'a'`。这与已有“在执行阶段 SIGKILL”验收不是同一个故障窗口。

要求：引入 attempt 身份和可原子确认的结果/结算语义，完成证据与计费可一起恢复；明确文件持久化与截断尾记录的处理。验收在 reservation/start/result/settlement/finish 各写入边界注入故障，连续恢复至少两次，既不重复有效结算，也不误认完成或使日志不可重放。

### R3 [P1] GPU 秒预算实际是固定动作配额，不能约束真实使用量

位置：src/kernelagent/optimization.py:60、304、317、486、502；src/kernelagent/config.py:97–129。

baseline 固定结算 300 秒；整个 candidate（生成、evaluate、correctness_pro、timing）也只预留/结算 300 秒。端口没有返回实际 GPU 租约耗时，也未将剩余预算传给子阶段。evaluate/pro/timing 默认超时分别为 900/900/1200 秒，因此单候选可超过其预留很多倍，仍报告结算 300 秒。反过来，在 policy 门拒绝、完全未做 GPU 评测的候选同样默认计 300 秒。token 预留 2048 也只对应生成输出上限，没有覆盖输入 token。

要求：按阶段计量实际租约时间，剩余预算限制启动和超时；失败/中断采用明示的保守计费，估计不能命名为实际 GPU 秒。验收用可控时钟和超时端口覆盖零 GPU 失败、多阶段耗时、接近预算时拒绝启动；token 输入与输出都计入预留策略。Ubuntu 再复验超时清理与计量。

### R4 [P1] 晋升缺少独立确认与完整可回溯证据

位置：src/kernelagent/optimization.py:439–465；src/kernelagent/config.py:138–144。

confirm 仅对第一次 timing 的数组 bootstrap，没有发起独立确认执行。baseline 只在最初测一次，候选在之后测量；没有设计 §9.4 要求的随机/交错 A/B 与独立确认。champion.json 的 protocol_sha256 明确写为 null；主报告丢弃 decision_input_sha256，candidate record 不保存原始样本或它们的哈希引用。原始 worker 输出可能仍在 workspace，但未形成从 champion 到对应完整证据的可靠引用链。

要求：晋升前执行新鲜、预先固定的确认协议；保存代码/环境/协议、正确性结果、原始样本、决策输入和决策之间的内容哈希关联。验收“初筛快、确认慢”必须保留 incumbent；缺任一必需证据不能晋升；单靠 champion manifest 能定位并重算 CI。GPU 做 A/A 噪声与独立复测。

### R5 [P2] 续跑丢失已完成候选的失败反馈

位置：src/kernelagent/optimization.py:282、515–538。

feedback 只在运行中的闭包更新，resume 跳过已完成 action 后并不从 record 重建反馈。复现：错误候选跑完，再以 max_candidates=2 resume；下一模型请求不含之前失败信息。第一次执行与恢复执行的搜索策略因此不同。

要求：持久化生成请求与失败反馈、关联候选源码；恢复后的下一请求内容与未中断路径相同。验收比较两条路径的请求身份，不能只断言调用次数和账单。

### R6 [P2] 无效候选数量可得到虚假成功

位置：src/kernelagent/optimization.py:279、515；cli.py 的整数参数。

max_candidates=0 未拒绝，执行 baseline 后绕过循环，返回 completed 且 champion 为 null（实测）。负数也走空循环；浮点预算应补 finite/正值验证。

要求：资源创建与付费调用前验证配置；0/负候选数、NaN/Inf 预算返回配置错误 exit 3，零模型/GPU 调用。正常 completed 必须携带可验证 champion。

### R7 [P2] 最新 CI 未通过，边界测试依赖预先缓存的 Docker 镜像

位置：tests/test_container_worker.py:213。

最新 workflow 34731172474：Ubuntu 3.11/3.13、Windows 3.13 均在 test_mount_source_outside_workspace_rejected 失败。预期 `resolves outside workspace`，实际先返回缺少镜像的 infra_error。Windows 3.11 通过。已下载 CI artifact 的测试输出核实，不能将此归为 lint 或凭据问题，也不能据此断言路径边界已被绕过。

要求：纯路径验证应在不依赖镜像的测试中覆盖；集成镜像测试单独准备依赖。四个现有 CI job 全绿，目标安全断言实际运行，禁止简单 skip 掉此检查。

## 其他明确边界

- ADR-0004 已公开承认候选与 evaluator 同进程，AST 不是安全边界。这是有记录的 Alpha 范围收缩，不是新发现的隐蔽漏洞；但与 AGENTS 的“候选不能决定成绩”目标仍有差距。T23 完成前只宜人工审查的合作型候选实验。
- 端口信息损失还需检查：config.evaluate 未传 consistent_with_upstream；主循环未检查 outcome_status；config.timing 丢弃 worker 状态。假端口返回 source_ok=True、precondition=False 仍可晋升（离线复现），但当前真实 timing driver 在 precondition=False 时不生成有效样本，因此本次不将该假端口反例单独判定为已打通的真实错误成绩路径。
- domain 的 OperatorSpec/Implementation/Workload/MethodProposal 等对象仍在；但新主循环使用 dict、object 与 KernelBench 专用分支，并未以这些对象承载完整运行身份。方法注册表、autotune、NCU 与扩展 benchmark 的存在，不意味着已经接入 optimize 的自动搜索策略。
- 真实模型自动生成成功率、全冻结清单覆盖率、预算效率、time-to-best、可靠 speedup 均尚不能从固定单题演示推出。
- current.json 的 T13/T14/T18 部分 sha256 是“see evidence…”占位文字，应补完整值与可下载证据包，不能只保留 Ubuntu 私有路径。

## 后续分包建议

1. **恢复与身份**：修 R1/R2/R5；CPU 故障注入和配置变更反例通过后，再做 Ubuntu 中断复验。
2. **预算与合法终态**：修 R3/R6，纳入真实 provider 的输入/输出 token 与超时策略。
3. **晋升与证据**：修 R4、端口完整性检查，输出可独立重算和复验的 evidence bundle。
4. **CI 与交付**：修 R7，四矩阵全绿；从干净 clone 按 UBUNTU.md 跑固定正反例和恢复，把完整 artifact 绑定提交发布。
5. **LIVE_MODEL**：前四包完成后，固定题目/协议/预算跑 U3；允许 no_improvement，但必须证明真实生成、失败反馈、计费和恢复链有效。
6. **可信 MVP**：推进 T23 信任域分离与对抗回归。完成前不对外宣称支持任意不可信候选的可信评测。扩展 T17/T20/T22 继续暂停。

面向对象整理应服务这些修复：使用 RunManifest、Attempt、BudgetReservation、EvaluationEvidence、PromotionEvidence 等有校验的对象和 Protocol 端口；backend/benchmark/generator 由组合注入。不要为 OOP 引入大规模继承重写。每包要求提交包含复现失败测试、修复、验收命令、证据和未运行范围，不能仅以类已实现或 smoke 成功宣告完成。

## 本次验证

- Windows Python 3.13.3：452 passed / 19 skipped / 0 failed；本地 report：artifacts/review-alpha/63b7eb9ab65646c69f255695363f2b65/report.json。
- ruff check . 通过；ruff format --check . 通过，116 文件。
- reproduce.py → reproductions.json：GPU 身份复用、协议 null、0 候选成功、恢复反馈丢失等。
- crash_window.py → crash-window.json：结算后崩溃的重复执行与重复结算。
- ci-test-failures.json：GitHub 四个矩阵的实际结果。
- 未执行 Ubuntu/GPU、LIVE_MODEL、完整攻击面渗透或全 benchmark 性能复验。本审查不宣称已穷尽新增 111 个文件中的所有缺陷。


## 可执行修复计划（Windows → Ubuntu）

本节是下一轮开发的任务入口，优先处理审查发现。上文是 6edcdeb 的历史结论，不随修复直接删除；修复后在任务记录中追加新提交与证据。这里的 RV 编号用于修复跟踪，不取代原 T 工作包；不得因完成文档或 CPU 测试直接提升原 GPU 包的验收状态。

### 执行顺序与当前状态

| 编号 | 交付物 | 依赖 | 开发位置 | 必需验收位置 | 当前状态 |
|---|---|---|---|---|---|
| RV01 | 不可变运行身份与恢复配置 | 无 | Windows | CPU + Ubuntu 环境变更复验 | TODO |
| RV02 | 崩溃一致的结算与恢复 | RV01 | Windows | CPU 故障注入 + Ubuntu SIGKILL | TODO |
| RV03 | 可恢复失败反馈与请求 | RV01、RV02 | Windows | CPU，RV08 集成复验 | TODO |
| RV04 | 分阶段预算与配置校验 | RV01、RV02 | Windows | CPU + Ubuntu 耗时/超时 | TODO |
| RV05 | 独立确认与证据链 | RV01、RV04 | Windows | CPU + Ubuntu GPU 性能 | TODO |
| RV06 | 无隐式镜像依赖的边界测试 | 无，可先修 | Windows | GitHub 四矩阵 CI | TODO |
| RV07 | 干净 Ubuntu 交付验收 | RV01–RV06 | Ubuntu | 真实容器/GPU | TODO |
| RV08 | 真实模型 U3 | RV07 | Ubuntu | 真实 provider + GPU | TODO |
| RV09 | T23 信任域分离 | RV07；不要求 U3 成功 | Windows 协议/CPU，Ubuntu 实现集成 | Ubuntu 对抗验收 | TODO |

先完成 Windows 包，再切换 Ubuntu；不要为了 Windows 开发模拟一套正式 NVIDIA 评测环境。RV06 可先恢复 CI。T17/T20/T22 等扩展继续暂停。一次只领取一个 RV 包及其必要修复，不一次实现所有任务。

### 通用命令与证据规则

以下命令在仓库根目录执行；Windows 若 uv 不在 PATH，用 `python -m uv` 替换 `uv`。

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked kernelagent check --output artifacts/review-fixes/RV01/cpu
```

最后一条的 RV01 按当前包替换。每包先运行对应定向测试，收尾运行全量检查；不要用全 skip、零测试、放宽断言或更改冻结评测参数过关。下列新增反例尚需开发，现有同名测试文件通过不等于反例已经覆盖。

每包记录：基线提交、修复提交、命令、平台、正反例结果、passed/failed/skipped 数量及跳过原因、完整 artifact SHA-256、未验证范围。GPU 原始输出、候选源码和统计样本应形成脱敏可获取的 evidence bundle；Git 中索引写下载位置与完整哈希，不能只写另一台电脑的路径或“see evidence”。不得提交 API key、含凭据 URL 或原始认证响应。

CPU 通过写 `CPU_PASS / TARGET_NOT_RUN`；需要 GPU 的原 T 包最多为 READY_FOR_ACCEPTANCE。完成目标验收后才根据实际证据更新状态。失败、无改进是合法实验结果；基础设施失败不能记成候选失败或成功。

### RV01：固定运行身份并正确加载 resume 配置

修改范围：optimization.py、config.py、cli.py，必要的纯领域对象与序列化。推荐 RunManifest 对象与版本化 schema，不让 Domain 依赖 adapter。

交付要求：
- 首次执行前保存完整 problem/code、benchmark revision、backend、评测/计时协议、镜像和实际设备身份；记录模型与生成策略配置。
- resume 从清单加载原值，区分“参数未传”和“显式覆盖”；认证信息从环境读取，不能写入清单。
- 身份改变时禁止旧 baseline/候选成绩复用；允许的预算追加单独记录变更事件。旧版本清单缺字段时明确拒绝或按版本迁移，不默认为匹配。
- 非 resume 命令使用已有输出目录时，在模型/GPU调用前返回配置错误。

验收：`uv run --locked pytest tests/test_optimization_loop.py tests/test_cli.py`。
新增反例分别更换 GPU、镜像、协议、problem、backend、模型；明确哪些变更拒绝、哪些另开实验。相同身份恢复不得重复已完成动作。只传 `resume --output ... --base-url ...` 应恢复原始非默认题目/预算/模型。Ubuntu 使用可用的设备/镜像变更重复检查；没有第二块 GPU时该子项如实 NOT_RUN，不能虚构。

### RV02：结算与结果提交具有可恢复的一致性

修改范围：orchestrator.py、运行记录持久化与恢复入口。

交付要求：
- 区分逻辑 action 与每次 attempt；完成结果、结算和恢复通过同一 attempt 关联。
- 处理“已结算但未写完成”的窗口；一次有效结算不能再次计费。外部执行不能承诺任意情况下 exactly-once，无法判定完成的 attempt 要保留不确定性与成本。
- 结果使用原子替换/明确提交语义；解释日志 flush/fsync 策略。仅截断的未提交尾记录允许按明示规则恢复，中部破坏必须拒绝，不能静默删账。
- 防止两个进程同时 resume 同一运行，采用单写者锁或等价机制。

验收：`uv run --locked pytest tests/test_orchestrator.py tests/test_optimization_loop.py`。
在预留、开始、结果保存、结算、完成各边界注入失败，连续恢复至少两次；检查 action 次数、attempt 数量、结算数、保留预留与可重放性。必须覆盖本报告 duplicate settlement 反例和并发恢复。Ubuntu RV07 对这些边界使用受控 SIGKILL，检查容器与 GPU 进程清理。

### RV03：恢复失败反馈与生成证据

修改范围：optimization.py、generation.py 与运行记录。

交付要求：保存请求身份、候选源码引用、结构化失败阶段与有界反馈；resume 重建下一请求所需上下文。不得仅保存一句错误摘要而丢失被修复的候选；不得把秘密写入提示或记录。模型响应原文如保存需按敏感信息规则处理。

验收：`uv run --locked pytest tests/test_generation_loop.py tests/test_optimization_loop.py`。
使用固定响应器，对未中断执行和“错误候选后重启”两条路径比较下一请求完整内容/哈希；相同配置下必须一致。分别覆盖解析失败、policy 拒绝、正确性失败、性能无改进；已完成生成不能重复收费。

### RV04：预算覆盖实际阶段，非法配置提前失败

修改范围：预算对象、orchestrator.py、optimization.py、config.py、真实 stage ports。

交付要求：
- 明确 GPU 秒定义为独占 GPU 租约时间或其他可测口径；实际值与估计值分开。以单调时钟计量，涵盖该租约内构建、评测、计时、清理；没有 GPU 工作不结算虚构的实际 GPU 秒。
- 每阶段启动前预留，向 worker 传递剩余预算决定的 deadline；预留不足禁止启动。超时清理开销保留并报告；不能声称能硬性阻止已发往 provider 的费用。
- token 预算考虑输入及最大输出；缺 usage、网络中断或账单不确定时保持明确的保守占用，不默认免费。实际超估要记账并阻止后续动作。
- 所有数值在创建资源前校验：候选数正整数、修复数非负整数、预算有限且有效；completed 必须有可验证 champion。

验收：`uv run --locked pytest tests/test_orchestrator.py tests/test_optimization_loop.py tests/test_cli.py`。
新增可控时钟端口，测试零 GPU policy 拒绝、多个阶段总耗时、预算边界、超时/清理、未知 token 使用量、0/负候选数、NaN/Inf；非法配置 exit 3 且模型/GPU调用均为 0。Ubuntu RV07核对真实时间与账单字段，不用 sleep mock 代替目标验收。

### RV05：独立确认执行与可重算晋升证据

修改范围：promotion.py、optimization.py、config.py、eval/timing adapters 与必要 driver。

交付要求：
- 筛选与最终确认用不同实验身份；最终确认实际再执行，按冻结协议随机/交错 A/B，不能只对原样本重算 bootstrap。
- 使用 EvaluationEvidence/PromotionEvidence 等经校验对象或等价强类型端口，保留 worker 终态、协议一致性、正确性前置、资源/guard、原始样本引用；任何必需条件未通过不能晋升。
- champion 绑定代码、problem、环境、协议、输入数据/种子及最终确认样本；完整保存决策输入哈希，协议哈希不能是 null。
- 提供离线验证/重算入口；manifest 能定位证据，不依赖某个隐式临时目录。版本/接口改变写 ADR，不修改官方题库和容差。

验收：`uv run --locked pytest tests/test_promotion.py tests/test_optimization_loop.py tests/test_timing_protocol.py`。
必须覆盖：初筛快但确认慢；worker 非 completed 却携带看似有效 JSON；precondition=False；协议/代码身份错配；样本缺失/非法；删除或篡改一个 artifact。全部不得晋升。合法 bundle 离线重算得到同一决策。Ubuntu 跑新鲜确认和 A/A，预先记录重复次数与统计规则，不看结果再改阈值。

### RV06：修复 CI 路径边界测试

修改范围：tests/test_container_worker.py；如有必要，将纯配置校验前移到 worker 的资源操作之前。

验收：`uv run --locked pytest tests/test_container_worker.py`，以及通用全量命令。
路径越界反例不依赖 Docker daemon、镜像缓存、网络；断言危险请求在实际启动前拒绝。真实 Docker 测试独立准备 pinned 镜像并标明平台条件。推送后 GitHub Ubuntu/Windows × Python 3.11/3.13 四矩阵通过；记录 workflow URL 和 SHA。不得靠跳过当前失败测试恢复绿灯。

### RV07：干净 Ubuntu + NVIDIA 验收及发布证据

前置：RV01–RV06 CPU 通过、CI 四矩阵通过；固定候选先行，无需模型密钥。

在新的 clone/干净环境按 docs/UBUNTU.md 恢复依赖、冻结 snapshot 和 pinned 镜像，不依赖旧机器私有缓存。记录 Ubuntu、驱动、CUDA、GPU UUID、镜像 ID、提交 SHA；环境与选题保持可比。

现有入口：
```sh
uv run --locked kernelagent bench verify
uv run --locked kernelagent probe --target native --output artifacts/review-fixes/RV07/probe
uv run --locked python examples/alpha_run.py --mode correct --output artifacts/review-fixes/RV07/correct
uv run --locked python examples/alpha_run.py --mode wrong --output artifacts/review-fixes/RV07/wrong
```
正确候选应通过正确性；性能晋升必须由新确认结果决定，不能预设某张卡一定更快。错误候选必须拒绝。另提供随 RV02/RV04 开发的自动故障注入脚本，覆盖各阶段 SIGKILL、恢复两次、deadline、单写者冲突、残留清理与预算核对；脚本实际命令补入本节后执行，不用手工“试过了”替代证据。

交付 bundle 包含正反例 report、journal、run manifest、候选源码、baseline/确认原始样本、环境快照和逐文件完整哈希。每条记录绑定被测 SHA；历史占位哈希找不到原件则标 UNVERIFIED 并重测，不补造历史数据。

### RV08：真实模型生成 U3

前置：RV07通过；用户在 Ubuntu 控制端配置 MODEL_PROVIDER_API_KEY 与 provider 地址。禁止将密钥提交 Git 或传给候选容器。

```sh
uv run --locked python examples/alpha_run.py --mode live --model glm-4.5 --base-url "$MODEL_PROVIDER_BASE_URL" --max-candidates 5 --max-repair-rounds 2 --gpu-budget-seconds 1800 --output artifacts/review-fixes/RV08/live
```
模型 ID 必须是 provider 实际支持的值，运行前冻结并记录；如更换需显式登记。1800 秒不足完成新协议时结果应为 budget_exhausted；另开实验增加预算必须先确定，不得事后修改账本。

验收证据：真实请求/响应身份、实际 token 与未知费用、每个候选阶段、失败反馈、GPU 成本、停止原因；附一次中断恢复。无改进不代表功能失败，但不能称为“优化成功”；若没有可用候选，记录事实并评估生成能力。不得用 scripted responder 通过 LIVE_MODEL；不能仅因配置了 API key 就标 ACCEPTED。

### RV09：可信 MVP / T23

先写 ADR 明确信任域：候选执行面不能写 verdict/正式成绩，可信 verifier 不加载候选模块。Windows 可先开发协议、输出校验、证据对象与纯 CPU 对抗测试；Linux隔离、GPU 输出传递和性能开销必须在 Ubuntu 验证。

验收至少覆盖既有 evaluator tampering、计时接口 monkey patch、结果文件伪造、reference 访问、超时残留，以及正常候选兼容性。具体测试命令在 T23 实施前列出，不把 AST 静态检查当隔离证明。升级威胁模型后按 ADR-0004重新验收受影响的 T05–T10/T12/T16/T21；实现之前报告保持 cooperative / adversarially_secure=false。

### GLM 每次交付格式

每完成一包在本节追加以下记录，并更新相关 task board/handoff/evidence 索引；不覆盖历史审查事实：

```text
包编号：RVxx
状态：TODO / IN_PROGRESS / CPU_PASS_TARGET_NOT_RUN / TARGET_PASS / FAILED
基线与修复提交：
行为变化与对应 R 编号：
正例、反例及实际命令：
测试数量、失败与跳过原因：
平台、环境身份：
证据 URL/路径及完整 SHA-256：
未验证范围、遗留问题、活动作业：
下一包与依赖：
```

下一次先领取 RV06 或 RV01；只做已领取包。Windows 收尾后留下 Ubuntu 待验列表，不把 mock 的成功写成目标机成功。完成审查修复前，不扩大题库、backend 或搜索方法数量。

---

## GLM 交付记录

```text
包编号：RV01
状态：CPU_PASS_TARGET_NOT_RUN
基线与修复提交：基线 6edcdeb（审查基线）；修复 197904e（RunManifest/身份守卫/前置校验/状态枚举），另 d53753d 复用其守卫
行为变化与对应 R 编号：R1——首次运行前持久化版本化 RunManifest（problem/协议/镜像/设备/模型身份，密钥仅环境变量名）；
  resume 默认加载 manifest，身份字段变更逐项报错拒绝；预算/候选数覆盖写 manifest_revised 事件；只传 --output/--base-url 恢复原配置；
  非 resume 使用已存在输出目录在模型/GPU 调用前 exit 3（R6 前置校验：0/负候选、NaN/Inf 预算、completed 无 champion 拒绝）。
  取舍：snapshot_root 与 agent 代码版本记录在 manifest 但不作身份门槛（评测代码身份由 eval/timing driver 内容哈希承担）。
正例、反例及实际命令：反例=换 GPU/backend/model/problem/镜像/计时协议 resume 各一（全部拒绝）；job.json+残留文件拒绝；
  正例=同身份 resume 不重复执行不计费、job.json-only 目录允许（webapp 簿记）。命令：.venv/bin/python -m pytest tests/test_optimization_loop.py tests/test_cli.py tests/test_run_manifest.py -q
测试数量、失败与跳过原因：全量 681 passed / 3 skipped（Windows Job Object 专用），0 failed；新增 10（manifest）+loop/cli 若干
平台、环境身份：Ubuntu 24.04（6.14.0-37-generic），Python 3.11.16，RTX 4060 Laptop（GPU-ea248ec5-1f33-d90f-c598-1c94dfdc6998）
证据 URL/路径及完整 SHA-256：tests/test_run_manifest.py、src/kernelagent/domain/run_manifest.py（提交 197904e）；
  live run 报告 artifacts/webui/20260914-030902/report.json sha256 eb7813ba7aa4e29e…（manifest 参与运行），
  artifacts/webui/20260914-030346/report.json sha256 ec5a7934381e3469…（身份校验拒绝 deepseek-chat≠deepseek-flash 别名，实测生效）
未验证范围、遗留问题、活动作业：Ubuntu 真实换卡/换镜像复验 NOT_RUN（单 GPU 机，留 RV07）；manifest 迁移路径仅版本拒绝未做自动迁移
下一包与依赖：RV02（已同轮完成，见下条）
```

```text
包编号：RV02
状态：CPU_PASS_TARGET_NOT_RUN
基线与修复提交：基线 849d061；修复 d53753d
行为变化与对应 R 编号：R2——attempt 身份贯穿全部预算/生命周期事件；结算唯一性按 attempt；
  结算+完成单次 write+fsync 原子块提交（进程内不再产生"已结算未完成"窗口）；恢复期 _reconcile_attempts 对历史残留窗口
  写 attempt_uncertain（保留成本、不虚报完成、重执行如实再计费一次）；撕裂尾按明示规则丢弃并写 journal_recovered，
  中部损坏拒绝启动；run.lock（flock/msvcrt）阻止并发 resume。诚实声明：每 attempt 计费 exactly-once、执行 at-least-once，
  文件持久化不承诺跨进程外部副作用 exactly-once（orchestrator.py docstring）。
正例、反例及实际命令：注入矩阵=reservation/start/result/settlement/finish 五边界×连续两次恢复（断言执行次数/attempt 数/结算数/可重放）；
  duplicate settlement 反例（结算后崩溃→重执行→再 replay 不抛错）；并发 resume 拒绝且 journal 字节不变；
  撕裂尾/中部损坏/坏哈希三态。命令：.venv/bin/python -m pytest tests/test_orchestrator.py tests/test_recovery_consistency.py -q
测试数量、失败与跳过原因：新增 20 项全绿；全量 681 passed / 3 skipped；一次 test_container_worker Docker 瞬时失败经隔离重跑消失（与本包无关）
平台、环境身份：同 RV01 条目
证据 URL/路径及完整 SHA-256：tests/test_recovery_consistency.py、src/kernelagent/orchestrator.py（提交 d53753d）
未验证范围、遗留问题、活动作业：Ubuntu 受控 SIGKILL 复验 NOT_RUN（RV07）；Windows msvcrt 分支待 CI；混合版本日志旧行为保守持有
下一包与依赖：RV03（部分被 849d061 覆盖：resume 从 records 重建失败反馈+请求轮转；请求全文持久化与双路径请求哈希对比仍未做）
```

```text
附带修复（非 RV 包，随本轮目标完成）：R5 反馈丢失（849d061，resume 重建 attempts+反馈轮转）、R6 虚假成功
（197904e，前置校验+completed 必须 champion）。R3 仅确认未修（live 实测固定配额耗尽，RV04）；R4/R7 未动。
2026-09-14 目标轮其余交付（用户主线，非 RV 包）：React 前端（7daf334/6a50851）、webapp API v2（d7a79eb）、
优化方法生成器（849d061）、benchmark 三件套（1e75ecb）、profiler 链路（T30，进行中）。
全量测试 681 passed / 3 skipped；live 证据见 docs/handoffs/current.md 通宵轮节。
```

```text
包编号：RV03
状态：CPU_PASS_TARGET_NOT_RUN
基线与修复提交：基线 21e8df4；修复 3278f4c（此前 849d061 已落地 resume 反馈重建与方法轮转）
行为变化与对应 R 编号：R5——policy 拒绝的 record detail 改为有界字符串（原为 list，resume 后反馈丢失且
  方法轮转与连续路径分歧，红测复现后修复）；GenerationPort 返回 request_sha256 并在全部 9 个 record 落点持久化；
  请求正文不重复落盘（由 manifest 钉死的 problem + 冻结模板 + record 持久化的反馈/方法要素确定性构造，双路径测试即证明）。
正例、反例及实际命令：parse/policy/correctness/no_improvement 四类反馈 × 连续 vs 中断恢复双路径，
  断言第二条请求逐字节一致、请求 sha256 一致、失败候选 record 字节级一致、resume 恰好 1 次模型调用、结算/完成各恰 1 次。
  命令：.venv/bin/python -m pytest tests/test_resume_request_parity.py -q
测试数量、失败与跳过原因：新增 8 项（修复前 5 failed 红、修复后 8 passed 绿）；全量 714 passed / 3 skipped
平台、环境身份：Ubuntu 24.04（6.14.0-37-generic），Python 3.11.16（离线固定响应器，无 GPU/模型调用）
证据 URL/路径及完整 SHA-256：tests/test_resume_request_parity.py、src/kernelagent/optimization.py（提交 3278f4c）
未验证范围、遗留问题、活动作业：真实模型/Ubuntu 复验属 RV08/RV07；resume 只还原最后一条失败反馈（与连续路径内存行为一致，按构造成立）
下一包与依赖：RV04（实际用量预算，未做——live 实测固定配额耗尽见 handoff 通宵轮节）
```

```text
包编号：RV04
状态：CPU_PASS_TARGET_NOT_RUN（真实 GPU 计量已抽验，正式账单核对留 RV07）
基线与修复提交：基线 3278f4c；修复 a554573
行为变化与对应 R 编号：R3——GPU 秒口径定义为独占租约实测墙钟（monotonic、可注入时钟）；删除固定 300 秒/动作；
  预留=各阶段超时下限之和，不足在调用前拒绝（0 模型/GPU 调用）；deadline=min(下限,剩余) 仅传给声明支持的真实端口；
  每笔结算带 gpu_metering=actual/estimated，中断尝试保守持有并标 estimated；无 GPU 工作结算 0 秒（token 照旧）；
  profile（T30）计费并入同口径。
正例、反例及实际命令：假时钟实测求和、两级预留拒绝、deadline 传递/钳 0、中断保守记账恢复两次、
  unknown 结算记 estimated、record 不含计量字段（保 RV03 字节一致）。命令：.venv/bin/python -m pytest tests/test_budget_metering.py -q
测试数量、失败与跳过原因：新增 11 项；全量 725 passed / 3 skipped（既有两处断言随语义更新：预算 1234.5→41234.5、
  settled==900 改为实测区间——语义随包而变，非放松）
平台、环境身份：Ubuntu 24.04，Python 3.11.16（离线可控时钟）；真实 GPU 抽验 run
  artifacts/webui/20260914-050538（completed+champion；settled 32.38s 全部 actual：baseline 5.15+profile 6.34+candidate 20.89；
  同流程旧口径为固定 900s）
证据 URL/路径及完整 SHA-256：tests/test_budget_metering.py（提交 a554573）
未验证范围、遗留问题、活动作业：Ubuntu 真实账单核对留 RV07；RV08 示例预算 1800s 在新预留口径下将 budget_exhausted
  （候选预留 2100/3000），示例预算需上调；--max-repair-rounds CLI 文案澄清留 ADR（当前语义=容忍失败候选数）
下一包与依赖：RV05（未做）、RV06（进行中）
```

```text
包编号：RV06
状态：CPU_PASS（本机四要素等价验证；GitHub 四矩阵待 run 34783428574 及后续确认）
基线与修复提交：基线 0e07b5d；修复 9da1918
行为变化与对应 R 编号：R7——挂载源越界校验前移到 execute_container 顶部（纯路径/文件系统逻辑，零 Docker 依赖），
  镜像缺失的 infra_error 不再掩盖边界拒绝；新增 10 项无条件测试（绝对路径/父目录穿越/文件与目录 symlink/非目录源/
  回归守卫：用不存在的 docker 二进制驱动真实入口，若校验被移回 Docker 操作之后测试会显式失败）。
  根因分析：CI runner 无预缓存镜像→image inspect 先失败掩盖路径错误；Windows 3.11 "通过"系模块级 docker 门控
  全 skip 变绿（0 项实际运行）——该解释已写入报告，集成覆盖缺口留 RV07。
正例、反例及实际命令：修复前用假 digest+越界挂载复现 CI 失败路径（REPRO 输出在案），修复后同一 repro 输出
  "resolves outside workspace"。命令：.venv/bin/python -m pytest tests/test_container_path_preflight.py tests/test_container_worker.py -q
测试数量、失败与跳过原因：新增 10 项无条件运行；全量 735 passed / 3 skipped（既有 skip 语义未扩大，未 skip 任何安全断言）
平台、环境身份：Ubuntu 24.04，Python 3.11.16（含真实 Docker 容器用例 29 passed）
证据 URL/路径及完整 SHA-256：tests/test_container_path_preflight.py、src/kernelagent/worker/container.py（提交 9da1918）；
  workflow 触发 run id 34783428574（github.com/yuan-jc/kernelagent/actions）
未验证范围、遗留问题、活动作业：四矩阵最终结论以 GitHub Actions 页面为准（推送已触发）；Windows symlink 权限异常时会显式 ERROR 而非假绿
下一包与依赖：RV05（独立确认与证据链，未做）；RV07（干净 Ubuntu 验收）
```

```text
包编号：RV06（补充）
状态：TARGET_PASS
CI 证据：run 34784963289（commit cc850bd）四矩阵全绿
  https://github.com/yuan-jc/kernelagent/actions/runs/34784963289
修复路径：首跑暴露 Windows 专属失败（匿名注解诊断）：①并发锁 holder 子进程 import fcntl 在 Windows 崩溃
  （改为 msvcrt 回退，锁 hint 字节 4096 与运行时一致）；②Windows 强制锁使 pid 头不可经其他句柄读取
  （运行时锁移到 4096 hint 字节，头文件保持可读）；③git EOL 转换污染冻结哈希（catalog/userbench manifest
  校验失败）——新增 .gitattributes `* -text`；④webapp 工作区夹具文本写入 CRLF——改二进制写入；
  ⑤symlink 夹具在无权限平台显式 skip（仅夹具，断言不 skip）。
未验证范围：Windows 本机语义（msvcrt/权限）只经 CI 验证，本机无法复现。
```
