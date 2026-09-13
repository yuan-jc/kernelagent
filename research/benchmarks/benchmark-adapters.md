# Benchmark 适配调研与接入方案（A3 调研，2026-09-14）

调研子任务：为 RTX 4060 Laptop 8GB（SM89/Ada）选定 **2 个新外部 benchmark + 1 个自建 benchmark** 的接入组合，
并给出接入方案与自建接口设计。本文档只承载调研与设计，不改任何 `src/`、`tests/`、`docs/`、`configs/`。

阅读前提：设计文档 §3.1/§4/§5.4/§8（Benchmark 契约与接口草案）、现有
`src/kernelagent/adapters/benchmarks/`（kernelbench.py 静态快照读取器、flashinfer_bench.py trace 适配器）、
`src/kernelagent/domain/`（OperatorSpec/IOPort、Workload、OptimizationTask、Implementation）、
`src/kernelagent/dispatch.py`（WorkloadSpec/WorkloadOutcome/加权聚合）、
`research/fetch_kernelbench_problems.py` + `research/snapshot_manifest.json` + `research/SOURCES.md`（快照获取机制）。

---

## 0. TL;DR：推荐组合

| 槽位 | 选定 | 一句话理由 |
|---|---|---|
| 外部 #1 | **MultiKernelBench**（wzzll123/MultiKernelBench，MIT，固定 commit 快照） | 与 KernelBench 完全同构的 `Model/get_inputs/get_init_inputs` 格式 + MIT 许可 + 上游显式支持 Ada 架构，接入成本最低、题源（约 300 题/15 类）与 L1 互补 |
| 外部 #2 | **GPU-MODE reference-kernels**（pmpp_v2 + linalg 子集，固定 commit 快照） | `task.yml` 把 tests/benchmarks 写成声明式 (shape, seed) 对，与本项目 `Workload` 模型一一对应；协议（per-set `eval.py`/`utils.py`）完整随仓库分发；题源（归约/排序/前缀和/分解类）与 KernelBench 系完全互补 |
| 自建 | **user-bench**（用户自定义算子题） | 用户给任意 torch callable（forward 函数）+ 输入生成配置即可产出可评测 problem；domain 层协议零改动，torch 只进 worker 驱动 |

降级路径：外部 #2 若 license/时间受阻 → 用本地已有 KernelBench L2/L3/L4 快照扩清单（零下载，见 §2.1/§3.4）。

---

## 1. 调研方法

- 沿用仓库既有快照模式：固定 commit + 逐文件 SHA-256 清单 + 恢复脚本（模板：`research/fetch_kernelbench_problems.py`）。
  新 benchmark 一律"清单先行、评分在可信 evaluator"：adapter 只读字节与元数据，不执行题目代码（与现有
  `adapters/benchmarks/__init__.py` 声明的边界一致）。
- 检索日期 2026-09-14；下述 commit/文件数以当日 GitHub main 分支 API 为准，接入时以抓取时刻重新固化。
- 每个候选按统一维度评估：算子数量与类型 / license / 获取方式 / 评测协议（正确性判定 + 计时口径）/
  8GB 消费卡可跑性 / 接入工作量（对照现有 adapter 模式）。

---

## 2. 候选对比

### 2.1 对比总表

| 候选 | 算子数量与类型 | License | 获取 | 正确性协议 | 计时协议 | 8GB SM89 可跑 | 接入工作量 |
|---|---|---|---|---|---|---|---|
| **KernelBench L2/L3/L4**（ScalingIntelligence/KernelBench，同库扩清单） | L2=100 融合、L3=50 整模型、L4=20；本地快照已全量（270 文件，见 `research/sources/ScalingIntelligence__KernelBench/KernelBench/`） | MIT | **已本地化，零下载**；仅需扩 `dev-manifest` 开发子集 | 上游 `eval_kernel_against_ref`（已接入 T05 驱动） | 上游协议（已冻结） | L2 部分大题需排除（同 L1 的清单过滤机制）；L3 整模型 8GB 吃紧，需逐题预筛 | **极小**：只改清单不改代码（实现 agent 需动 `configs/kernelbench/dev-manifest.json`，本包未动） |
| **MultiKernelBench**（wzzll123/MultiKernelBench） | main 分支 `reference/*/*.py` 共约 300 个 NVIDIA 平台任务（activation 15、arch 50、attention 15、broadcast 10、convolution 34、fuse 100、index 12、loss 7、math 6、matmul 17、normalization 8、optimizer 5、pooling 6、reduce 5、resize 10；论文 arXiv:2507.17773 报 285 任务/14 类 + 后加 15 attention）；全为 KernelBench 风格单题 | **MIT**（LICENSE © 2025 南京大学作者） | git 快照（仅 `reference/` + `utils/` + `config.py` 必需），逐文件 sha256 | `utils/correctness.py`：`torch.allclose(atol=1e-4, rtol=1e-4)`（bool 输出要求完全相等），`num_correct_trials=5`，`seed_num=1024`，形状不匹配即 FAIL；另有 `utils/cheating_detection.py` 反作弊 | `utils/performance.py`：CUDA Event、warmup 3、perf trials 100、逐次 elapsed ms；**不清 L2 cache**（与本项目 T07 协议不同，分数不可互比） | 良好：`config.py` 的 `arch_list=['Ada']` 显式覆盖 SM89；个别题输入超 8GB（如 softmax 4096×393216 ≈ 6GB fp32），需清单期排除 + 运行期 `resource_exceeded` 兜底 | **小**：快照脚本 + 静态读取器可大量复制 `kernelbench.py` 模式；执行驱动可复用 KernelBench eval 路径或钉住上游 3 个 util 文件 |
| **GPU-MODE reference-kernels**（gpu-mode/reference-kernels） | main 分支 42 题：pmpp_v2=8（vectoradd/vectorsum/prefixsum/sort/histogram/matmul/conv2d/grayscale）、linalg=4（cholesky/eigh/qr/qr_v2）、helion=5、bioml=1、nvidia=5（nvfp4 系列，Blackwell fp4，SM89 不可用）、amd* 三组（AMD 目标，排除）；对本机可用约 **17 题** | **June 9 Researcher Reciprocity License v1.0**（Open RAIL-S 衍生，"If you train on it, you let us generate"；见 §3.3 风险） | git 快照；注意题目引用 `../utils.py`、`../eval.py` 等集合级共享文件，快照须按文件依赖闭包抓取 | per-set `eval.py`：`tests` 列表（小形状 + 公开 seed）逐个跑 `check_implementation`（`utils.verbose_allclose`，rtol=1e-5/atol=1e-8，逐题可在 `reference.py` 覆写）；子进程隔离、`test_timeout` 守护 | per-set `eval.py`：`benchmarks` 列表先生成输入再计时，重复运行受 `max_repeats`/`max_time` 上限，结果为 ns 时长统计（mean/std/err/best/worst），运行间 `clear_l2_cache`；官方榜 GPU 列表为 B200/H100/A100/L4——**本地 4060 分数不得声称榜单可比** | pmpp_v2/linalg 最大 benchmark 输入在几十 MB–百 MB 级（vectorsum 5.2e7 元素 ≈ 200MB），8GB 可跑；nvfp4 组不可跑 | **中**：多文件打包（task.yml/reference.py/task.py + 集合级 utils/eval）+ YAML 解析 + 独立驱动 |
| **ParEval**（parallelcodefoundry/ParEval） | 420 题、7 种执行模型（含 CUDA）、12 类计算问题（scan/stencil/ 等）；偏 HPC 代码生成 | MIT | git 快照（默认分支 develop） | 测试驱动式 pass/fail（编译 + 运行单元测试），与 torch 无关 | 上游以 speedup 相对串行为主口径，非 CUDA-event 微基准 | CUDA 子集规模小可跑，但需 nvcc 编译链与 CPU 侧 harness，走现有 torch-centric worker 要额外胶水 | **中大**：执行模型不同（裸 CUDA + nvcc），协议需单独固 |
| **TorchBench operator_benchmark**（pytorch/benchmark 的 microbenchmarks 子集） | 官方算子微基准框架（add/conv/matmul/…）+ 形状配置；测的是 torch 算子本身 | BSD 类（pytorch/benchmark 仓库） | git 快照 | 依赖 torch 自身正确性（参考与被测同为 torch 调用），对"候选 kernel"场景判定力弱 | `torch.utils.benchmark` 口径 | 可跑 | **中**：作为"参考=torch 算子调用"的题源可用，但与 LLM-kernel 评测目标错位；价值可被 user-bench 完全覆盖 |
| **CUTLASS examples / profiler**（NVIDIA/cutlass） | 示例 + Profiler 覆盖 GEMM/卷积族，极深 | 仓库 LICENSE 为自定义/NVIDIA 组合（GitHub API 报 NOASSERTION），非纯 BSD | git 快照 + C++ 构建链 | Profiler 自带验证 | Profiler 自带（device 端计时，口径独立） | 大数示例可跑（8GB 内），但无 torch 参考实现 | **大**：C++ 模板实例化 + 无 torch callable 协议，与 domain `Implementation` 模型落差大；设计文档仅定位为 backend 适配来源，非题源 |
| **LeetGPU 题库**（leetgpu.com） | 平台 50+ 题（矩阵/内存/融合） | 平台私有；GitHub 上只有第三方题解集（HaoyangPing0324/LeetGPU 等），**无官方题目+协议仓库** | 无快照来源 | 平台内私有 | 平台内私有 | 未知 | **不可接**：没有可固化的题目清单与协议来源，违反"证据驱动 + 快照可校验"约束 |
| **gpu-mode/Triton-Puzzles**（附带调研） | 教学谜题，非性能题库 | 见仓库 | git | 谜题式 | 无 | 可跑 | 不适用（无计时协议，教育向） |
| thunlp/TritonBench、meta tritonbench、FlashInfer-Bench、nvbench | 已接入或已在设计文档 §3.1 排期（T18/T19）；nvbench 是计时 harness 而非题库（设计文档已注明） | — | — | — | — | — | 不重复调研 |

### 2.2 关键结论

1. **格式同构性决定成本**：MultiKernelBench 的题目就是 KernelBench 格式（模块级 `Model`/`get_inputs`/`get_init_inputs`），
   现有 KernelBench 快照读取器、dev-manifest 绑定、eval 驱动模式几乎可整体平移——接入成本三个外部候选里最低。
2. **声明式 workload 只有 reference-kernels**：其 `task.yml` 的 `tests:`/`benchmarks:` 是显式 (size, seed) 列表，
   可直接映射 `domain.Workload`（无需执行题目代码即可枚举），是三个外部候选里唯一与 §4.2 "Workload=具体输入配置"
   模型严格对应的题源；也是唯一自带 DPS（目的张量作为输入元组一员）样例的题源，对 worker 输入物化是有价值的协议参考。
3. **协议完整性**：两者协议文件（MKB 的 `utils/correctness.py`、`utils/performance.py`、`config.py`；
   rk 的 per-set `eval.py`、`utils.py`）都随仓库分发、可 sha256 钉死，满足"新 benchmark 自带明确协议、不与现有协议混用"。
4. **8GB 是清单问题不是协议问题**：都采取"清单期冻结排除名单 + 运行期 OOM 记为 `resource_exceeded`（合法结果）"，
   不因过测改协议。

---

## 3. 推荐组合：理由、工作量、风险

### 3.1 外部 #1 首选：MultiKernelBench

- **理由**：MIT 许可无障碍；KernelBench 同构格式让快照读取器/驱动/dev-manifest 三件套模式整体复用；
  约 300 题、15 类（尤其 fuse 100 题）显著扩充题源多样性；上游 `arch_list=['Ada']` 显式覆盖本机；
  类别粒度（activation/reduce/loss 等单算子）比 KernelBench L1 更细，适合方法生成器的分域评估。
- **工作量估计**（给实现 agent，人工日折算 ≈1.5–2 天；夜间包可先交子集）：
  1. 快照脚本 `research/fetch_multikernelbench_problems.py`（模板照抄 fetch_kernelbench_problems.py）+ 清单 0.5d；
  2. 静态读取器 `adapters/benchmarks/multikernelbench.py`（复制 kernelbench.py 的 Manifest/校验骨架，正则改为
     `reference/<category>/<op>.py`）0.5d；
  3. 开发子集 dev-manifest（首批建议每类 2 题 ≈ 25–30 题，排除已知 >8GB 输入题）+ worker 驱动复用 KernelBench 路径
     （MKB 参考文件可按 KernelBench problem 源码字符串喂给同一驱动；ModelNew 视为 candidate 源）0.5–1d。
- **风险**：
  - 仓库年轻（~70 star）、main 活跃，题目可能增删 → 一次性钉 commit + 逐文件 sha256，之后不追新；
  - 部分题输入超 8GB（如 `reference/activation/softmax.py` 固定 4096×393216）→ 清单排除 + 运行期 resource_exceeded；
  - 其计时不清 L2 cache、逐次 event 计时 → **只能在"mkb_upstream"轨道引用其原始数字**；本项目正式计时仍走 T07 冻结协议，
    两者分开报告，绝不平均；
  - 题源公开且可能已进 LLM 训练数据 → 按 §4.3 记录污染假设，正式对比用未参与调参的扩展 workload。

### 3.2 外部 #2 首选：GPU-MODE reference-kernels（pmpp_v2 + linalg 子集）

- **理由**：声明式 workload 与 domain 模型一一对应；题源（归约、前缀和、排序、直方图、灰度、Cholesky/QR/eigh）
  与 KernelBench/MKB 的 nn.Module 风格互补；协议文件随仓库完整分发、可钉死；入口是普通函数
  `custom_kernel(data)`（非 nn.Module），恰好锻炼 Implementation/entry_point 抽象的普适性。
- **工作量估计**（≈2–2.5 人工日；夜间包可先交 pmpp_v2 8 题）：
  1. 快照脚本（按题目目录 + 集合级共享文件 `problems/<set>/{eval.py,utils.py,template.py}` + `problems/<set>.yaml`
     的依赖闭包抓取）0.5–1d；
  2. 读取器 `adapters/benchmarks/reference_kernels.py`：解析 task.yml（需引入 yaml 依赖或手写受限解析）、
     task.py 的 TestSpec、reference.py 的 `generate_input`/`ref_kernel`/`check_implementation` 引用关系，
     产出 ProblemRef + Workload 列表（tests→正确性 workload，benchmarks→perf workload）0.5–1d；
  3. 驱动：优先原样钉住上游 per-set `eval.py`/`utils.py` 在 worker 内执行（submission.py=candidate 源），
     Popcorn 输出解析为 result.json；若上游 eval.py 与容器边界冲突（多进程池、secret seed 注入），允许做"逐条对齐"的
     最小驱动，但正确性判定必须调用钉死的上游 `check_implementation` 0.5–1d。
- **风险**：
  - **license**：Researcher Reciprocity（Open RAIL-S 衍生）。义务触发点是"用材料训练/改进 AI 系统"；本项目对题目的
    用途是本地评测（与官方榜单同用途），不训练、不再分发快照，判读为可用；但必须在 SOURCES/ADR 记录该许可与义务，
    且**禁止**把题目内容用于候选生成的 few-shot 检索库或微调数据（这正是 AGENTS.md "训练/检索资料与测试隔离"的既有要求）；
  - 官方 GPU 列表不含 4060/Ada 消费卡，benchmark 形状按 H100 调 → 本地结果一律标注"本地复刻协议，不与榜单互比"；
  - 题量少（可用约 17 题）→ 定位为协议互补/正确性压力面，不当主力分数来源；
  - 秘密 seed 机制（eval.py 的 Cantor 组合）在官方榜用；本地复刻固定 `--seed` 行为并记录，不声称防污染等效。

### 3.3 自建：user-bench（用户任意 torch 算子题）

- **理由**：用户明确要求"针对任意算子测评"；KernelBench/rk/MKB 都不能让用户零成本带来自己的算子。
  设计原则：domain 层零改动（全部复用 OperatorSpec/Workload/OptimizationTask/Implementation）、
  torch 只出现在 problem 的 `reference.py`（worker 内执行物）、协议独立成文（`userbench-timing-v1`），
  容差由题作者在 problem.yaml 声明并在评测前随 manifest 冻结——冻结后不可为过测修改。
- 详细接口设计见 §5；工作量 ≈1.5 人工日（schema+loader 0.5d、驱动模板 0.5d、示例题+验收 0.5d）。
- **风险**：用户参考实现本身可能是错的/非确定的 → 协议规定参考实现的输出即真值（用户责任），驱动记录参考输出的
  张量 hash；非确定性参考（dropout 等）必须声明 `nondeterministic: true` 并改用统计容差，否则判 inconclusive 而非错。

### 3.4 备选与降级

- **外部 #2 备选 A：KernelBench L2/L3/L4 扩清单**——零下载（本地快照已有全部 270 文件），只需扩 dev-manifest 子集；
  若 rk 的 license 审查不过或时间不够，这是零风险替代。代价：不新增"库"，算不上新 benchmark，且 L3 整模型题在 8GB 上
  预筛成本高。
- **外部 #2 备选 B：ParEval CUDA 子集**（MIT）——若后续要非 torch 的裸 CUDA 题；接入成本中大（nvcc 链路），
  不建议夜间包处理。
- **不推荐**：CUTLASS examples（无 torch 协议、构建重、license 待人工确认）；LeetGPU（无可固化题源）；
  TorchBench operator_benchmark（价值被 user-bench 覆盖）。

---

## 4. 外部 benchmark 接入方案

### 4.1 MultiKernelBench 接入

**快照获取脚本要点**（新文件建议 `research/fetch_multikernelbench_problems.py`，结构照抄
`research/fetch_kernelbench_problems.py`）：

- 输入 manifest（建议 `configs/multikernelbench/snapshot-files.manifest.json`，由实现 agent 生成）：
  `{"schema_version":"1.0","repository":"wzzll123/MultiKernelBench","commit":"<40-hex>","files":[{path,git_blob_sha1,sha256,size}...]}`；
- 抓取范围（文件依赖闭包，勿整仓库）：
  - `reference/<category>/<op>.py`（题目本体，首批按 dev 子集列）；
  - `utils/correctness.py`、`utils/performance.py`、`utils/utils.py`、`utils/evaluation_utils.py`、`config.py`
    （协议引用，同 KernelBench dev-manifest 的 `protocols.references` 做法，逐文件 sha256 钉死）；
- 下载 `https://raw.githubusercontent.com/<repo>/<commit>/<path>`，落盘前后各校验一次 sha256，临时文件 + `os.replace` 原子写，
  拒绝路径逃逸（照抄现有实现的全部防护）；
- SOURCES.md 增补一行（research/ 下允许新增文件；本包不改 docs/，交接时提醒实现 agent 补 `docs` 侧索引——若流程要求）。

**协议映射表（MKB → 本项目）**：

| MKB 概念 | 本项目 domain/adapter 概念 | 备注 |
|---|---|---|
| `reference/<category>/<op>.py` | `ProblemRef`（suite=multikernelbench，category+op 复合 id：`mkb-<category>-<op>`）；granularity 按题注 operator/subgraph | fuse/attention 类多为 subgraph |
| 模块级 `get_inputs()`/`get_init_inputs()` + 常量 | **代码定义的 workload**：静态 adapter 不物化 Workload；worker 驱动执行该函数生成具体输入后构造 `Workload`（每题 1 个 workload，`weight=1.0`） | 与 rk 的声明式 workload 相反，记录为已知差异 |
| `Model.forward` | 参考实现（torch eager），对应 KernelBench 驱动的 original_model | 上游 `execute_template` 同样以 Model 为真值 |
| `ModelNew.forward` | `Implementation(entry_point="ModelNew", backend=<cuda|triton>)` | candidate 源即提交代码字符串 |
| `utils/correctness.py`（allclose atol=rtol=1e-4、5 trials、seed 1024、bool 精确相等） | MKB 协议（独立轨道，如 `mkb_upstream`），容差**不得**与 KernelBench/rk/user-bench 混用 | 协议文件 sha256 入 dev-manifest |
| `utils/performance.py`（CUDA event、warmup 3、100 trials、无 L2 清理） | 同上，仅用于 MKB 轨道内自比；本项目正式计时数字仍由 T07 冻结协议产出，两者分开报告 | 记入协议描述字段 |

**驱动要点**：MKB 参考文件是完整合法的 KernelBench problem 源码（模块级 Model/get_inputs），
因此执行路径可直接复用 KernelBench eval 驱动（把 MKB 文件作为 `problem_path` 传入，candidate 作为 `custom_model_src`）；
正确性比较若要逐字对齐上游（1e-4/bool 规则/5 trials），优先把钉死的 `utils/correctness.py` 一并挂进 /task，
由驱动调用其 `execute_template`，避免重写规则引入漂移。

### 4.2 GPU-MODE reference-kernels 接入

**快照获取脚本要点**（建议 `research/fetch_reference_kernels.py`）：

- manifest：`repository:"gpu-mode/reference-kernels"`，commit 40-hex，逐文件 sha256；
- 抓取闭包以题为单位：`problems/<set>/<name>/{task.yml,task.py,reference.py}` + 集合级
  `problems/<set>/{eval.py,utils.py,template.py}` + 顶层 `problems/<set>.yaml`（题集元数据：deadline/gpus 列表，用于记录
  "官方未含本机 GPU"）；首批仅 `pmpp_v2` + `linalg`；
- 其余防护（路径校验、原子写、校验失败不改缓存）照抄现有脚本。

**协议映射表（rk → 本项目）**：

| rk 概念 | 本项目 domain/adapter 概念 | 备注 |
|---|---|---|
| `problems/<set>/<name>/` | `ProblemRef`：task_id=`rk-<set>-<name>`；granularity=operator | 集合级 yaml 记录进 suite 元数据 |
| `task.yml: tests[]`（如 `{"size":1023,"seed":4242}`） | `Workload`（role=correctness；shapes 从 TestSpec 字段推导，seed 直取） | **声明式**：静态读取器即可物化，无需执行代码 |
| `task.yml: benchmarks[]` | `Workload`（role=performance） | 计时只走 rk 轨道协议 |
| `task.yml: test_timeout/benchmark_timeout/ranked_timeout` | 驱动 case.json 的超时守卫 | 合法超时=timeout 结果 |
| `task.py: TestSpec` | `OperatorSpec.inputs` 的端口模式（字段名→port 名，dtype 由 `generate_input` 语义注释固定） | port shape 允许 -1（tests 大小不同） |
| `reference.py: generate_input(**spec)` | worker 输入物化策略（确定性：torch.Generator + seed） | 张量 hash 记录 |
| `reference.py: ref_kernel` | 参考实现（torch eager；vectorsum 用 float64 归约） | |
| `reference.py: check_implementation`（默认 `make_match_reference` → `utils.verbose_allclose` rtol=1e-5/atol=1e-8） | 正确性判定器：驱动**必须调用钉死的上游函数**，不得自写比较规则 | 逐题可覆写，覆写也随快照钉死 |
| `submission.py: custom_kernel(data)` | `Implementation(entry_point="custom_kernel")`；DPS：输出张量可作为输入元组一员传入 | 与 domain `Implementation` 无冲突 |
| `gpus: [B200,H100,A100,L4]` | 环境守卫记录：本机不在官方列表 → 报告标注"本地复刻" | 不冒充榜单分 |

**驱动要点**：上游 per-set `eval.py` 是完整 harness（子进程池 + Popcorn 输出 + secret seed 组合）。
优先整文件钉进 /task 原样跑、解析其输出键（`check`、`test.*.status`、benchmark 统计）；若多进程池与容器冲突，
降级为最小驱动但保持三点不变：判定函数来自钉死的 `utils.py`/`reference.py`、输入生成来自钉死的 `generate_input`、
计时口径（L2 清理 + max_repeats/max_time + ns 统计）逐项对齐并在协议描述里写明差异。

---

## 5. 自建 benchmark（user-bench）接口设计

### 5.1 目标与边界

- 用户给：一个 forward 函数（torch callable）+ 输入形状/dtype 配置（可多条 workload）→ 生成可评测 problem。
- domain 层（`src/kernelagent/domain/`）**零改动**：problem.yaml 在 adapter/loader 内映射为
  `OperatorSpec` + `tuple[Workload]` → `OptimizationTask`；`reference.py` 文本作为 `SourceFile` 进参考
  `Implementation(backend="torch_eager")`。
- torch 只出现在 problem 目录的 `reference.py` 与 worker 驱动中；loader（读 yaml、建 domain 对象、算 sha256）不得 import torch。
- 候选不自证：正确性判定与计时都由 worker 驱动（可信 evaluator 侧）执行；`reference.py` 是用户输入而非候选，
  其内容 sha256 随 problem 冻结，驱动同时记录参考输出张量 hash。
- 协议独立：user-bench 有自己的协议身份（`userbench-correctness-v1` + `userbench-timing-v1`），不得与
  KernelBench/rk/MKB 的容差或计时数字混排；现有 T07 冻结协议不被修改——user-bench 只是**另一个**明确协议。

### 5.2 目录布局（建议，由实现 agent 创建）

```
benchmarks/user/
  README.md                        # 如何贡献一题（用户视角文档）
  <suite_name>/                    # 一个 suite = 一批一次冻结的题
    suite.yaml                     # 协议声明（见 5.4）
    manifest.json                  # 冻结清单：suite.yaml 与每个 problem 文件的 sha256；评测前生成，之后漂移即 FAIL
    problems/<op_name>/
      problem.yaml
      reference.py
      examples/                    # 可选：example candidate 与期望输出记录，供冒烟验收
```

运行目录、结果与 hash 记录沿用设计文档 §11（runs/…），不在本题库目录内写结果。

### 5.3 problem 定义文件格式

`problem.yaml` 示例（RMSNorm，2 workloads，其中 1 条带 dispatch guard）：

```yaml
api_version: kernelagent/userbench/v1alpha1

problem:
  name: rmsnorm_fp32                 # 全局唯一；task_id = userbench-<suite>-<name>
  granularity: operator              # operator | subgraph | model（同 domain.Granularity）
  description: "RMSNorm without affine weights"
  reference:                         # torch callable 注入点（worker 内执行）
    source: reference.py             # 相对本 problem 目录；sha256 入 manifest
    entry: run                       # reference.py 里的符号名；签名 run(*inputs) -> outputs
    nondeterministic: false          # true 时正确性改走统计容差并标 inconclusive 语义
  inputs:                            # 映射 OperatorSpec.inputs（IOPort；-1 为动态维）
    - name: x
      dtype: float32
      shape: [-1, -1]
  outputs:
    - name: y
      dtype: float32
      shape: [-1, -1]
  tolerance:                         # 映射 userbench-correctness-v1；冻结后不得为过测修改
    rtol: 1.0e-5
    atol: 1.0e-8
    equal_nan: false
    policy: allclose                 # allclose | exact |（新增策略须改 suite.yaml 白名单 + 驱动实现，不许题内自定义代码）
  input_generation:                  # 驱动物化输入的策略（确定性：per-workload seed）
    distribution: randn              # randn | rand | uniform | zeros | ones | arange
    scale: 1.0
    offset: 0.0
  workloads:                         # 映射 tuple[Workload]，id 在题内唯一
    - id: w1_base
      shapes: [[1024, 4096]]
      dtypes: [float32]
      seed: 0
      weight: 1.0
    - id: w2_wide
      shapes: [[8192, 8192]]
      dtypes: [float32]
      seed: 1
      weight: 2.0
      guard_max_shape: [8192, 8192]  # 可选；映射 dispatch.WorkloadSpec.max_shape（逐输入张量逐维 cap）
```

配套 `reference.py`（用户视角：就是一个普通函数 + 说明注释）：

```python
# 仅在 worker 容器内执行；本文件内容 sha256 随 suite manifest 冻结。
import torch

def run(x: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1.0e-6)
```

候选接口（用户/agent 视角）：`candidate.py` 提供同名 entry 符号 `run(*inputs) -> outputs`，
映射 `Implementation(entry_point="run")`；允许多文件（SourceFile 列表）与 `build_spec`（toolchain 声明），
与现有 domain 完全一致。

`suite.yaml` 示例：

```yaml
api_version: kernelagent/userbench/v1alpha1
suite: rmsnorm_suite_v1
protocol:
  correctness: userbench-correctness-v1   # 判定在驱动内实现，输入容差来自各 problem.yaml
  timing: userbench-timing-v1             # CUDA event；warmup=10；iters=2000；独立 batch 为重采样单位；
                                          # 数字只与本协议内结果比，不与 T07/KernelBench/rk/MKB 混排
  default_tolerance_ceiling: {rtol: 1.0e-4, atol: 1.0e-5}   # 题内容差不得宽于此上限
allowed_dependencies: ["torch"]           # reference/candidate 可 import 的白名单
denominator: "全部冻结 workload；缺 workload 时分数未知，不算通过"
```

对应的 JSON Schema 要点（`schemas/` 由实现 agent 落盘时补）：
`api_version`(const)、`problem.name`(pattern `^[a-z0-9_]+$`)、`granularity`(enum)、
`inputs[]/outputs[]`（name 唯一、dtype 字符串、shape 整数含 -1）、`workloads[]`（id 唯一、
shapes 长度==len(inputs)、每维正整数、seed>=0、weight>0）、`tolerance`（rtol/atol 数值、policy enum、
且不超过 suite 上限）、`reference.source`（安全相对路径）。这些规则与
`domain/operator.py`、`domain/workload.py`、`domain/_validation.py` 的既有校验一一对应。

### 5.4 domain 映射与 dispatch 关系

| user-bench 概念 | domain 对象 | 说明 |
|---|---|---|
| `problem.name` | `OperatorSpec.operator_id = "userbench/<suite>/<name>"` | granularity 直取 |
| `inputs`/`outputs` | `tuple[IOPort]` | dtype 用字符串（与现有一致），-1 动态维 |
| `workloads[]` | `tuple[Workload]` | loader 校验 shapes/dtypes 长度与 OperatorSpec.inputs 一致（`OptimizationTask` 已强制） |
| `weight`/`guard_max_shape` | `dispatch.WorkloadSpec(weight, required, max_shape)` | 每 workload 一条 guard；缺测 workload → 分数未知（`select_implementation`/`weighted_aggregate` 既有语义，不改） |
| `reference.py` | `Implementation(backend="torch_eager", entry_point=entry)`（SourceFile 1 个） | 参考实现也是 Implementation，身份 hash 同 §11.1 |
| `candidate.py` | `Implementation(entry_point=entry)` | 候选不自证 |
| `tolerance` + suite protocol | 驱动 case.json 的一部分，随协议 hash 入 `EvaluationResult.protocol_sha256` | 冻结于 manifest |

### 5.5 worker 注入（驱动模板要点）

建议新驱动 `configs/userbench/userbench_driver.py`（实现 agent 落盘；结构对照 `configs/kernelbench/eval_driver.py`），
父服务把 5 个文件只读挂进 `/task`：

```
/task/problem.yaml            # 已通过 loader 校验的原始 yaml
/task/reference.py            # 用户参考 callable（sha256 已验）
/task/candidate.py            # 候选源（entry 同名）
/task/case.json               # 本 case 的 workload 参数 + 容差 + 计时协议参数 + entry 符号名
/task/userbench_driver.py     # 本驱动
```

驱动行为（伪代码）：

1. 读 `case.json`；按 `seed`+`distribution` 用 `torch.Generator` 确定性物化输入张量；记录输入张量 sha256（沿用
   `adapters/evals` 现有张量 hash 助手思路）；
2. import `reference.py`，跑参考得 ref 输出（记录输出张量 sha256）；`nondeterministic: true` 时跑 K 次取统计界；
3. clone 输入后 import `candidate.py` 跑候选；比较按 `tolerance.policy`（判定代码在驱动内，候选不可注入比较器）；
4. 正确性通过后按 `userbench-timing-v1` 计时（CUDA event，warmup/iters 来自协议参数，重采样单位为独立 batch）；
5. 写 `/out/result.json`：correctness verdict、timing raw batch、输入/输出 hash、协议版本与参数、
   状态机取值（passed/incorrect/build_failed/timeout/resource_exceeded/inconclusive）。

信任边界与现有约定相同：用户参考代码与候选代码同属"worker 内执行的不可信代码"，只进容器/进程执行器，
父服务不 import 它们（现有进程执行器不是沙箱——遵循 AGENTS.md 对 T04 边界的既有限制，不新增承诺）。

### 5.6 正/反验收例（供实现 agent 写验收命令）

- 正例 1（loader，无 torch）：加载随包示例 problem → 得到的 `OptimizationTask` 字段与 yaml 逐项相等；
  loader 模块 import 时不出现 torch（用 `sys.modules` 断言）。
- 正例 2（端到端，需 GPU）：示例题 + 一个平凡正确 candidate → result.json 为 passed 且 timing batch 数与协议一致。
- 反例 1：candidate 返回错误数值 → incorrect；反例 2：candidate entry 缺失 → build_failed；
  反例 3：reference.py 或 problem.yaml 被改动、与 manifest sha 不符 → 驱动/loader 拒绝执行（FAIL 而非静默通过）；
  反例 4：workload 缺测 → `weighted_aggregate`/`select_implementation` 给出未知/拒绝（既有行为）。
- 冻结纪律：suite manifest 生成后改容差/计时参数必须换协议版本号并重建 manifest；验收含"改动后校验失败"用例。

---

## 6. 实施顺序建议（给实现 agent 的分步指南）

阶段 0（无需 GPU）
1. 建 `configs/multikernelbench/`、`configs/reference-kernels/`、`configs/userbench/` 三个目录的 manifest 骨架
   （schema_version 沿用 "1.0"；协议引用单独成节，模式照抄 `configs/kernelbench/dev-manifest.json`）。
2. MKB：选定 commit → 生成 files manifest（sha256 由抓取时计算）→ dev 子集（首批：activation/matmul/reduce/loss/
   normalization 各 2–3 题 + fuse 5 题；排除输入估算 >6GB 的题，排除名单写进 dev manifest 注记）。
3. rk：选定 commit → 抓 pmpp_v2 全 8 题 + linalg 4 题 + 各集合级共享文件；manifest 记录闭包。
4. user-bench：写 suite.yaml + 2 个示例题（rmsnorm_fp32、gelu_bf16）+ manifest 生成脚本。

阶段 1（读取器与 loader，无需 GPU）
5. `adapters/benchmarks/multikernelbench.py`（复制 kernelbench.py 骨架；problem_members 正则改
   `reference/<cat>/<op>.py`；静态校验 + 只读接口）。
6. `adapters/benchmarks/reference_kernels.py`（受限 YAML 解析可用 PyYAML，若依赖策略不允许则手写
   task.yml 受限子集解析器——task.yml 字段面很小；产出 ProblemRef + 声明式 Workload 列表）。
7. `userbench` loader（yaml → domain 对象 + sha256 绑定校验），含 §5.6 正反例 1/3/4。

阶段 2（驱动与执行，需 GPU/容器边界）
8. MKB 驱动：复用 KernelBench eval 驱动路径 + 钉死的 `utils/correctness.py`；先跑通 1 题（如 leaky_relu）冒烟。
9. rk 驱动：钉死 per-set eval.py/utils.py 跑通 vectorsum（最简单 DPS 例）；再跑 prefixsum/sort。
10. userbench 驱动：§5.5 模板 + 示例题端到端（正例 2）。

阶段 3（收尾）
11. 全部子集跑一遍：记录每个题的状态（含 timeout/resource_exceeded/incorrect 的合法分布），不允许"全 skip"或 0 项清单；
12. SOURCES.md 增补 MKB/rk 行（commit、license、本地材料路径、许可义务注记）；若需 ADR（rk 的 license 判读、
    user-bench 的"容差由题作者声明"决策），交接给主 agent 判断是否立项；
13. 更新 task board 与 handoff：写明哪些检查 NOT_RUN、rk "本地复刻不与榜单互比"的声明位置。

步骤合计 ≈ 13 步；MKB 与 rk 相互独立可并行；user-bench 依赖最少，可最先落地端到端。

---

## 7. 风险汇总

| 风险 | 影响 | 缓解 |
|---|---|---|
| rk Researcher Reciprocity license 的解释空间 | 合规 | 仅本地评测用途、不训练不检索不再分发；SOURCES/ADR 记录义务；不过审则切备选 A（KernelBench L2/L3 扩清单） |
| MKB 部分题 >8GB、仓库年轻题目漂移 | 可跑性/可复现 | 清单期排除名单 + 运行期 resource_exceeded；一次性钉 commit+sha256 |
| rk 官方 GPU 列表不含 4060 | 分数可比性被误读 | 所有本地 rk 数字标注"本地复刻协议"；不进入与 KernelBench/T07 同表对比 |
| 三种新协议 + 既有 T07 并存，数字被混排 | 评测可信度 | 每个 result 强制携带协议身份与 hash；聚合层（既有代码）拒绝缺协议来源 |
| 用户参考实现错误/非确定 | user-bench 结论无效 | 参考输出 hash 记录 + `nondeterministic` 声明 + 统计容差路径；容差冻结上限由 suite.yaml 约束 |
| PyYAML 等新依赖 | 依赖策略 | task.yml 字段面小，可手写受限解析；problem.yaml 同理；依赖决策交给实现 agent 按仓库约束处理 |

## 8. 来源（检索日期 2026-09-14）

- MultiKernelBench: https://github.com/wzzll123/MultiKernelBench （LICENSE=MIT；`utils/correctness.py`、`utils/performance.py`、`config.py`、`dataset.py`、`backends/cuda_backend.py`、`reference/activation/softmax.py` 为直接核对文件）；论文 https://arxiv.org/abs/2507.17773
- GPU-MODE reference-kernels: https://github.com/gpu-mode/reference-kernels （LICENSE=June 9 Researcher Reciprocity v1.0；`problems/pmpp_v2/{eval.py,utils.py,template.py}`、`problems/pmpp_v2/vectorsum_py/{task.yml,task.py,reference.py}`、`problems/pmpp_v2.yaml` 为直接核对文件）
- ParEval: https://github.com/parallelcodefoundry/ParEval （LICENSE=MIT，默认分支 develop）；论文 https://arxiv.org/abs/2401.12554
- GPU MODE 榜单/生态： https://gpumode.com/ 、https://www.nvidia.com/en-us/on-demand/（NVIDIA blog: Topping the GPU MODE kernel leaderboard）
- LeetGPU: https://leetgpu.com/ ；第三方题解 https://github.com/HaoyangPing0324/LeetGPU 、https://github.com/AlphaGPU/leetgpu-challenges （均非官方题源）
- 上文本地快照核对：`research/sources/ScalingIntelligence__KernelBench/KernelBench/`（level1=100、level2=100、level3=50、level4=20 文件）
- 许可证元数据经 GitHub API `/repos/<org>/<repo>/license` 核对（2026-09-14）。
