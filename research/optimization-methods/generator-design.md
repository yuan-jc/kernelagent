# 优化方法生成器设计（generator-design）

状态：调研设计稿（A2，2026-09-14 凌晨）。本文档只描述设计，不改动 src/、tests/、docs/。
术语遵循 `docs/NVIDIA算子优化Agent设计文档.md`（§6 方法生成器、§10 证据驱动诊断）；
知识目录见同目录 `methods.yaml`（38 个方法条目）。

## 0. 定位与 MVP 边界

生成器是设计文档 §6.2 谱系中 `RuleGenerator`（证据 → 有限方法）与
`LLMHypothesisGenerator`（方法计划 → 代码候选 prompt）之间的**计划层**：
它不直接生成 kernel 代码，而是根据证据产出"本轮该试哪些方法、每个方法的
prompt 片段与参数空间建议"，再交给现有 T16 生成路径产出候选。

MVP 最小闭环（建议早晨可实现的范围）：

```
evidence(现有字段) ──► BottleneckClassifier(纯函数) ──► 方法匹配/排序(查 methods.yaml)
      ▲                                                        │
      │                                                        ▼
   上一轮 feedback ◄────────────────────── MethodPlan(每方法: prompt 片段 + 参数空间)
      │                                                        │
      └──────────────── T16 generate(seed_note=计划片段) ◄──────┘
```

MVP 只要求三类瓶颈可分类、约 10 个高频方法可匹配（见 §6 评估集），
不要求 NCU 深度指标、不要求方法记忆持久化完整落地。

## 1. 输入：生成器消费的证据字段

生成器是证据的消费者。所有输入字段按现状取自现有代码，**缺失指标语义为
MISSING 而非 0**（`adapters/profiling/ncu.py` 的既有约定，design §10.2）。

### 1.1 证据来源 A：NCU profile（E3 层，可选输入）

来源：`adapters/profiling/ncu.py` 的 `LaunchMetrics` 列表与 `evidence_view()`。
注意 ADR-0003 边界：诊断容器只 profile 父方固定二进制，**候选不被 profile**。
因此 NCU 证据主要刻画 baseline/reference 实现，用于首轮瓶颈分类。

当前 `MetricCatalog` 默认字段（生成器 Tier-A 信号只允许引用这些名字）：

| 字段 | 含义 | 生成器用途 |
|---|---|---|
| `gpu__time_duration.sum` | kernel 时长（us） | 短 kernel / launch-bound 判定 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 计算内存吞吐 % | 内存流水饱和度 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 吞吐 % | 带宽受限判定 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐 % | 计算受限判定 |
| `launch__grid_size` / `launch__block_size` | 网格/块规模 | 欠并行 / launch-bound |
| `launch__registers_per_thread` / `launch__occupancy_limit_registers` | 寄存器与占用限制 | register pressure 信号 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际占用率 | 欠并行确认 |

硬规则：Tier-A 信号条件引用的任一指标为 MISSING 时，该信号**不成立也不反证**，
分类器输出 `evidence_coverage` 降级标记（见 §2.3）。

### 1.2 证据来源 B：任务与负载描述（E0 层，恒可用）

- `methods.py::TaskProfile`：`task_id, level, problem_id, problem_name, category`。
  `category` 是现有注册表匹配键（如 "layernorm"、"matmul"、"conv2d_biasadd"）。
- 从 `problem_source`（reference 实现）静态提取的**代码特征**（新增提取器，
  见 §4.2）：是否有 `tl.dot`、kernel 数目、forward 内 host 分配、三角掩码、
  autotune 有无、dtype、shape（从输入构造推断）。
- `Workload` 级 shape/stride/dtype（design §4.2 的对象；当前主循环用 dict，
  MVP 可继续以 dict 承载，字段名固定即可）。

### 1.3 证据来源 C：迭代历史（E1/E2 层，反馈回路）

来源：T16 主循环 `optimization.py` 的 `_feedback_note(stage, detail)` 五个阶段：
`generation / evaluate / correctness_pro / timing / confirm`，以及每候选的
`batches_ms`、champion 决策记录。REVIEW.md R5 已指出 resume 会丢失该反馈；
生成器设计把"反馈必须持久化并按 request 身份可重放"作为前置依赖（§4.4）。

### 1.4 输入对象草案（新增，纯 data，无 torch/triton 依赖）

```python
@dataclass(frozen=True)
class GeneratorInput:
    task: TaskProfile                    # methods.py 现有对象
    code_features: CodeFeatures          # 从 problem_source/parent_source 静态提取
    ncu_view: Mapping[str, Mapping] | None   # evidence_view() 输出；None=未 profile
    timing_summary: TimingSummary | None # baseline/候选 batches_ms 的聚合(仅描述性)
    attempts: tuple[AttemptRecord, ...]  # (method_id|None, stage, failure_class, note)
    hardware: HardwareFacts              # SM 数、smem/SM、cc(8.9)、带宽标定值(可 MISSING)
```

`HardwareFacts` 是 design §5.2 `HardwareTarget` 的最小投影；cc>=9.0 守卫
（TMA/warp-spec 条目）由它驱动，不靠 GPU 型号字符串（design §5.2 明示）。

## 2. 流程第一步：瓶颈分类

### 2.1 四个瓶颈类（MVP）+ 两个扩展类（预留）

| 瓶颈类 | 判定条件（按优先级短路） | 典型方法族（methods.yaml id 前缀） |
|---|---|---|
| `numerical` | correctness_pro/timing 阶段失败含 inf/NaN/越界/间歇性错 | numerics.* |
| `launch_bound` | duration 极短（us 级）或多 launch 序列间隙（NSYS/launch 数） | launch.*、memory.fuse_elementwise_chain |
| `parallel_deficit` | grid_size < ~2×SM 数 或 warps_active 低且双吞吐都低 | parallel.*、reduction.split_large_axis、pipeline.persistent_kernel |
| `memory_bound` | dram__throughput 高（≥60%）或观测字节 >> 必要字节 | memory.*、algorithm.*、reduction.online_streaming |
| `compute_bound`（预留） | sm__throughput 高且达精度匹配峰值的显著比例 | compute.* |
| `latency_bound`（预留） | 双吞吐中等（40-70%）失速特征 | pipeline.* |

### 2.2 无 NCU 时的分类（MVP 主路径）

候选不被 profile 是常态，因此 MVP 的分类主要靠 Tier-B/C：

1. `numerical`：历史里 correctness 失败 → 直接进数值方法族（这是硬规则，
   正确性优先于性能，design §6.4"优先修复正确性"）。
2. 其余情况按 `operator_class + code_features` 走**先验表**：
   例如 elementwise+多 kernel → fusion/launch；matmul+无 tl.dot → tensor_core；
   matmul+tile 数 < SM 数 → splitk/persistent；softmax/attention → streaming/stable。
3. 无任何信号（新题第一轮）→ 输出"基线方法计划"：
   launch.autotune_key_pruning + tuning.autotune_space_design + memory.coalesce
   （低风险默认三件套），并明确标注 `classification=uncertain`。

### 2.3 诚实输出：分类置信与证据覆盖

分类器输出附带：

```python
@dataclass(frozen=True)
class BottleneckClass:
    label: str                      # 上表之一
    signals_fired: tuple[str, ...]  # 触发的信号（含字段名与阈值）
    signals_missing: tuple[str, ...]# 因 MISSING 而未评估的信号
    coverage: str                   # ncu_full | ncu_partial | static_only | history_only
```

`coverage != ncu_full` 时，排序器降低 Tier-A 依赖方法的收益先验而不是隐藏不确定性；
`signals_missing` 原文进入候选生成请求的保留字段，防止下游把"没测到"当"正常"。

## 3. 流程第二步：方法匹配与排序

### 3.1 匹配（三道过滤）

对 methods.yaml 每条目依次检查（全部拒绝都要留显式理由，沿用
`MethodRegistry.select` 的 refusal 风格）：

1. **硬能力守卫**：`risks_preconditions`/信号中的 cc/dtype/轨道约束
   （如 pipeline.tma_tensor_descriptors 在 cc=8.9 恒拒绝，refusal 理由记录
   `device cc 8.9 < required 9.0`）。
2. **类别匹配**：`applicability_signals` 与 §1 输入比对；一条信号都没有 → 不匹配。
3. **重复/反例过滤**：`attempts` 中该方法在同题已失败（compile/correctness 类失败）
   → 拒绝或降级；同题已确认成功 → 不重复提议（改为提议其参数邻域）。

### 3.2 排序（design §6.4 的启发式，不是统计概率）

```
score(m) = prior_gain(m) * success_p(m) / est_cost(m)  +  epsilon * exploration(m)
```

- `prior_gain/success_p`：来自 methods.yaml `expected_gain`（low/medium/high →
  1/2/3）与失败历史修正（同题失败一次 ×0.3）。
- `est_cost`：`MethodProposal.estimated_gpu_seconds` 的先验（autotune 类按
  config 数 × 单次试跑估计；结构变换按 1 次生成+评测估计）。
- `exploration`：当前父候选集合中未覆盖的方法族 +1（保持族多样性，
  对应 §6.4 的"质量与方法族有差异"beam 约束）。
- 纪律约束（硬规则，非评分）：每轮计划内各方法互不叠加同一主要因素
  （meta.one_factor_per_round）；数值类方法优先于性能类（正确性优先）。

输出 top-K（建议初值 K=2-4，与 design §6.4 每轮 2-4 个实验一致）。

### 3.3 输出：MethodPlan

```python
@dataclass(frozen=True)
class MethodPlanItem:
    method_id: str                    # methods.yaml id
    hypothesis: Hypothesis            # domain/method.py 现有对象（statement/predicted_observations/falsification_conditions）
    prompt_fragment: str              # 注入生成请求的指令片段（见 §3.4）
    parameter_space: tuple[dict, ...] # autotune config 先验（gemm/elementwise/reduction 模板）
    estimated_gpu_seconds: float
    target_signals: tuple[str, ...]   # 预期验证该机制的观察（进入候选记录，供 E3 对照）
```

`MethodPlanItem` 是 `domain/method.py::MethodProposal` 的应用层投影：
MVP 直接构造 `MethodProposal`（method_version 取 methods.yaml meta.version，
`parameter_space_sha256` 对 parameter_space 的规范序列化取 hash），
使计划天然进入现有证据体系（E0 可追溯）。

### 3.4 prompt 片段模板（每方法一条）

片段由方法条目的 `triton_howto` + `risks_preconditions` 渲染，模式固定：

```
TARGET CHANGE (apply ONLY this change relative to the parent source):
- <mechanism 一句话>
REQUIREMENTS:
- <triton_howto 要点, 每条一行>
GUARDS (mandatory):
- all tl.load/tl.store masked (numerics.boundary_masking)
- fp32 accumulators for fp16/bf16 reductions (numerics.fp32_accumulators)
- do not measure or report performance yourself (tuning.do_bench_internal_protocol)
AUTOTUNE SPACE (budget-bounded):
- <parameter_space 渲染成 @triton.autotune config 列表>
```

注入点：现有 `adapters/models/generation.py::build_request` 的 `seed_note`
参数已支持附加文本（T16 现状：seed_note=失败反馈）。扩展方式是
`seed_note := render_feedback(attempts) + render_plan(top_items)`——
拼装顺序与内容纳入 `request_sha256` 身份，天然满足 R5 对"恢复后请求内容
与未中断路径一致"的要求（前提：attempts 持久化，§4.4）。

## 4. 与现有代码的对接点（新增 vs 扩展）

| 组件 | 现状 | 动作 |
|---|---|---|
| `src/kernelagent/methods.py` | 注册表：category 精确匹配、license/deps 门、template/reuse 工厂 | **扩展**：`MethodRegistry.select` 之外新增 `select_by_signals(task, generator_input)`（不动现有三道门；信号匹配是新增入口）。`Method.category` 语义不变 |
| `research/optimization-methods/methods.yaml` | 本次新增数据 | **新增**：作为目录数据源；加载器（新模块）解析后供注册表工厂生成 LLM 方法。加载失败→ 显式错误，不允许空目录静默通过 |
| `adapters/models/generation.py::build_request` | PROMPT_TEMPLATE + seed_note | **扩展**：seed_note 由 §3.4 渲染器拼装（渲染器是新增纯函数）；prompt 模板文本变更属于生成协议变更，须记录版本 |
| `optimization.py`（T16 主循环） | feedback 闭包字符串；generate→evaluate→correctness_pro→timing→confirm | **扩展**：在 generate 调用前插入 `plan = planner.plan(input)`；feedback 字符串升级为结构化 AttemptRecord（保留 `_feedback_note` 文本作为 note 字段） |
| `adapters/profiling/ncu.py` | MetricCatalog/LaunchMetrics/evidence_view | **只读复用**：分类器消费 `evidence_view()` 输出。不改 MISSING 语义 |
| promotion/confirm 路径 | 独立确认、champion 记录（R4 指出的缺口） | **只读复用 + 依赖**：生成器把晋升/失败决策作为 attempts 来源；不修改晋升规则（清单、容差、协议不可为过测改变） |
| 新模块（建议路径）`src/kernelagent/generators/` | 不存在 | **新增**：`catalog.py`（yaml 加载+校验）、`features.py`（problem_source 静态特征提取）、`classify.py`（瓶颈分类）、`plan.py`(匹配/排序/MethodPlanItem)、`prompt_fragments.py`（§3.4 渲染）。全部为纯 domain 友好代码：不 import torch/triton；triton 具体参数只作为数据/prompt 文本存在 |
| 方法记忆（design §10.4） | 无 | **新增（可后置）**：按"算子族+硬件能力+shape/dtype+父实现"检索的记录文件；MVP 用 attempts 持久化文件代替 |

信任边界不变式：生成器（含 LLM 假设）只能提出计划与候选；正确性、计时、
晋升一律由 trusted evaluator/promotion 路径决定（AGENTS 不变约束、design §9.2）。
methods.yaml 的 expected_gain 仅是排序先验，任何"该方法通常快 2x"的叙述
不得进入候选代码注释或报告正文。

## 5. 数据流示例（walkthrough）

题目：L2 融合题（category="layernorm_gelu"，fp32，单 kernel baseline，NCU 有 basic set）。

1. features：problem_source 无 tl.dot、单 kernel、forward 有两个逐元素/归一化调用。
2. classify：dram__throughput=78%、grid=4096 → `memory_bound`（signals_fired 记录两字段）。
3. match：memory.fuse_elementwise_chain（命中）、reduction.online_streaming（命中）、
   pipeline.tma_tensor_descriptors（拒绝：cc 8.9 < 9.0）、compute.*（sm__throughput=30% 不命中）。
4. rank：fusion 得分最高；top-K=[fusion, online_streaming]（两轮分别应用，单因素）。
5. plan：fusion 的 prompt_fragment=合入同一 kernel+边界 mask 守卫；online_streaming 的
   fragment=running max/sum 模板。parameter_space=[]（非 autotune 类）。
6. T16 generate(seed_note=fragment1) → 候选 → evaluator 裁决；失败原因进 attempts，
   下一轮 seed_note=render_feedback + fragment2。

## 6. 评估设计：生成器本身如何被验证

### 6.1 离线回归集（无 GPU 也能跑的部分）

固定"输入 → 期望"的表驱动用例（新增测试，只读现有 src）：

| 用例输入（构造） | 期望 |
|---|---|
| elementwise、多 kernel、duration 短 | 分类=launch_bound，计划含 memory.fuse_elementwise_chain |
| matmul、无 tl.dot | 计划含 compute.tensor_core_dot |
| matmul、grid_size=16、SM=24 | 计划含 parallel.grid_expansion_splitk 或 persistent；不含 tma（cc 守卫拒绝且 refusal 有理由） |
| softmax、历史含 NaN | 分类=numerical，numerics.stable_softmax_logsumexp 排第一 |
| 全 MISSING 的 ncu_view | coverage=static_only，不产生任何依赖具体指标的 signals_fired |
| attempts 含方法 m 失败 | m 被过滤或降级，且拒绝理由可读 |
| seed_note 注入前后 | build_request 的 request_sha256 变化且可复现（同输入两次构造 hash 相同） |

正反例都要有：守卫拒绝（tma/warp_spec on SM89）必须出现且理由正确；
"0 个方法匹配"的输入必须产出显式空计划原因（uncertain/全部被过滤），
而不是静默空列表冒充通过。

### 6.2 离线回归指标

- 分类命中率（上表用例全对 = 100% 为验收线）；
- 守卫召回：所有应被硬件/依赖守卫拒绝的方法 100% 拒绝；
- 诚实性检查：任何计划项的 target_signals 不得引用其输入中 MISSING 的字段；
- 确定性：同输入两次 plan 的 method_id 序列与 prompt_fragment 完全一致
  （生成器无隐藏状态；LLM 调用不在生成器内部）。

### 6.3 在线（GPU）评估——标记 NOT_RUN 直到真实执行

冻结 KernelBench L1 代表子集（design §4.3 的既有清单），同协议 A/B：
`T16 现状路径` vs `T16 + 生成器计划`，各固定候选数/预算。指标：
首个正确候选所需轮数、time-to-best、champion speedup 分布、每题消耗 GPU 秒。
按 AGENTS 约束：无 GPU 验收时该部分状态只能是 READY_FOR_ACCEPTANCE 以下，
不得用离线表通过冒充生成器"有效"。

### 6.4 已知风险与开放问题

1. **NCU 证据稀薄**（最大不确定性）：候选不被 profile（ADR-0003），分类高度依赖
   baseline NCU + 静态特征；baseline 瓶颈 ≠ 候选瓶颈（融合后瓶颈会迁移，
   design §10.3 明示融合改变 F/B）。缓解：计划标注"该假设基于 baseline 证据，
   候选需重新分类"；中期依赖 timing 差值 + 每轮低成本 NCU（若信任域允许）。
2. R5 修复是前置依赖：attempts 不持久化则反馈回路在 resume 下不等价，
   生成器会放大该缺陷。建议把 attempts 持久化列为生成器落地的同一工作包。
3. 方法的 expected_gain 先验未经本机标定（来源多为 A100/H100 数字），
   排序权重应先保守（探索项权重调高），积累本机方法记忆后再收紧。
4. category 键与 operator_class 特征提取对 KernelBench 混合题（子图/模型级）
   覆盖有限——L3/L4 题目 MVP 明确不支持，计划输出 `unsupported_granularity`。
