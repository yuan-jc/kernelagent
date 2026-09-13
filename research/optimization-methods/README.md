# NVIDIA 算子优化方法调研（A2 通宵产物）

日期：2026-09-14（凌晨至早晨截止前）。范围：为「优化方法生成器」构建知识基础——
根据 profile 证据与算子类型自动选择/排序优化方法并生成候选策略。
配套文件：`methods.yaml`（38 个方法条目）、`generator-design.md`（生成器设计）。
仓库术语与信任边界遵循 `docs/NVIDIA算子优化Agent设计文档.md`；工具/仓库的
工程接入边界见该文档附录 A（S01–S21），本文不重复。

## 调研范围

1. 论文：LLM 生成/优化 kernel（KernelBench、Mirage、KEVIN、CUDA-L1、AutoTriton
   及 RL 同族）、编译器调参空间（Ansor）、IO-aware 算法（FlashAttention）、
   低比特计算（Atom）、综述与机制分析（Hopper microbenchmarking、Triton attention 解剖）。
2. 官方教程与文档：Triton tutorials（vector-add、fused-softmax、matmul、layer-norm、
   fused-attention、grouped-gemm、persistent-matmul、block-scaled-matmul、gluon warp-spec）、
   CUDA C++ Programming Guide（async copies）。
3. 工程 blog/讲座：NVIDIA Hopper architecture、PyTorch TMA/warp-specialization deep dive、
   Colfax TMA tutorial、GPU-MODE 讲座系列（profiling checklist、reductions、
   flash attention、practitioner's guide to Triton、CUTLASS）。
4. 仓库：Triton（autotuner/testing）、CUTLASS、FlashInfer、GPU-MODE lectures、KernelBench。

检索限制（诚实边界）：WebSearch 在本时段间歇限流（HTTP 429），arXiv 检索为主要
替代通道。以下条目按给定名称**未能确证**为算子优化方法论文，未列入目录引用：
Liquid、Metis（检索到的同名论文为 USENIX ATC'24 分布式训练方向）、Bird、Gamma、
Meta-Schedule（arXiv 条目未定位到）。另：任务清单中的 "Atom" 检索结果为
低比特量化推理论文（arXiv:2404.14289, MLSys'24），其方法价值在量化/精度条目中
体现；"MPIr" 名称未能解析到任何相关论文。若这些是内部别名，请持有者补充线索后再入目录。

## 来源清单（全部为本次实际访问或搜索结果确认的 URL）

### 论文
- KernelBench: Can LLMs Write GPU Kernels? — https://arxiv.org/abs/2502.10517 （ICML 2025 项目；执行/profiling 反馈提升迭代结果的证据来源）
- Mirage: A Multi-Level Superoptimizer for Tensor Programs — https://arxiv.org/abs/2405.05751 （OSDI'25；kernel/thread block/thread 三级 μGraph、persistent 抽象）
- Kevin: Multi-Turn RL for Generating CUDA Kernels — https://arxiv.org/abs/2507.11948
- CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning — https://arxiv.org/abs/2507.14111 （ICLR 2026）
- AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs — https://arxiv.org/abs/2507.05687
- Ansor: Generating High-Performance Tensor Programs for Deep Learning — https://arxiv.org/abs/2006.06762 （OSDI 2020；分层搜索空间/进化搜索）
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — https://arxiv.org/abs/2205.14135
- Atom: Low-bit Quantization for Efficient and Accurate LLM Serving — https://arxiv.org/abs/2404.14289 （MLSys'24）
- Towards Automated Kernel Generation in the Era of LLMs（综述）— https://arxiv.org/abs/2601.15727
- Dissecting the NVIDIA Hopper Architecture through Microbenchmarking — https://arxiv.org/abs/2501.12084
- The Anatomy of a Triton Attention Kernel — https://arxiv.org/abs/2511.11581
- Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation — https://arxiv.org/abs/2201.05587 （PACT 2022）
- 同族（搜索确认存在、本次未深读，仅备查）：CUDA-L2 https://arxiv.org/abs/2512.02551 ·
  ConCuR https://arxiv.org/abs/2510.07356 · CUDA-LLM https://arxiv.org/abs/2506.09092 ·
  StitchCUDA https://arxiv.org/abs/2603.02637 · daVinci-kernel https://arxiv.org/abs/2606.16497 ·
  DRTriton https://arxiv.org/abs/2603.21465 · TritonRL https://arxiv.org/abs/2510.17891 ·
  Profiling-Guided Framework for Automated Triton Kernel Optimization https://arxiv.org/abs/2512.09196

### 官方教程/文档
- Triton tutorials 索引 — https://triton-lang.org/main/getting-started/tutorials/index.html
  - 01 vector add — https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
  - 02 fused softmax — https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
  - 03 matrix multiplication（autotune config 表、GROUP_SIZE_M）— https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - 05 layer norm — https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
  - 06 fused attention — https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
  - 07 libdevice — https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html
  - 08 grouped GEMM — https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
  - 09 persistent matmul（TMA 描述符、CC>=9.0 门、消费卡 smem 注记）— https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - 10 block-scaled matmul — https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - gluon warp specialization — https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html
- CUDA C++ Programming Guide: asynchronous copies（cp.async/mbarrier/TMA）— https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html

### Blog / 讲座
- NVIDIA Hopper Architecture In-Depth（TMA、mbarrier、clusters）— https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- PyTorch blog: Deep Dive on the Hopper TMA Unit for FP8 GEMMs — https://pytorch.org/blog/hopper-tma-unit/
- PyTorch blog: Enabling Advanced GPU Features in PyTorch – Warp Specialization — https://pytorch.org/blog/warp-specialization/
- Colfax Research: Mastering the NVIDIA Tensor Memory Accelerator (TMA) — https://research.colfax-intl.com/tutorial-hopper-tma/
- GPU-MODE lectures（含 lecture_008 CUDA Performance Checklist、lecture_009 Reductions、
  lecture_012 Flash Attention、lecture_014 A Practitioner's Guide to Triton、
  lecture_015 CUTLASS）— https://github.com/gpu-mode/lectures

### 仓库（工程接入以设计文档附录 A 为准）
- triton-lang/triton（runtime/autotuner.py、testing.py）— https://github.com/triton-lang/triton
- NVIDIA/cutlass — https://github.com/NVIDIA/cutlass
- flashinfer-ai/flashinfer — https://github.com/flashinfer-ai/flashinfer
- ScalingIntelligence/KernelBench — https://github.com/ScalingIntelligence/KernelBench

## 核心结论摘要

1. **反馈回路是已验证的最高杠杆**：KernelBench 论文明确"执行与 profiling 反馈
   显著提升迭代式生成"；KEVIN/CUDA-L1 的 RL 本质是把该反馈做成奖励信号。
   对本项目：MVP 生成器首先应是"结构化失败反馈 + 方法计划注入 seed_note"，
   而不是更花哨的搜索策略（R5 指出的反馈丢失必须先修）。
2. **小而准的参数空间优于自由发挥**：Triton 官方 matmul/persistent 教程给出了
   成熟 config 表（stages 3-5、GROUP_SIZE_M=8、BLOCK 组合）；Ansor 的教训是
   分层空间+预算控制。生成器应按算子类提供 8-16 个 config 的模板空间，
   并用 prune/key 控制调参 GPU 预算（调参计入预算是不变约束）。
3. **硬件能力守卫是一等方法**：TMA 与 warp specialization 需要 CC>=9.0，
   当前 SM89（RTX 4060 Laptop）不适用；tutorials/09 自注其在小 shared memory
   消费卡上会失败。方法目录必须携带可执行的适用性守卫，拒绝要有显式理由。
4. **数值不变量是 LLM 候选成功率的最大来源**：减 max 的 exp、fp32 累加器、
   全量 mask——这三类"方法"不提速但把失败变通过，应作为对应算子类的
   prompt 必选项而非可选优化。
5. **融合/中间消除是 memory-bound 的主导方法**：k 段逐元素链融合后流量比约 2/k，
   同时消除 launch 间隙；softmax/attention 类进一步用流式/online 归约
   （FlashAttention 同源机制）。
6. **高级流水线不是普适升级**：persistent kernel 在 tutorials/09 的实测中
   一个尺寸上慢于朴素版与 cuBLAS；收益强形状依赖，与设计文档 §6.3
   "不保证所有形状获益"一致。证据上必须逐形状验证。
7. **候选不可自证性能**：autotuner 内部 do_bench 计时只服务 config 选择，
   与正式 timing 两轨分离；生成器 prompt 必须显式禁止模型自测/自报 speedup
   （KEVIN 文献中亦报告了 timing 作弊型 reward hacking）。
8. **证据诚实分层**：本项目候选不被 profile（ADR-0003），NCU 证据刻画 baseline；
   生成器分类必须区分 ncu_full/ncu_partial/static_only/history_only 四档覆盖，
   MISSING 指标不得当 0。
9. **单因素变换纪律**在计划层可行：每个 MethodPlanItem 只授权一个主要机制，
   正确性护栏（mask/累加器）作为 GUARDS 段落随行，不占"因素"名额。
10. **本机先验缺失**：expected_gain 数字多来自 A100/H100（如 GROUP_SIZE_M
    在 A100 上 220→245 TFLOPS），SM89 消费卡上必须视为未标定先验，
    排序权重保守起步，靠本机方法记忆逐步校准。

## 与产出文件的分工

- `methods.yaml`：方法目录（meta + 38 条目；字段含 tier 分层适用信号、
  expected_gain、risks_preconditions、triton_howto、references）。
- `generator-design.md`：输入 schema（对接 ncu.py MetricCatalog 字段、
  methods.py TaskProfile、T16 feedback 阶段）、瓶颈分类、匹配排序、
  MethodPlan/prompt 片段模板、与现有代码的新增/扩展对照表、离线回归评估设计。
