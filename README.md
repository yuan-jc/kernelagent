# KernelAgent

面向 **Ubuntu + NVIDIA GPU** 的算子优化 agent：自动执行、证据驱动、面向对象，优先复用 KernelBench、Triton 和现有优化工具。

当前状态（v1 Alpha）：`kernelagent optimize` 产品入口已在真实 GPU 上打通
**生成 → 策略门 → 容器评测 → 增强正确性 → 正式计时 → 晋升 → 报告导出**的最小纵向闭环，
支持断点恢复与 durable 预算。Alpha 威胁模型为"合作型候选"（每份报告显式携带
`candidate_trust=cooperative`、`adversarially_secure=false`，见 ADR-0004）；
champion 晋升后需人工审查。真实模型从零生成（LIVE_MODEL）待配置 provider 凭据后开启。

## 从这里开始

1. [Ubuntu 上手指南](docs/UBUNTU.md)：安装、恢复固定基准、CPU 自检、GPU 探测。
2. [当前状态与下一步](docs/handoffs/current.md)：唯一当前交接；机器状态见 [task board](docs/task-board.json)。
3. [设计文档](docs/NVIDIA算子优化Agent设计文档.md)：对象职责、工具复用、评测与证据协议。
4. [任务与验收计划](docs/开发任务与验收计划.md)：T00–T25 的依赖、范围与阶段门。

## 快速上手（GPU 优化闭环）

前置：Ubuntu + NVIDIA GPU、Docker + NVIDIA Container Toolkit、已恢复 KernelBench 快照（见下）、
ADR-0002 评测镜像（构建见 `docs/UBUNTU.md` 与 `configs/eval-image/`）。

```bash
# 三组 Alpha 验收（固定正确/错误候选，真实 GPU 容器评测与计时）
.venv/bin/python examples/alpha_run.py --mode correct --output artifacts/alpha/run
.venv/bin/python examples/alpha_run.py --mode wrong   --output artifacts/alpha/run

# 真实模型生成（需要自己的 OpenAI 兼容凭据，仅存控制端环境变量）
export MODEL_PROVIDER_API_KEY='<你的密钥>'
kernelagent optimize \
  --problem kernelbench:l1:40 --backend triton \
  --model glm-4.5 --base-url '<OpenAI-compatible base URL>' \
  --max-candidates 5 --max-repair-rounds 2 \
  --gpu-budget-seconds 1800 --output artifacts/alpha/run

kernelagent status --output artifacts/alpha/run      # durable 状态与预算
kernelagent resume --base-url '...' --output artifacts/alpha/run  # 断点续跑
```

退出码：成功 0 / 无改进 1 / 预算耗尽 2 / 配置错误 3 / 基础设施错误 4。
详见 [Alpha Runbook](docs/alpha-runbook.md)。

所有命令都提供可直接执行的示例、参数单位、默认值和继承规则：

```bash
kernelagent --help
kernelagent optimize --help
kernelagent resume --help
kernelagent profile --help
```

其中 `--max-repair-rounds` 是允许失败/拒绝的候选数量，不会额外增加
`--max-candidates`；`resume` 未显式指定的运行参数从 `run_manifest.json` 恢复。
模型密钥只通过 `MODEL_PROVIDER_API_KEY` 提供，不应写入命令参数或运行目录。

## Web 控制台

```bash
.venv/bin/python -m kernelagent.webapp --port 8501   # http://127.0.0.1:8501
```

本地网页（标准库实现，零新增依赖）：填写 API Key（仅本机内存）、按 Level/Problem 选择 KernelBench、
实时展示候选阶段流水与加速比 CI、预算进度与历史运行。见 [docs/web-ui.md](docs/web-ui.md)。

## 快速自检

已有 Git 与 uv 0.12.13 时，在仓库根目录运行：

```bash
git clone https://github.com/yuan-jc/kernelagent.git
cd kernelagent
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked kernelagent check --output artifacts/local
uv run --locked python examples/worker_smoke.py
```

这组命令不需要 GPU，也不安装 torch/Triton。没有 uv 时按 [Ubuntu 安装步骤](docs/UBUNTU.md)操作。CPU CI 在 Ubuntu/Windows × Python 3.11/3.13 上运行；GPU 能力以目标机器的实际证据为准。

恢复 KernelBench 固定快照：

```bash
uv run --locked python research/fetch_kernelbench_problems.py
uv run --locked kernelagent bench verify
```

下载器从已提交清单恢复 270 个问题文件及协议引用，逐文件校验 SHA-256；不需要历史缓存，不改写清单。原始文件保存在忽略目录 `research/sources/`。

## 代码位置与能力边界

| 目录 | 当前能力 |
|---|---|
| `src/kernelagent/domain/` | 不可变对象、严格 JSON、内容身份；不依赖 GPU/模型 SDK |
| `adapters/benchmarks/` | 固定 KernelBench 文件与协议校验；尚不执行评测 |
| `adapters/storage/` | 追加式记录，artifact/索引/记录哈希验证 |
| `adapters/models/` | OpenAI 兼容真实提供方 + 离线回放、解析、有限重试、token 预算 |
| `src/kernelagent/worker/` | 容器边界（ADR-0001：禁网/只读输入/非 root/cgroup 限额）+ 进程机制 |
| `src/kernelagent/adapters/evals/` | pinned KernelBench 评测、增强正确性、正式计时（TimingProtocol v1） |
| `src/kernelagent/optimization.py` | `optimize/resume/status` 产品主循环：生成→策略门→评测→计时→晋升→持久化 |
| `src/kernelagent/probe.py` | 真实 CUDA 小探针与工具状态；不是性能 benchmark |

当前 worker 只运行受控开发载荷。任意模型生成代码必须等 [T04 容器隔离](docs/work-packages/T04.md)验收后执行。正式计时与 profiling 分开，候选不能自行宣布正确或晋升。

## 开发检查

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked kernelagent check --output artifacts/local
uv build --no-sources
```

`kernelagent check` 输出 JSON、JUnit、日志和 hash；失败、零测试、全跳过不会通过。具体 GPU/模型验收不由它替代。开发规则见 [AGENTS.md](AGENTS.md)，新包用[工作包模板](docs/work-package-template.md)。

[调研来源](research/SOURCES.md)和[固定来源清单](research/snapshot_manifest.json)保留选型依据。过时的会话指令、反复审查流水和一次性抓取脚本已移出当前目录，清理前历史可在提交 `1b4a306` 查阅。有效验收索引见 [current.json](docs/evidence/current.json)。
