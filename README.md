# KernelAgent

面向 **Ubuntu + NVIDIA GPU** 的算子优化 agent：自动执行、证据驱动、面向对象，优先复用 KernelBench、Triton 和现有优化工具。

目前是可测试的工程基础，**尚未实现完整 GPU 自动优化闭环**。已具备纯 Domain 契约、KernelBench 静态适配、证据存储、环境探针、进程执行机制和模型客户端离线层。下一步是目标 Ubuntu 环境验收与可信 GPU worker。

## 从这里开始

1. [Ubuntu 上手指南](docs/UBUNTU.md)：安装、恢复固定基准、CPU 自检、GPU 探测。
2. [当前状态与下一步](docs/handoffs/current.md)：唯一当前交接；机器状态见 [task board](docs/task-board.json)。
3. [设计文档](docs/NVIDIA算子优化Agent设计文档.md)：对象职责、工具复用、评测与证据协议。
4. [任务与验收计划](docs/开发任务与验收计划.md)：T00–T25 的依赖、范围与阶段门。

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
| `adapters/models/` | 离线回放、解析、有限重试、token 预算；尚无真实提供方 |
| `src/kernelagent/worker/` | 私有目录、进程生命周期、环境过滤；不是不可信代码沙箱 |
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
