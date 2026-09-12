# NVIDIA 算子优化 Agent 设计

T00 工程基础已验收：Python 包、CPU 验收命令、依赖锁和 CI。[四组跨平台 CI 已通过](https://github.com/yuan-jc/kernelagent/actions/runs/34667461881)。尚未实现 GPU 优化 Agent，T01 及后续功能未开始。

## 本地开发（Windows / Linux）

需要 Python 3.11–3.14；CI 验证 Python 3.11 和 3.13。

```bash
python -m pip install uv==0.12.13
python -m uv sync --locked
python -m uv run --locked kernelagent --version
python -m uv run --locked kernelagent check --output artifacts/local
python -m uv run --locked ruff check .
python -m uv run --locked ruff format --check .
python -m uv build --no-sources
```

运行时无第三方依赖；开发环境安装 pytest/jsonschema/ruff，不安装 torch、Triton 或 CUDA。请在仓库根目录执行验收命令。

`kernelagent check` 运行可信项目 pytest 测试，生成唯一运行目录，包含 JSON 报告、JUnit、stdout/stderr 和 SHA-256。失败、零测试、全 skip、collection error 或超时均非零退出。混合 skip 明确计数，需要各工作包再判断覆盖是否足够。默认超时为 300 秒，可用 `--timeout` 调整。

此命令是开发检查入口，不是不可信 kernel 的沙箱，不代表 GPU 正确性/性能验收，也不会自动修改任务状态。GPU worker 在 T04 开发。

报告格式见 [CPU 检查 schema](schemas/cpu-test-run.schema.json)；本包验收见 [T00 工作记录](docs/work-packages/T00.md)。GitHub Actions 在 Windows/Ubuntu 上执行相同检查，并保留运行 artifact。

## 设计与任务

- [详细设计文档](docs/NVIDIA算子优化Agent设计文档.md)：项目选型、面向对象架构、方法生成器、证据链、可信评测、具体接入方法与实施计划。
- [开发任务与验收计划](docs/开发任务与验收计划.md)：26 个工作包、阶段门、额度窗口安排与会话指令。
- [任务状态](docs/task-board.json) / [当前交接](docs/handoffs/current.md)：跨会话推进依据。
- [来源索引](research/SOURCES.md)：官方项目固定 commit、NVIDIA 文档与本地快照。
- [快照清单](research/snapshot_manifest.json)：文件来源与 SHA-256。

推荐起点是 KernelBench + 现有 Meta/PyTorch KernelAgent 适配器；优先复用工具，在统一评测与证据接口上扩展自有方法生成能力。

调研日期：2026-09-12。调研快照不是运行环境兼容性认证；未进行 GPU 性能测试。公开仓库只包含来源索引/清单与获取脚本，第三方原始快照保留本地，不随仓库分发。
