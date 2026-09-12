# NVIDIA 算子优化 Agent 设计

T00 工程基础、T01 核心 domain 契约、T02 KernelBench 数据适配、T03 GPU 环境探测（READY_FOR_ACCEPTANCE）、T08 证据存储与 T12a 模型客户端离线层已验收：CPU 验收命令、不可变契约对象与内容 hash 身份、KernelBench 快照读取器、追加式证据存储、跨平台 GPU 探测（`kernelagent probe`），以及模型客户端离线层（请求身份 hash、录制回放、有限重试、成本预算、结构化解析）。[T00 四组跨平台 CI 已通过](https://github.com/yuan-jc/kernelagent/actions/runs/34667461881)。真实 GPU 优化闭环（T04+）等待实体卡环境验收。

## 模型客户端离线层（T12a）

```python
from kernelagent.adapters.models import (
    ModelRequest, ModelResponse, ModelUsage,
    RecordedModelClient, RetryingModelClient, CostLedger, TokenBudget,
    parse_structured,
)

request = ModelRequest(model_id="glm-5.3", messages=(("user", "propose"),))
# RecordedModelClient 按 request.request_sha256 精确回放；RetryingModelClient
# 仅对瞬时错误有限重试；CostLedger/TokenBudget 让 token 成本成为一等入账对象；
# parse_structured 把模型输出解析为校验过的 JSON（失败显式 ParseError）。
```

录制响应只用于控制流验证，不冒充生成能力；真实模型调用与"模型→代码→GPU"闭环属 T12 主体验收。

## 证据存储（T08）

```python
from kernelagent.adapters.storage import EvidenceStore, evaluation_key, new_experiment_id

with EvidenceStore(root) as store:
    ref = store.put_artifact(b"raw timing samples", kind="timing_samples", producer_version="t08")
    key = evaluation_key(impl_id, workload_sha, protocol_sha, environment_sha)
    store.record_experiment(new_experiment_id(), key, impl_id, "passed", [ref])
    assert store.audit().healthy
```

存储是追加式的：artifact 按内容寻址、原子发布、读取时验证 hash；实验记录不可变（重复提交相同内容幂等，不同内容硬冲突）；环境/协议变更必然改变 `evaluation_key`，缓存永不误命中；`audit()` 检出损坏、缺失与孤立 artifact。

## KernelBench 快照（T02）

```bash
python -m uv run --locked kernelagent bench verify   # 校验本地快照与提交的清单一致
```

快照题目文件留在本地 gitignored 缓存，公开仓库只携带 hash 清单（`configs/kernelbench/`）；开发清单为上游代表子集的引用，不是新题库。adapter 只读元数据与字节，不执行题目代码——加载与评测属于 T04/T05 的 GPU worker。

## 核心 domain（T01）

`kernelagent.domain` 提供纯契约层：frozen dataclass + 构造期校验、`schema_version` 信封序列化（未知版本/篡改字段明确拒绝）、按设计 §11.1 从内容派生的 `implementation_id`。domain 不导入 torch、Triton、模型 SDK 或任何 adapter，由 AST 边界测试与子进程导入测试双重锁定。示例：

```python
from kernelagent.domain import IOPort, OperatorSpec, dumps, loads

spec = OperatorSpec(
    operator_id="matmul",
    granularity="operator",
    inputs=(
        IOPort(name="a", dtype="float32", shape=(-1, -1)),
        IOPort(name="b", dtype="float32", shape=(-1, -1)),
    ),
    outputs=(IOPort(name="y", dtype="float32", shape=(-1, -1)),),
)
restored = loads(dumps(spec))  # 往返保持语义；非法字段在构造期被拒绝
```

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
