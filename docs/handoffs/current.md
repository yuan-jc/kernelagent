# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T01，ACCEPTED；T02 尚未启动，T03 及以后未动。
- 当前代码状态：T00 工程验收入口 + T01 核心 domain 契约（12 个 frozen 契约对象、版本化序列化、内容 hash 身份、import 边界测试）；无 GPU 优化功能。
- 已完成：详细设计、任务计划、T00、T01 实现，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：本地 117 项 CPU 测试（含 T00 回归 5 项）、ruff、打包与干净 wheel 安装 domain 导入。
- 被验证实现：T01 工作包卡片记录的提交（见 docs/work-packages/T01.md 与 docs/evidence/T01.json）。
- 尚未执行：T02 及全部后续工作包；没有 GPU/模型实验。
- 环境未知项：目标 NVIDIA GPU、Linux worker 访问方式、NCU 权限、运行时模型配置。
- 活动作业：无。
- 下一步：用户要求继续时，按 T02 开发 KernelBench 数据适配（固定 SHA 快照、开发清单、缺失 ID 明确错误）；先填写工作包卡片。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
