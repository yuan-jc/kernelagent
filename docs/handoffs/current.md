# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T00，ACCEPTED；本次到此结束，T01 尚未启动。
- 当前代码状态：Python 包、CPU check CLI、JSON Schema、uv.lock、pytest 正反例和跨平台 CI 已实现；无 GPU 优化功能。
- 已完成：详细设计、任务计划、T00 实现，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：本地 15 项 CPU 测试、ruff、打包、干净 wheel 安装；远程 Ubuntu/Windows × Python 3.11/3.13 全部通过，详情见 `docs/work-packages/T00.md` 和 `docs/evidence/T00.json`。
- 被验证实现：`d549f92ebf7f904f15d9f4c45c21cac475d95f6f`；之后仅更新验收记录。
- 尚未执行：T01 及全部后续工作包；没有 GPU/模型实验。
- 环境未知项：目标 NVIDIA GPU、Linux worker 访问方式、NCU 权限、运行时模型配置。
- 活动作业：无。
- 下一步：用户要求继续时，按 T01 开发核心 domain；先填写工作包卡片。复查命令：`python -m uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
