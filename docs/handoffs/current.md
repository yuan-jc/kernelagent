# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T02，IN_PROGRESS（数据准备检查点完成，adapter 未实现）；T01 ACCEPTED；T03 及以后未动。
- 当前代码状态：T00 工程验收入口 + T01 核心 domain 契约；T02 已完成 KernelBench 快照题目抓取（270 文件过 blob SHA 校验，hash 清单入库），adapter/测试/CLI 未写。
- 已完成：详细设计、任务计划、T00、T01，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：T01 本地 117 项 CPU 测试、ruff、wheel；T02 数据清单校验（命令与 hash 见 docs/work-packages/T02.md）。
- 被验证实现：T01 提交 `0813410`；T02 检查点随后提交。
- 尚未执行：T02 验收表全部检查项（adapter 未实现）；T03 及以后；没有 GPU/模型实验。
- 环境未知项：目标 NVIDIA GPU、Linux worker 访问方式、NCU 权限、运行时模型配置。网络：GitHub 访问需走本机代理 127.0.0.1:7893（已配置为本仓库 git http.proxy，仅仓库级）。
- 活动作业：无。
- 下一步：继续 T02——实现 `src/kernelagent/adapters/benchmarks/kernelbench.py`、`configs/kernelbench/dev-manifest.json`、CLI verify 子命令与 test_kernelbench_*.py 正反例，按卡片冻结验收表逐项验收后提交推送。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
