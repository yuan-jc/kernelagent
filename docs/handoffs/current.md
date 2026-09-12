# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T02，ACCEPTED；下一个可做包是 T08（依赖 T01 已满足）与 T09（依赖 T07/T08，暂不可做）；T03 及 GPU 链未动。
- 当前代码状态：T00 工程验收入口 + T01 核心 domain 契约 + T02 KernelBench 数据适配（只读快照读取器、双 hash 清单、冻结开发清单 35 题、协议分离、CLI bench verify、150 项测试）。
- 已完成：详细设计、任务计划、T00、T01、T02，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：T02 本地 150 项 CPU 测试、ruff、真实快照 bench verify（270/270 + 协议引用 + 子集 18/8/9）、CLI 篡改负例 exit 1、wheel 打包含 adapters。
- 被验证实现：T02 工作包卡片记录的提交（见 docs/work-packages/T02.md 与 docs/evidence/T02.json）。
- 尚未执行：T05/T07 及所有 GPU 检查；T08 及以后；没有 GPU/模型实验。
- 环境未知项：目标 NVIDIA GPU、Linux worker 访问方式、NCU 权限、运行时模型配置。网络：GitHub 访问走本机代理 127.0.0.1:7893（已配置为仓库级 git http.proxy）。
- 活动作业：无。
- 下一步：用户要求继续时，按 T08 开发证据存储与身份（SQLite 索引、artifact、追加事件、原子发布和查询；损坏 hash/孤立 artifact 检出；环境/协议变更不误命中；重复 experiment 不产生矛盾终态）；先填写工作包卡片。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
