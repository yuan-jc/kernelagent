# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T08，ACCEPTED；G0 剩 T03（GPU 环境探测，需用户提供 Linux NVIDIA worker），G1 剩 T04–T07/T09/T10（均依赖 GPU 链或 T07）。
- 当前代码状态：T00 工程验收入口 + T01 domain 契约 + T02 KernelBench 数据适配 + T08 证据存储与身份（内容寻址 artifact、SQLite 追加式索引、evaluation_key 身份、实验记录不可变、完整性审计、事件流，179 项测试）。
- 已完成：详细设计、任务计划、T00、T01、T02、T08，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：T08 本地 179 项 CPU 测试、ruff、wheel 干净安装含 storage 功能验证、audit 健康检查。
- 被验证实现：T08 工作包卡片记录的提交（见 docs/work-packages/T08.md 与 docs/evidence/T08.json）。
- 尚未执行：T03–T07、T09–T25；没有 GPU/模型实验。
- 环境未知项：目标 NVIDIA GPU、Linux worker 访问方式、NCU 权限、运行时模型配置。网络：GitHub 走本机代理 127.0.0.1:7893（仓库级 git http.proxy 已配置）。
- 活动作业：无。
- 下一步：两条路任选——(a) 用户提供 Linux GPU 环境后推进 T03（环境探测）；(b) 无 GPU 时先做 T12 的离线前置（ModelClient 结构化解析/异常分支/成本入账，用录制响应验证，真实验证留到有凭据时）。均需先填写工作包卡片。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
