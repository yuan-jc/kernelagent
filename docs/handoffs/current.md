# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T03，READY_FOR_ACCEPTANCE（探测工具在真实硬件验证完成，GPU 验收被环境阻塞）；T00/T01/T02/T08 ACCEPTED；T04+ 等待 T03 转 ACCEPTED。
- 当前代码状态：T00 工程验收入口 + T01 domain 契约 + T02 KernelBench 数据适配 + T08 证据存储 + T03 GPU 环境探测（自包含跨平台 probe、驱动检查子进程隔离、CUresult 全程留痕、native/wsl 双传输、EvidenceStore 集成，200 项测试）。
- 已完成：详细设计、任务计划、T00–T03、T08，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：本地 200 项 CPU 测试、ruff、wheel 探测冒烟；两份真实探测报告（native sha256 3715b324…、wsl b2ab7ab0…）已入证据库（audit healthy）。
- 关键环境发现：本 Windows 会话是 VM，GPU 为 GPU-PV 透传（RTX 4060 可查询/JIT/启动/同步，但设备内存分配 0xc9、结果回读 0xc0000006）；嵌套 WSL 驱动执行层 SIGSEGV；WSL 的 ncu 是 Windows shim 不可用。**用户将安装实体 NVIDIA 卡**——装好后复跑 `kernelagent probe`，kernel_launch=pass 即 T03 转 ACCEPTED 并解锁 T04。
- 尚未执行：T04–T07、T09–T25；没有 GPU/模型实验。
- 网络与工具：GitHub 走本机代理 127.0.0.1:7893（仓库级 git http.proxy）；日志/工件 API 需认证，CI 失败用"干净 clone + uv sync --locked + 复现 workflow 命令"本地诊断。
- 活动作业：无。
- 下一步：用户装好实体卡后执行 `uv run --locked kernelagent probe --target native --output artifacts/t03-native2 --evidence-root artifacts/t03-evidence`，cuda_kernel_launch=pass 即验收 T03 → 推进 T04（可信 worker 边界）。无卡窗口期可考虑 T12 离线前置（需用户同意放宽依赖门）。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
