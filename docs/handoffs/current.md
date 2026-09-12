# 当前开发交接

- 更新时间：2026-09-12
- 当前工作包：T04a 修复轮，**READY_FOR_ACCEPTANCE**（审查发现 5 项缺陷已修复并附反例证据，待独立审查确认后恢复 ACCEPTED）。
- 当前唯一有效状态：本地与 origin/main 同步于 `6f40d49`；本轮修复提交后以该提交为准（提交 SHA 与 CI 绑定记录在 docs/evidence/review-fixes-t04a.json）。
- 当前代码状态（单一事实）：T00 工程验收入口、T01 domain 契约（严格反序列化）、T02 KernelBench 数据适配、T03 GPU 环境探测（READY_FOR_ACCEPTANCE）、T04a 可信 worker 进程隔离（修复轮）、T08 证据存储（消费时校验）、T12a 模型客户端离线层；**当前测试基线 302 项全过**（历史数字 15/117/150/232/269/293 均为各时点快照，不再引用）。
- 已完成：详细设计、任务计划、T00–T03、T08、T12a、T04a（修复轮），已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查（本轮）：302 项 CPU 测试、ruff check/format、KernelBench 快照 270/270、wheel 干净安装含严格契约与 worker 冒烟。
- 环境结论（审查后修正，旧归因作废）：修复前探针存在 kernelParams 间接寻址与旧 ABI 缺陷，此前"GPU-PV 拒绝回读/WSL 驱动 SIGSEGV"的结论 SUPERSEDED。修复后探针实测：本机 native 6/6 全 pass（设备内存往返写读 42），WSL kernel 启动 pass（nvcc 缺失、ncu shim 损坏为真实缺口）。审查修复详情见 docs/evidence/review-fixes-2026-09-12.json 与 docs/evidence/review-fixes-t04a.json。
- T03 唯一真实阻塞：目标 Ubuntu + NVIDIA 环境的正式验收尚未执行（冻结规格：报告有效 + cuda_kernel_launch pass + 42 写读验证 + 证据入库）。本机通过不替代验收。
- 尚未执行：T04 父包（容器级隔离 + 真实 GPU 评测请求）、T05–T07、T09–T11、T12 主体（真实模型闭环）、T13–T25；没有 GPU/模型实验。
- 网络与工具：GitHub 走本机代理 127.0.0.1:7893（仓库级 git http.proxy）；CI 失败用"干净 clone + uv sync --locked + 复现 workflow 命令"本地诊断。
- 活动作业：无。
- 下一步：(a) 本轮修复经独立审查确认后 T04a 恢复 ACCEPTED；(b) 目标 Ubuntu 环境就绪 → 复跑 probe 验收 T03 → 按 GLM-next 分包计划推进 T04 父包（容器隔离 + 真实 GPU 评测请求）→ T05。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
