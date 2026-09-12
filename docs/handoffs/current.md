# 当前开发交接

- 更新时间：2026-09-12
- 归档终检：本会话收尾时全套检验通过——ruff check/format 干净、232 项 CPU 测试全过（run_id 47a1ca2871154042ae056a54e1553538）、KernelBench 快照校验 270/270（开发子集 18/8/9）、wheel 干净安装四模块导入正常、证据库 7 份报告 audit healthy；工作区干净，本地与 origin/main 同步。
- 当前工作包：T12a（模型客户端离线层），ACCEPTED；T03 READY_FOR_ACCEPTANCE（等实体卡）；T00/T01/T02/T08 ACCEPTED；其余等 GPU 链。
- 当前代码状态：T00 工程验收入口 + T01 domain 契约 + T02 KernelBench 数据适配 + T03 GPU 环境探测 + T08 证据存储 + T12a 模型客户端离线层 + T04a 可信 worker 进程隔离机制（私有工作目录、环境净化、超时杀进程树、可信目录篡改检测，293 项测试）。
- 已完成：详细设计、任务计划、T00–T03、T08、T12a、T04a，已上传 GitHub 公开仓库 https://github.com/yuan-jc/kernelagent。
- 已通过检查：本地 293 项 CPU 测试、ruff、wheel 干净安装含 worker 执行冒烟；修复后探针实测 native 6/6 全 pass、WSL kernel 启动 pass，报告已入证据库。
- 被验证实现：T03 提交 937b564、T12a 提交见 docs/work-packages/T12a.md 与 docs/evidence/T12a.json。
- 关键环境发现（2026-09-12 审查后修正）：GPT-6 审查发现 probe 把设备地址直接放进 kernelParams（缺一层间接）且绑定了旧内存 ABI 符号——此前"GPU-PV 拒绝回读/WSL 驱动 SIGSEGV"的结论均被该缺陷污染，已标记 SUPERSEDED。修复后探针实测：本机 Windows native 6/6 全 pass（设备内存往返写读 42），WSL kernel 启动也 pass（仅 nvcc 缺失、ncu shim 损坏）。T03 正式验收仍按冻结规格等待目标 Ubuntu + NVIDIA 环境复跑，本机通过不替代验收。审查修复详情见 docs/evidence/review-fixes-2026-09-12.json。
- 尚未执行：T04–T07、T09–T11、T12 主体（真实模型闭环）、T13–T25；没有 GPU/模型实验。
- 网络与工具：GitHub 走本机代理 127.0.0.1:7893（仓库级 git http.proxy）；CI 失败用"干净 clone + uv sync --locked + 复现 workflow 命令"本地诊断。
- 活动作业：无。
- 下一步：(a) 实体卡/真实 Ubuntu 就绪 → 复跑 `kernelagent probe` 验收 T03 → 父包 T04 用真实 GPU 评测请求验收 worker；(b) VM 使用窗口期无其它可诚实推进的工作包（T13/T14/T18–T21 均依赖 GPU 链上未验收的包，不硬凑）。复查命令：`uv run --locked kernelagent check --output artifacts/local`。

## 后续每次交接必须补齐

1. 当前 task/subtask 与实际代码版本或工作区差异。
2. 实现了哪些可观察行为。
3. 执行过的精确命令、退出状态、原始日志和环境位置。
4. 失败/未执行检查及原因；不得省略 GPU NOT_RUN。
5. 活动作业 ID、工作目录、结果位置和状态查询方式。
6. 未提交变更/接口变更/已知风险。
7. 下一条可执行命令或所需输入。
