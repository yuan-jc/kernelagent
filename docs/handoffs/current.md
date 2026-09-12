# 当前状态与下一步

首用环境：**Ubuntu + NVIDIA GPU**。先按 [Ubuntu 指南](../UBUNTU.md)自检与恢复数据，然后只推进 [T03](../work-packages/T03.md)。状态清单见 [task-board](../task-board.json)，有效检查索引见 [current.json](../evidence/current.json)。

| 范围 | 状态与能力边界 |
|---|---|
| T00/T01/T02/T08/T12a | ACCEPTED：工程、纯契约、静态基准适配、证据存储、离线模型客户端；当前 CPU 回归持续覆盖 |
| T03 | READY_FOR_ACCEPTANCE：探针可运行；目标 Ubuntu/NVIDIA 正式验收 NOT_RUN |
| T04a | READY_FOR_ACCEPTANCE：进程执行机制与失败回归可用；目标运行环境需复核，不等于沙箱 |
| T04 | TODO：可信容器边界、资源控制、真实 GPU 请求尚未实现/验收 |
| T05–T25（除 T08 及 T12a 子包） | 按任务计划依赖推进，未实现完整自动优化闭环 |

## 本次移交内容

- 清理重复工作流水、模型专用会话指令和一次性调研脚本；设计基线、冻结清单、来源索引保留。
- README 与 Ubuntu 指南成为新机器入口。KernelBench 获取脚本从提交清单恢复缓存，不依赖历史 tree.json，不修改清单；已在空目录完整恢复并校验。
- worker 收尾补充：部分日志打开失败关闭已开的句柄；日志回读失败返回 infra_error；基础设施失败仍回收直接载荷；错误保留目录策略覆盖 infra_error；Windows 线程恢复与 Job 关闭结果明确检查。
- 测试结果、实现身份与发布 CI 见证据索引，不在多个文档重复维护测试数字。

## 已知限制

- 当前 POSIX 进程组回收不能阻止 setsid 逃逸；文件权限、网络和 GPU/内存资源边界必须由 T04 容器/cgroup 强制执行。只用受控载荷测试现有 runner。
- 记录状态 completed 只表示进程成功退出，不能代表候选正确、性能有效或允许晋升。
- 模型层只有离线能力；回放要求请求/响应 model_id 精确一致。未来 provider 的别名映射须显式设计并测试。
- 证据存储约定单写入器、同一内容 hash 对应一组索引元数据；hash 是完整性检查，不是抵御整个存储被任意修改的签名。
- GPU 环境版本与容器镜像尚未冻结，正式测量与模型调用均未执行。

## 下一步

在目标 Ubuntu 执行 `uv run --locked kernelagent check --output artifacts/ubuntu-cpu`，然后按 T03 卡片产出真实 GPU 报告。T03 验收通过再进入 [T04](../work-packages/T04.md)，不要从旧的“等待实体卡/GPU-PV”描述推断环境原因。

本地调研/实验缓存未随 Git 分发。清理前记录可在提交 `1b4a306` 查阅；不要将历史摘要当作当前验收事实。每包收尾替换本页的当前状态、活动作业、结果位置与下一步，避免追加聊天流水。

当前无持续 GPU/模型实验作业。发布检查以证据索引和本次提交对应 CI 为准。
