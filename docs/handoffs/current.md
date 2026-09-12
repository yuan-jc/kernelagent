# 当前状态与下一步

首用环境：**Ubuntu + NVIDIA GPU**（已确认）。环境安装见 [Ubuntu 指南](../UBUNTU.md)，状态清单见 [task-board](../task-board.json)，有效检查索引见 [current.json](../evidence/current.json)。

| 范围 | 状态与能力边界 |
|---|---|
| T00/T01/T02/T08/T12a | ACCEPTED：工程、纯契约、静态基准适配、证据存储、离线模型客户端；当前 CPU 回归持续覆盖 |
| T03 | ACCEPTED（2026-09-12）：目标 Ubuntu/NVIDIA 真机验收通过；探针 4 pass / 0 fail / 2 unavailable（nvcc、ncu 不在 PATH） |
| T04a | READY_FOR_ACCEPTANCE：进程执行机制与失败回归可用；目标运行时已由 T03 确认，Ubuntu 复核待做 |
| T04 | TODO：可信容器边界、资源控制、真实 GPU 请求尚未实现/验收 |
| T05–T25（除 T08 及 T12a 子包） | 按任务计划依赖推进，未实现完整自动优化闭环 |

## 本次移交内容（2026-09-12，ZCode/GLM）

- 目标机建立锁定开发环境：`.bootstrap` 内 uv 0.12.13 + uv 托管 Python 3.11.16；`uv sync --locked` 13 个依赖。系统 `python3.13-venv` 缺失导致 `python3 -m venv` 失败，改用 `venv --without-pip` + 系统 pip `--python` 目标安装，未动系统 Python、未用 sudo。
- CPU 自检全过：ruff check/format、`kernelagent check`（315 项：312 过 / 3 跳过均为 Windows Job Object 专用 / 0 失败）、`examples/worker_smoke.py` status=completed、`uv build` sdist+wheel。
- 真实 GPU 探针（`--target native`）：nvidia_smi、cuda_driver_api、cuda_kernel_launch 均 pass；PTX JIT kernel 真实写 42 并经设备内存回读验证，13 步 CUresult trace 全为 0x0。报告通过 `schemas/gpu-probe.schema.json` 校验，sha256 `ab9f5627797f…` 已入 EvidenceStore，audit healthy。GPU 身份：RTX 4060 Laptop / UUID `GPU-ea248ec5-1f33-d90f-c598-1c94dfdc6998` / 驱动 580.95.05 / CC 8.9。
- nvcc、ncu 缺失按探针语义记为 `unavailable`：影响为 T04 容器镜像需自带 CUDA toolkit、T05 上游评测需 nvcc、T15 真实 `.ncu-rep` profiling 需 ncu。本轮未安装，也未改动驱动。
- 证据与结果位置：`artifacts/ubuntu/`（commit.txt、worktree.patch（空）、os.txt、nvidia-smi.txt、probe/、probe-run.txt、evidence/）与 `artifacts/ubuntu-cpu/2ebe2e803bd14621a9fce08bf049c1bf/`；索引见 `docs/evidence/current.json` 的 `target_ubuntu_t03`。

## 已知限制

- 当前 POSIX 进程组回收不能阻止 setsid 逃逸；文件权限、网络和 GPU/内存资源边界必须由 T04 容器/cgroup 强制执行。只用受控载荷测试现有 runner。
- 记录状态 completed 只表示进程成功退出，不能代表候选正确、性能有效或允许晋升。
- 模型层只有离线能力；回放要求请求/响应 model_id 精确一致。未来 provider 的别名映射须显式设计并测试。
- 证据存储约定单写入器、同一内容 hash 对应一组索引元数据；hash 是完整性检查，不是抵御整个存储被任意修改的签名。
- 宿主机无 nvcc/ncu；容器镜像与 GPU 运行环境版本尚未冻结（T04/T05 范围）。NCU 版本查询不属本包；真实 profiling 属 T15。

## 下一步

进入 [T04](../work-packages/T04.md)：复用容器与 NVIDIA Container Toolkit（已确认 `nvidia-container-cli` 1.19.1、Docker 28.2.2），落实可信 evaluator 与候选分离、禁网、只读可信输入、限额及进程回收。随后 T04a 在 Ubuntu 复核进程机制，再 T05 原生基线。

本地调研/实验缓存未随 Git 分发。清理前记录可在提交 `1b4a306` 查阅；不要将历史摘要当作当前验收事实。每包收尾替换本页的当前状态、活动作业、结果位置与下一步，避免追加聊天流水。

当前无持续 GPU/模型实验作业。T03 验收对应本页记录的本地命令与证据；推送后按惯例绑定发布 CI。
