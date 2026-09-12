# 当前状态与下一步

首用环境：**Ubuntu + NVIDIA GPU**（已确认）。环境安装见 [Ubuntu 指南](../UBUNTU.md)，状态清单见 [task-board](../task-board.json)，有效检查索引见 [current.json](../evidence/current.json)。

| 范围 | 状态与能力边界 |
|---|---|
| T00/T01/T02/T08/T12a | ACCEPTED：工程、纯契约、静态基准适配、证据存储、离线模型客户端；当前 CPU 回归持续覆盖 |
| T03 | ACCEPTED（2026-09-12）：目标 Ubuntu/NVIDIA 真机验收通过；探针 4 pass / 0 fail / 2 unavailable（nvcc、ncu 不在 PATH） |
| T04a | ACCEPTED（2026-09-13）：Ubuntu 复核通过——check 315 项 312 pass / 3 Windows 专用 skip / 0 fail，worker_smoke completed；仅进程机制，不是沙箱 |
| T04 | ACCEPTED（2026-09-13）：容器边界（Docker+CDI、禁网、只读输入、非 root、cgroup 限额）+ 真实 GPU 请求通路（PTX JIT 写 42 验证通过、写 43 负例被父端拒绝）；ADR-0001 |
| T05–T25（除 T08 及 T12a 子包） | 按任务计划依赖推进，未实现完整自动优化闭环 |

## 本次移交内容（2026-09-13，ZCode/GLM，宿主 CUDA 工具链安装）

- 应用户要求安装宿主工具链（用户已授权安装，无 sudo 密码故全程免 root）：CUDA 13.0.2 runfile 用户级安装到 `/home/y/toolchains/cuda-13.0`（~7.1GB，`~/toolchains/cuda` 为稳定软链；`~/.profile` 导出 `CUDA_HOME` 并前置 PATH）。runfile sha256 `81a5d0d0…`、makeself 负载 MD5 校验通过；选 13.0.2 因其捆绑驱动 580.95.05 与主机完全一致。安装包保留在 `~/cuda-dl/` 供 T04/T05 容器镜像复用。
- 组件：nvcc 13.0.88、Nsight Compute 2025.3.1.4、Nsight Systems 2025.3.2.474、compute-sanitizer、cuda-gdb。注意 nvcc 脚本按 `$0` 定位顶层目录，只能从真实路径调用（一次软链覆盖事故已从 runfile payload 原样修复，记录于证据索引）。
- 端到端验证：`-arch=sm_89` 编译 SAXPY 并在 RTX 4060 上运行通过（driver/runtime 13.0，n=1048576 结果校验）；nsys 生成真实 `.nsys-rep`；重跑 gpu-probe：**6 pass / 0 fail / 0 unavailable**（nvcc、ncu 检查转 pass），报告 sha256 `d5e5a004…`，EvidenceStore audit healthy。
- 遗留一条 sudo 动作：非 root ncu profiling 被 `NVreg_RestrictProfilingToAdminUsers=1` 拦截（ERR_NVGPUCTRPERM）；sudo ncu 当前可用，普通用户需写入 `/etc/modprobe.d/nvidia-profiling.conf` 后重启。T15 前完成即可。
- 证据：`artifacts/ubuntu/toolchain/`（install.txt、各工具版本、saxpy 编译运行、ncu 权限测试、probe/ 与 evidence/）；索引见 `docs/evidence/current.json` 的 `target_ubuntu_toolchain`。

## 本次移交内容（2026-09-13，ZCode/GLM）

- T04a 转 ACCEPTED：`kernelagent check`（315 项 312 过 / 3 跳过 Windows 专用 / 0 失败）+ worker_smoke 在目标 Ubuntu 复核通过。
- T04 实现并转 ACCEPTED（ADR-0001 + 子步1 容器边界 + 子步2 GPU 通路）：
  - 容器执行器 `kernelagent.worker.container`：`--network none`、只读 rootfs 与只读可信输入、非 root uid 20000、cap-drop ALL、no-new-privileges、cgroup 内存/CPU/PID 限额、随机容器名、kill+rm 回收经 inspect 确认、container-record.json 证据；19/19 项边界测试在目标机真实运行（C1–C10：协议身份、自报拒绝、只读输入、路径越界、禁网、环境净化、超时/取消含 setsid 后代、内存/PID/输出/日志限额、基础设施失败）。
  - GPU 通路：`examples/container_gpu_smoke.py` 按冻结 PTX 载荷经容器边界在真实 GPU 上执行，父端独立验证；正例写 42 通过，负例写 43（exit 0）被父端拒绝——候选退出码/工件不能自证正确。GPU 按 CDI UUID 独占分配。
  - 实证发现（ADR-0001 修订）：容器退出后 tmpfs 随 mount namespace 销毁，`docker cp` 对已停止容器返回 rc=0 但目录为空——输出通道改为宿主 bind 目录 + 父端 0.5s 监视限额（内核 cgroup 限额兜底）。
  - 全量回归：334 项 331 过 / 3 Windows 专用 skip / 0 失败；报告 sha256 `e5eff8f1…`。
- 镜像与网络事实：Docker Hub 直连超时、无密码 sudo；`docker.m.daocloud.io` 直连可用。已按 digest 锁定 `python:3.11-slim-bookworm`（`sha256:528257d4…`）与 `ubuntu:24.04`（`sha256:224a1869…`）。
- 证据：`artifacts/t04-gpu/container-gpu-smoke.json`（sha256 `13c9fd64…`）、`artifacts/t04-cpu/32ef3cc1…/report.json`、`docs/work-packages/T04.md`、`docs/adr/0001-…md`；索引见 `docs/evidence/current.json` 的 `target_ubuntu_t04`。

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
- 宿主 nvcc/ncu 已装（CUDA 13.0.2 用户级，见上文工具链章节）；ncu 非 root profiling 有一条待用户执行的 sudo 配置。torch/triton 按计划仍归 T05 的冻结 GPU 环境。

## 下一步

进入 [T05](../../docs/开发任务与验收计划.md)：上游 KernelBench 基线与原生评测（依赖 T02/T03/T04 均已 ACCEPTED）。前置条件：宿主 nvcc/ncu 已就绪（CUDA 13.0.2 用户级，`CUDA_HOME=~/toolchains/cuda`）；KernelBench 快照恢复（`research/fetch_kernelbench_problems.py --proxy http://127.0.0.1:7890`）。验收要求：至少三个不同类别问题，已知正确候选通过、错误候选失败、与直接上游调用对照。

本地调研/实验缓存未随 Git 分发。清理前记录可在提交 `1b4a306` 查阅；不要将历史摘要当作当前验收事实。每包收尾替换本页的当前状态、活动作业、结果位置与下一步，避免追加聊天流水。

当前无持续 GPU/模型实验作业。T03 验收对应本页记录的本地命令与证据；推送后按惯例绑定发布 CI。
