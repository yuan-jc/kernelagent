# GLM 与 Codex 协作交接

本文件供 ZCode/GLM Agent 与 Codex 在 `/home/y/kernelagent` 项目中交换当前任务、验证结果和阻塞信息。
它是协作入口，不替代以下正式事实来源：

- 项目约束：`AGENTS.md`
- 当前状态：`docs/handoffs/current.md`
- 任务状态：`docs/task-board.json`
- 当前工作包：`docs/work-packages/T03.md`
- 设计基线：`docs/NVIDIA算子优化Agent设计文档.md`

不要在本文件追加长篇聊天流水。每轮交接时覆盖“当前交接快照”中的对应字段。

## 一、当前交接快照

- 更新时间：2026-09-13
- 交出方：ZCode/GLM Agent（T05 验收轮）
- 接收方：Codex / 用户
- 仓库：`/home/y/kernelagent`
- 基线分支：`main`
- 基线提交：`15ee8a4`（T05 验收 + lint；其前 `db7e4a1` T05 契约与适配器、`4ae3489` 宿主工具链记录、`9e593cb` T04 验收）
- 当前工作包：**T07 正式计时协议（GLM 下一轮认领）；T06 增强正确性未被认领，可由 Codex 认领**
- 当前状态：T00–T05、T08、T12a 均 `ACCEPTED`（T03/T04/T04a/T05 为目标真机验收；T05 correctness-only，正式计时归 T07）。宿主工具链、ncu sudo 配置等事项见 `docs/handoffs/current.md`。

### 目标机器已知事实

- 系统：Ubuntu 25.04，x86_64
- Python：系统 Python 3.13.3；开发环境 uv 托管 Python 3.11.16
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- GPU UUID：`GPU-ea248ec5-1f33-d90f-c598-1c94dfdc6998`
- 驱动：580.95.05
- CUDA 计算能力：8.9
- 显存：8188 MiB
- Docker：28.2.2
- NVIDIA Container Toolkit：`nvidia-container-cli` 1.19.1
- 宿主工具链：CUDA 13.0.2 用户级（nvcc/ncu/nsys/compute-sanitizer/cuda-gdb）；非 root ncu profiling 待用户 sudo 开启计数器权限
- 网络备注：Docker Hub 直连超时（daemon 无代理），`docker.m.daocloud.io` 可直连；下载用 Mihomo `127.0.0.1:7890`

`nvcc`、`ncu` 缺少时必须按探针语义记录为 `unavailable`，不能伪造为通过，也不要为了“全绿”盲目更换驱动或 CUDA。T03 先运行现有探针并保存真实结果，再判断工具缺口是否需要用户授权安装。

## 二、给 ZCode/GLM Agent 的当前指令

### 1. 读取顺序

开始前只读取以下内容，不要重复全量调研：

1. `AGENTS.md`
2. 本文件
3. `docs/handoffs/current.md`
4. `docs/work-packages/T03.md`
5. `docs/UBUNTU.md`
6. 与失败检查直接相关的源码和测试

### 2. 本轮范围

只推进 T03 及其必要修复，不实现 T04、容器 worker、KernelBench 动态评测、搜索策略、模型生成或自动优化循环。

本轮目标：

1. 建立锁定的 Python 3.11 开发环境。
2. 运行 CPU 自检并保存报告。
3. 运行真实原生 GPU 探针。
4. 核对 schema、GPU 身份、CUDA Driver API、真实 kernel 写入并回读 42、工具状态和 EvidenceStore 审计。
5. 若发现代码缺陷，先保留失败证据和最小复现，再做最小修复及回归测试。
6. 只有真实 GPU 验收满足 T03 卡片时，才能把 T03 改为 `ACCEPTED`。

### 3. 开始前检查

```bash
cd /home/y/kernelagent
git status --short --branch
git rev-parse HEAD
nvidia-smi
```

如果工作区已有不属于你的修改，不要覆盖或回滚；先在本文件的回执区说明。

### 4. 环境安装

优先按 `docs/UBUNTU.md` 使用仓库内的 `.bootstrap`，不要污染系统 Python：

```bash
cd /home/y/kernelagent
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv==0.12.13
export PATH="$PWD/.bootstrap/bin:$PATH"
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked kernelagent --version
```

下载超时时，使用下文的命令级代理重试。不要把代理写入 shell 启动文件或 Git 全局配置。若 `python3 -m venv` 因系统包缺失而失败，记录原始错误并请求用户授权后再使用 `sudo apt`。

### 5. T03 验收命令

```bash
cd /home/y/kernelagent
export PATH="$PWD/.bootstrap/bin:$PATH"

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked kernelagent check --output artifacts/ubuntu-cpu
uv run --locked python examples/worker_smoke.py
uv build --no-sources

mkdir -p artifacts/ubuntu
git rev-parse HEAD > artifacts/ubuntu/commit.txt
git diff > artifacts/ubuntu/worktree.patch
uname -a > artifacts/ubuntu/os.txt
nvidia-smi > artifacts/ubuntu/nvidia-smi.txt

uv run --locked kernelagent probe \
  --target native \
  --output artifacts/ubuntu/probe \
  --evidence-root artifacts/ubuntu/evidence
```

不要只看退出码。逐项核对 `docs/work-packages/T03.md` 的验收矩阵，并检查生成报告是否符合 `schemas/gpu-probe.schema.json`。

## 三、外网与 GitHub 访问手册

### 网络事实

本机的 Cloudflare WARP 曾卡在 `Connecting`，造成 IPv4 公网和 DNS 超时。当前已执行 `warp-cli disconnect`，并确认 `Always On: false`。

Mihomo 当前以用户服务运行：

- HTTP/混合代理：`http://127.0.0.1:7890`
- DNS 监听：`127.0.0.1:1053`
- 控制端口：`127.0.0.1:9090`

不要同时启用 WARP 全局隧道与 Mihomo。不要修改路由、防火墙、系统 DNS 或 Mihomo 配置来“碰运气”。

### 1. 检查代理状态

```bash
warp-cli status
systemctl --user is-active mihomo.service
ss -lntup | rg '127\.0\.0\.1:(7890|1053|9090)'
curl -I --proxy http://127.0.0.1:7890 --max-time 15 https://github.com
```

预期：WARP 为 `Disconnected`，Mihomo 为 `active`，GitHub 返回 HTTP 响应。

如果 WARP 又进入 `Connecting`，先执行：

```bash
warp-cli disconnect
```

如果 Mihomo 没有运行，可执行：

```bash
systemctl --user restart mihomo.service
systemctl --user --no-pager status mihomo.service
```

重启失败时保留状态和日志，不修改订阅或代理配置。

### 2. Git 使用命令级代理

读取远端：

```bash
git -c http.proxy=http://127.0.0.1:7890 ls-remote origin HEAD
```

拉取更新：

```bash
git -c http.proxy=http://127.0.0.1:7890 fetch --prune origin
```

克隆其他仓库：

```bash
git -c http.proxy=http://127.0.0.1:7890 clone https://github.com/OWNER/REPO.git
```

不要执行：

```bash
git config --global http.proxy ...
git config --global https.proxy ...
```

### 3. Python、uv 与普通下载使用临时代理

只给单条命令传入代理：

```bash
HTTPS_PROXY=http://127.0.0.1:7890 \
HTTP_PROXY=http://127.0.0.1:7890 \
.bootstrap/bin/python -m pip install uv==0.12.13

HTTPS_PROXY=http://127.0.0.1:7890 \
HTTP_PROXY=http://127.0.0.1:7890 \
uv python install 3.11

HTTPS_PROXY=http://127.0.0.1:7890 \
HTTP_PROXY=http://127.0.0.1:7890 \
uv sync --locked --python 3.11
```

恢复 KernelBench 固定快照时优先使用脚本自身的显式参数：

```bash
uv run --locked python research/fetch_kernelbench_problems.py \
  --proxy http://127.0.0.1:7890
```

### 4. 故障分层检查

```bash
ping -c 3 192.168.1.1
ping -4 -c 3 1.1.1.1
timeout 8 getent ahostsv4 github.com
curl -I --max-time 15 https://github.com
curl -I --proxy http://127.0.0.1:7890 --max-time 15 https://github.com
```

解释顺序：

- 网关不通：局域网/Wi-Fi 问题。
- 网关通但公网 IPv4 不通：先检查 WARP 是否卡住。
- 直连不通但代理 curl 正常：对当前下载命令显式指定 Mihomo。
- 代理 curl 也失败：检查 Mihomo 服务和日志，不要立刻改系统网络。

### 5. 安全边界

- 不把代理订阅、API Key、Cookie、SSH 密码或访问令牌写入仓库、日志或本文件。
- 不提交 `.bootstrap/`、`.venv/`、下载缓存或 `artifacts/` 中可能含机器信息的大文件。
- 不关闭系统防火墙，不添加永久路由，不安装未知根证书。
- 不使用 `curl | sh`；先下载、校验来源，再按官方安装步骤执行。
- Git 推送、创建远端分支或 PR 前，先向用户确认目标和提交范围。

## 四、GLM 完成后的回执格式

GLM 完成或遇到阻塞时，请覆盖更新下面这一节，不要追加聊天记录。

### GLM 回执（第二轮：宿主 CUDA 工具链安装，2026-09-13）

- 更新时间：2026-09-13
- 当前分支与提交：`main` @ 本轮文档提交（工具链证据索引+交接更新；其下为并行会话的 T04/T04a 验收提交链 `45bef25`/`9875048`/`9e593cb`）。未推送、未建远端分支。
- 工作包状态：工具链安装属 T03 遗留缺口的用户授权补装，不改变包状态；T03/T04a/T04 已 ACCEPTED（后者由并行会话完成），下一步 T05。
- 行为变化：无仓库源码变化。宿主机新增用户级 CUDA 13.0.2 toolkit（`/home/y/toolchains/cuda-13.0`，~7.1GB；`~/toolchains/cuda` 稳定软链；`~/.profile` 导出 `CUDA_HOME` 并前置 PATH）；驱动未动。安装包保留 `~/cuda-dl/cuda_13.0.2_580.95.05_linux.run`（sha256 `81a5d0d0…`）供容器镜像复用。
- 修改文件：`docs/evidence/current.json`（新增 `target_ubuntu_toolchain`、`pending.NCU_COUNTER_PERMISSION`）、`docs/handoffs/current.md`（工具链移交章节、限制与下一步更新）；本文件。第一轮（T03 验收）回执见 Git 历史与本文件历史。
- 执行命令及退出码（关键项）：
  - 下载：8 段并行+断点续传经 Mihomo 代理，与并行会话协作完成（其间发生双写者分片损坏，坏片隔离后单写者重下；runfile makeself 负载 MD5 校验 OK）
  - `env -u DISPLAY sh cuda_13.0.2_580.95.05_linux.run --silent --toolkit --toolkitpath=/home/y/toolchains/cuda-13.0 --override --no-man-page --tmpdir=...` → 0（免 root；`/tmp` 为 7.5G tmpfs 故显式指定 tmpdir）
  - nvcc 修复：安装后一次软链覆盖事故致 `bin/nvcc` 变为自引用包装器（与并行会话同因），已从 runfile payload 按 cpio 偏移 14769 选择性提取 `builds/cuda_nvcc/bin/nvcc`（30153384 字节）原样还原；教训记录于证据索引（nvcc 按 `$0` 定位 TOP，禁用 `~/.local/bin/nvcc` 软链/包装）
  - `nvcc -O2 -arch=sm_89 toolchain_saxpy.cu -o toolchain_saxpy` → 0；运行 → 0（driver/runtime 13.0，saxpy n=1048576 数学校验通过）
  - `ncu --set basic ... ./toolchain_saxpy` → 应用正常运行但 profiling 被 ERR_NVGPUCTRPERM 拦截（`RmProfilingAdminOnly=1`，预期内，已记录）
  - `nsys profile ...` → 0（真实 `.nsys-rep` 189290 字节）
  - `uv run --locked kernelagent probe --target native --output artifacts/ubuntu/toolchain/probe --evidence-root artifacts/ubuntu/toolchain/evidence` → 0（**pass=6 / fail=0 / unavailable=0**；nvcc 13.0.88、ncu 2025.3.1.0 检查转 pass）
- CPU 验证：本轮未重跑（代码零改动；上一轮 312/3/0 与并行会话 331/3/0 均在案）。
- GPU 验证：PASS——见上探针与 saxpy 实测。
- 未通过或未执行项：非 root ncu 真实 profiling（需用户 sudo 配置计数器权限后重启；sudo ncu 当前可用）。KernelBench 快照恢复、torch/triton 冻结环境、T05 评测未动（T05 范围）。
- 证据路径与 hash：`artifacts/ubuntu/toolchain/{install.txt,nvcc_version.txt,ncu_version.txt,nsys_version.txt,nsys_test.txt,saxpy_compile_run.txt,ncu_permission_test.txt,toolchain_saxpy.cu}`；`artifacts/ubuntu/toolchain/probe/gpu-probe-report.json` sha256 `d5e5a0049dd3744ebc490af2de63238db265a4759327e104bfeb6cc538a9e1dd`（EvidenceStore audit healthy）。索引：`docs/evidence/current.json` 的 `target_ubuntu_toolchain`。
- 当前活动进程：无（所有下载/安装进程已退出）
- 阻塞与所需用户决定：无阻塞。请用户在方便时执行一条 sudo 动作（T15 前完成即可）：
  ```bash
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  # 之后重启（或重载 nvidia 模块）生效
  ```
- 建议 Codex 审查重点：① 工具链为用户级安装是否符合"T04/T05 环境冻结"预期（容器内 toolkit 仍以镜像为准，宿主 toolkit 供 T05 原生构建与探针）；② CUDA 13.0.2（对齐驱动 580.95.05）是否与 T05 计划的上游 KernelBench/PyTorch 版本兼容，如需 cu126/cu128 更低版本 toolkit 请指出；③ nvcc 调用约定（仅真实路径，`CUDA_HOME` 已入 `~/.profile`）是否需要写入 UBUNTU.md 正式文档。

回执完成后，同时按 `AGENTS.md` 更新正式的 `docs/task-board.json`、`docs/handoffs/current.md` 和 `docs/evidence/current.json`；只有实际检查证据支持时才改变状态。
