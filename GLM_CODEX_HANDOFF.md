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
- 交出方：ZCode/GLM Agent（launch plan Task 1–6 轮）
- 接收方：Codex / 用户
- 仓库：`/home/y/kernelagent`
- 基线分支：`main`（含本轮提交链 `5e9e51f`→`69ef32c`，见回执）
- 当前工作包：**T16 IN_PROGRESS（仅差 U3 LIVE_MODEL）；下一步 launch plan Task 7（可信 MVP 信任域分离，落在 T23）**
- 当前状态：按 `CODEX_REVIEW_AND_LAUNCH_PLAN.md` 完成 Task 1–6——五项可信性缺陷全部修复（红-绿），
  `kernelagent optimize|resume|status` 产品入口交付，固定正确/错误候选 + SIGKILL 恢复三组 Alpha 验收在真实 GPU 通过。
  T17/T20/T22 按 launch plan 暂停；完整快照见 `docs/handoffs/current.md`。

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

### GLM 回执（launch plan Task 1–6 轮，2026-09-13）

- 更新时间：2026-09-13
- 当前分支与提交：`main` @ `69ef32c`。提交链（逐项红-绿、独立提交）：
  `5e9e51f`（Task 1 篡改反例 + AST 门 + ADR-0004）、`3d636df`（Task 2 durable 预算重建）、
  `3d2db2f`（Task 3 计时证据严格校验）、`db33c6d`（Task 4 dispatch guard + lint）、
  `754d8f8`（Task 5 optimize/resume/status 闭环）、`69ef32c`（Task 6 Alpha 验收 + pro driver triton 修复）。
- 行为变化：新增 `kernelagent optimize/resume/status` 产品入口与 `examples/alpha_run.py`；
  `inspect_candidate_policy`/`reconstruct_budget`/`validate_batch_samples` 新接口；
  T06 pro driver 对 triton 族 backend 改用 tempfile 加载器（真实 GPU 首跑暴露的集成缺口）。
  每份评测/优化报告携带 `candidate_trust=cooperative`、`adversarially_secure=false`（ADR-0004）。
- 执行命令及退出码：统一验收命令（计划 §6）每提交均跑——`ruff check src tests examples configs` exit 0；
  pytest 最终 **468 passed / 3 skipped / 0 failed**；`PYTHONPATH=src python -m kernelagent --help` 含 optimize/resume/status。
- GPU 验证（RTX 4060 Laptop，ADR-0002 镜像 `cb7a9f4c…`）：
  - U1 正确候选：state=completed，champion `fd0eac1c…`，晋升 CI [1.654, 1.706]（report sha256 `b1fec230…`）
  - U2 错误候选：evaluate 阶段 `correct=False` 拒绝，state=no_improvement（sha256 `313e3494…`）
  - SIGKILL→resume：`action_interrupted` 标记、恰 2 次结算、无重复计费（sha256 `c6d6bbca…`）
  - 首次 U1 失败证据保留在 `artifacts/alpha/layernorm-001/002`（pro baseline 失败 → 驱动修复的根因）
- 未通过或未执行项：**U3 LIVE_MODEL NOT_RUN**——等待用户提供 `MODEL_PROVIDER_API_KEY`（T12 同一阻塞）；
  T05–T07 在对抗候选下不可称可信（合作型候选限定，信任域分离归 Task 7/T23）。
- 证据路径：`docs/work-packages/T16.md`（K/U 矩阵）、`docs/alpha-runbook.md`、`docs/evidence/current.json` 的 `target_ubuntu_t16_alpha`。
- 当前活动进程：无；`docker ps -a --filter label=kernelagent.worker=1` 无残留。
- 阻塞与所需用户决定：请用户提供 provider 凭据并运行 `examples/alpha_run.py --mode live`（命令见 runbook）。
- 建议 Codex 审查重点：① ADR-0004 的 AST 门实现与"非安全边界"表述是否一致；
  ② `_replay_budget` 的 LIFO 预留语义（中断尝试保留预留、重复崩溃收敛 budget_exhausted）；
  ③ T16 卡片验收矩阵与报告 hash 是否与 `docs/evidence/current.json` 一致。

回执完成后，同时按 `AGENTS.md` 更新正式的 `docs/task-board.json`、`docs/handoffs/current.md` 和 `docs/evidence/current.json`；只有实际检查证据支持时才改变状态。
