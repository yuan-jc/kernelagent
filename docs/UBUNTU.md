# Ubuntu 开发上手

首用目标：Ubuntu 24.04 LTS、x86_64、NVIDIA GPU。以下 CPU 开发步骤不依赖显卡。GPU/驱动/工具链的实际版本必须在目标机器记录，不能沿用旧 Windows 报告。

## 1. 克隆与安装

```bash
sudo apt update
sudo apt install -y git python3-venv
git clone https://github.com/yuan-jc/kernelagent.git
cd kernelagent
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv==0.12.13
export PATH="$PWD/.bootstrap/bin:$PATH"
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked kernelagent --version
```

重新打开终端后先 `cd kernelagent`，再执行上述 `export PATH=...`；也可直接使用 `.bootstrap/bin/uv`。不要复制 Windows 的 `.venv`。Python 3.11 是推荐开发基线，项目允许 3.11–3.14；CI 验证 3.11/3.13。`uv.lock` 已提交，日常使用 `--locked`，不在迁移时随意升级依赖。

## 2. CPU 自检与最小示例

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked kernelagent check --output artifacts/ubuntu-cpu
uv run --locked python examples/worker_smoke.py
uv build --no-sources
```

预期：检查退出 0，CPU 报告 PASS，worker 示例输出 `status=completed`。Linux 上仅 Windows Job 专用用例可按平台跳过，报告必须列出跳过数与原因；不能以全 skip 通过。任一步失败先记录并修复，不推进验收状态。

示例只执行固定的 `print` 载荷，不加载候选 kernel。当前 POSIX 进程组不能阻止后代 `setsid()` 逃逸；这是 T04 容器/cgroup 必须覆盖的边界。

## 3. 恢复冻结的 KernelBench 输入

```bash
uv run --locked python research/fetch_kernelbench_problems.py
uv run --locked kernelagent bench verify
```

预期：下载器验证 273 个文件（270 个问题 + 3 个去重后的协议/数据模块引用），bench verify 为 PASS、270/270 匹配，开发子集 L1/L2/L3 为 18/8/9。缓存不随 Git 上传，首次下载需要访问 raw.githubusercontent.com。

脚本读取 `configs/kernelbench/` 的版本与 hash，不生成新清单；已有正确文件不会重下，错误下载不会覆盖缓存。可使用标准 `HTTPS_PROXY` 或显式 `--proxy <URL>`；仓库不预设任何个人代理。重试同一命令即可恢复下载，不需要重做调研。

这些文件足以做静态适配与校验，**不是可运行的完整 KernelBench 安装**。T05 再按固定 commit 获取完整上游并锁定 GPU 环境，不能因快照下载成功就宣布上游评测可运行。

## 4. 目标 NVIDIA 环境与 T03

先确认目标系统驱动能运行 `nvidia-smi`。已有正常驱动时不要盲目重装；驱动/CUDA 兼容性参照 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。CUDA Toolkit 的 nvcc 与 Nsight Compute 的 ncu 是不同检查；具体版本在目标环境冻结。

```bash
mkdir -p artifacts/ubuntu
git rev-parse HEAD > artifacts/ubuntu/commit.txt
git diff > artifacts/ubuntu/worktree.patch
uname -a > artifacts/ubuntu/os.txt
nvidia-smi > artifacts/ubuntu/nvidia-smi.txt
uv run --locked kernelagent probe --target native --output artifacts/ubuntu/probe --evidence-root artifacts/ubuntu/evidence
```

按 [T03 卡片](work-packages/T03.md)核对报告：schema 有效、设备身份可查、CUDA 驱动与 kernel 启动通过、真实写读 42、证据可校验。仅 exit 0 不够，工具缺失可能为 `unavailable`。nvcc/ncu 缺失或权限不足应记录为具体缺口；ncu 能打印版本不证明真实 profiling 可用，后者属于 T15。

目标 Ubuntu GPU 尚未验证时，T03 保持 READY_FOR_ACCEPTANCE / GPU NOT_RUN。旧 Windows/WSL 的成功仅说明当时探针路径成功，不替代这台机器的验收。

## 5. 接下来写什么

1. 完成 T03，提交目标报告的身份、hash、检查状态与可访问证据。
2. 按 [T04](work-packages/T04.md)复用容器与 NVIDIA Container Toolkit，落实可信 evaluator 与候选分离、禁网、只读可信输入、限额及进程回收。
3. T04 验收后做 T05：少量不同类别的 KernelBench 原生基线与 adapter 对照；随后 T06 正确性、T07 计时。
4. 再按任务计划接晋升、预算恢复、真实模型生成和搜索。不能先做自动循环再补可信计时。

当前仓库没有可验收的 GPU 容器镜像、生产 worker 服务或真实模型 API 客户端；控制端虚拟环境保持轻量。T04/T05 建立并锁定独立 GPU 环境后才安装 torch/Triton 等依赖，凭据仅留在控制端。

每次开工只读 AGENTS、当前交接、当前工作包及相关设计章节。不要重读全部历史，也不要把未通过的包升级为 ACCEPTED。每包提交应绑定当前 CI，原始运行文件留在 `artifacts/` 或 CI artifacts，Git 中保留简洁索引。
