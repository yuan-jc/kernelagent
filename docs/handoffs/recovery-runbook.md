# 恢复手册：/bin/bash 丢失后的继续点（2026-09-13，更新：根因已定位）

## 事故与根因（已定位）

会话执行层报 `spawn /bin/bash ENOENT`（Bash 工具、沙箱绕过、子代理三条路径均失败）。

经 Read/Write 工具在真实文件系统上探测确认：

- `/bin/bash`（1.7MB ELF）、`/usr/bin/bash`、`/lib64/ld-linux-x86-64.so.2`、`/bin/dash`、`/bin/ls` **全部存在且完好**；
- 向 `/bin` 写探针文件被权限拒绝（EACCES，工具以普通用户身份访问真实 FS）。

**结论：宿主操作系统没有损坏，不需要 `apt install --reinstall bash`。** ENOENT 出在 Bash 工具自身的沙箱挂载命名空间（会话创建时 provision，其根内丢失了 `/bin/bash`）。该命名空间对本会话是固定的——本会话内无法自愈，子代理继承同一命名空间。

## 正确的恢复动作

**重启/新建 ZCode 会话（在 /home/y/kernelagent 下）**——新会话会重新 provision 沙箱，`/bin/bash` 即恢复。不要重装宿主 bash（无害但无关）。

## 恢复后验证（按序）

```bash
echo ok && git -C /home/y/kernelagent log --oneline -3 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
docker info >/dev/null && echo docker-ok
ls /tmp/cutlass-3.5.1.tar.gz /tmp/t18out/t15_two_kernels.ncu-rep 2>/dev/null
```

## 继续点：T20（进行中）

0. 恢复 shell 的第一件事：`echo ok` 确认 Bash 工具可用（新会话沙箱自动恢复；本会话内无法修复）。
1. 重新下载 CUTLASS（上一动作被环境故障打断）并解压到 research/sources/cutlass：
   ```bash
   cd /tmp && HTTPS_PROXY=http://127.0.0.1:7890 curl -fL --retry 3 -o cutlass-3.5.1.tar.gz https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.5.1.tar.gz
   sha256sum cutlass-3.5.1.tar.gz   # 记录
   tar -xzf cutlass-3.5.1.tar.gz -C /home/y/kernelagent/research/sources/ --strip-components=1 --one-top-level=cutlass --overwrite
   ls /home/y/kernelagent/research/sources/cutlass/include/cutlass/cutlass.h
   ```
2. 构建 T20 镜像层并记录 ID：
   ```bash
   docker build -t kernelagent-eval:t20 configs/t20
   docker inspect kernelagent-eval:t20 --format '{{.Id}}' > configs/t20/image_id
   ```
3. 验收运行：
   ```bash
   uv run --locked pytest tests/test_t20.py -q
   uv run --locked python examples/t20_smoke.py --output artifacts/t20
   ```
4. 验收后：T20 卡片/任务板/证据索引/交接更新 + git commit。

T20 预置产物（已写好，勿重写）：`configs/t20/{Dockerfile,cutlass_gemm.cu,t20_driver.py}`、`examples/t20_smoke.py`、`tests/test_t20.py`、`docs/work-packages/T20.md`（冻结矩阵）。

## 其余未完成包（依赖与计划）

- T11：外部 KernelAgent 适配器——上游生成本身需要 LLM 凭据（与 T12 同一阻塞）。
- T16/T17/T22：依赖 T12 的 ACCEPTED 状态（T16 ← T12；T17 ← T16；T22 ← T16）。
- T23：整体对抗回归门（依赖 T06–T22 全 ACCEPTED）。
- T24/T25：正式实验冻结/运行与分析（依赖 T23；搜索生成依赖 T12 LIVE_MODEL）。

## 关键路径解锁点（用户动作）

- `MODEL_PROVIDER_API_KEY`（OpenAI 兼容 base_url + key）→ 解锁 T12 G5 → 级联解锁 T13(已功能性验收)/T16/T17/T22 → T23 → T24/T25。
- ncu 非 root 计数器（`/etc/modprobe.d/nvidia-profiling.conf` + 重启）→ 收窄 ADR-0003 特权诊断租约（可选，非阻塞）。
