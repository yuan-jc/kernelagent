# ADR-0003：T15 NCU 诊断取证租约（提权容器）

- 状态：Accepted（2026-09-13）
- 影响包：T15；T17 的 NSYS/SASS 取证可复用同一租约

## 背景

T15 需要真实 `.ncu-rep` 证据。宿主非 root 与默认容器（root 且无 SYS_ADMIN）均被 NVreg 计数器权限拒绝（`ERR_NVGPUCTRPERM`，本机实测两次）；官方解锁路径（`/etc/modprobe.d/nvidia-profiling.conf` + 重启）需要用户 sudo 与重启，当前不可用。

## 决定

1. **诊断租约**：仅用于父端自己编译的固定负载（如 `examples/ncu_evidence_smoke.py` 的 two_kernels 二进制），以 `--privileged` 容器运行 NCU（工具链只读挂载，输出目录可写，禁网，GPU 按 CDI UUID）。
2. **范围铁律**：不可信候选代码绝不进入该租约——候选的评测/计时仍走 ADR-0001 非特权边界；本租约只对"父端自己写的代码"取证。
3. **解释半区**：`ncu --import --csv --page raw` 在宿主以普通用户运行（导入无需 GPU 与提权，已实测）；解析结果带 `source="ncu_profile"` 标签，与 T07 正式计时严格分离。
4. **诚实记录**：权限被拒时状态为 `gpu_counter_permission_denied`（实测存在），绝不静默吞掉；用户完成 modprobe 配置后，可降级为非特权路径并重跑。

## 后果

- `--privileged` 仅作用于诊断租约（固定二进制、无凭据、无网络、分钟级），攻击面可控且与候选隔离边界无交集。
- 用户日后开启计数器权限即可收窄此租约，验收证据需随之重跑。
