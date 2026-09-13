# 恢复继续点（新会话从这里开始）

上一会话在 T20 执行中断于环境故障（会话沙箱命名空间损坏，`spawn /bin/bash ENOENT`）。
宿主文件系统完好——**无需重装 bash**，只需重启 ZCode/新建会话。

## 新会话第一条提示（复制粘贴）

> 按 docs/handoffs/recovery-runbook.md 继续：验证 git log 与 nvidia-smi → 执行 T20 节（下载解压
> CUTLASS v3.5.1 tarball 到 research/sources/cutlass → docker build -t kernelagent-eval:t20 configs/t20
> → 把镜像 ID 写入 configs/t20/image_id → pytest tests/test_t20.py → python examples/t20_smoke.py
> → 全量回归 → 验收提交）→ 按依赖推进 T11/T16/T17/T22 → T23 → T24/T25。
> T12/T11 的 LIVE_MODEL 需要用户提供 MODEL_PROVIDER_API_KEY 凭据后以 --live 重跑
> examples/generation_loop_smoke.py。

## 当前状态快照（2026-09-13）

- 本地 main `ad88246`，25 个提交未推送。
- ACCEPTED：T00、T01、T02、T03、T04、T04a、T05、T06、T07、T08、T09、T10、T12a、T13、T14、T15、T18、T19。
- READY_FOR_ACCEPTANCE：T12（生成→GPU 闭环已通，仅差 LIVE_MODEL 凭据）。
- 未验收：T11、T16、T17、T20（产物已预置待执行）、T22、T23、T24、T25。
- 全量回归基线：406 项 = 403 passed / 3 Windows-only skips / 0 failed。
- 恢复细节与 T20 精确命令：docs/handoffs/recovery-runbook.md。
