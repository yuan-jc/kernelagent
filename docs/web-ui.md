# Web 控制台

本地网页控制台（`python -m kernelagent.webapp`），为零依赖标准库实现 + 单页前端，
参考 UCAgent 的"配置面板 + 任务进度"形态。

## 启动

```bash
cd /home/y/kernelagent
.venv/bin/python -m kernelagent.webapp --port 8501
# 浏览器打开 http://127.0.0.1:8501
```

默认只绑定 `127.0.0.1`；不要把端口暴露到局域网/公网（API Key 会经浏览器提交到本机服务端）。

## 功能

1. **API 填写**：模式选"真实模型生成"时填 Provider Base URL 与 API Key。
   Key 只在本机内存中存在一个 run 的时长，不写 job.json/report/journal/日志（有测试钉住）。
   未填 Key 时回退读控制端环境变量 `MODEL_PROVIDER_API_KEY`，两者都没有则该 run 以配置错误终止。
2. **Bench 选择**：KernelBench 按 Level → Problem 两级下拉（从冻结快照枚举，problem 值即
   `kernelbench:l<level>:<id>` 规约）。TritonBench / FlashInfer-Bench 适配器未接入 `optimize`，如实置灰。
3. **阶段展示**：当前候选的阶段流水（生成 → 策略门 → 容器评测 → 增强正确性 → 正式计时 → 晋升确认）。
   进行中的候选由 `.progress.json` 心跳标注阶段；已完成候选的记录（含拒绝原因、CI）来自 durable records。
4. **加速比展示**：每次 confirm 记录 batch 级 bootstrap CI（S = mean(incumbent)/mean(candidate)），
   界面按候选展示 `低×–高×` 区间条与文字；Champion 卡展示 sha256/路径/CI。
5. **预算与历史**：GPU 秒 / Token 两条预算进度条（settled/limit，来自 journal 重放）；运行历史表可点选回看。
6. **单 GPU 串行**：同时只允许一个活动 run，第二个启动请求返回 409。

## 诚实声明

- "演示 · 正确/错误固定候选"两个模式用于无凭据时证明闭环（同 Task 6 的 U1/U2），
  **不是** LIVE_MODEL 验收，不得以它们冒充真实生成。
- 界面展示的全部状态来自 run 目录的 durable 事实（journal/records/report），服务端不虚构进度。
- Alpha 威胁模型：合作型候选（ADR-0004），champion 晋升后需人工审查源码。
