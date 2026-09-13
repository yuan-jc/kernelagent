# ADR-0004：候选代码与评测器的信任模型（Alpha 过渡态）

- 状态：Accepted（2026-09-13）
- 影响包：T05/T06/T07 的威胁模型声明；T12 生成门；T16 主循环；T23 可信 MVP

## 背景

pinned 上游 evaluator 在容器内用 `exec()` 于评测器同一 Python 进程中执行候选源码
（`eval_kernel_against_ref` 的既定行为，ADR-0002 锁定不改）。Docker 边界（ADR-0001）
保护宿主，不保护同进程的 evaluator：候选可以改写 `torch.allclose`、`torch.cuda.Event`、
`json.dump`、随机数 API，或写父端所有的 `/out` 通道。已在真实 GPU 链上复现：返回全零但
把 `torch.allclose` 改为恒真的候选被判 `compiled=True/correctness=True/adapter_pass=True`。
因此 T05/T06/T07 的既有验收在对抗候选威胁模型下不可称为可信。

## 决定

1. **Alpha 威胁模型显式声明为"合作型候选"**：每份评测报告携带
   `candidate_trust="cooperative"`、`adversarially_secure=false`；champion 晋升后必须
   人工审查源码。报告不说谎：缺失的强度不以隐式方式声称。
2. **AST 预检测门 `inspect_candidate_policy`**（`adapters/models/generation.py`）：
   对候选源码做静态 misuse 检测——外部模块属性赋值、`/out` 访问、`eval/exec/compile`
   与命名空间重接线、受限模块导入（os/subprocess/socket/ctypes/importlib/sys/…）、
   `setattr` 打补丁、dunder 全局访问——返回结构化 violations。T16 主循环在花费 GPU
   之前以该门拒绝候选；评测报告原样记录 `policy_violations`。
3. **AST 门不是安全边界**：`exec()` 的能力面大于 AST 可见面（`getattr` 计算名、
   `__import__`、模块别名等均可绕过静态检查）。该门只是减少意外篡改和天真攻击的防线；
   判定强度的根本改进是第 7 节（launch plan）的信任域分离（候选容器只见输入不见参考、
   最终 verdict 由不加载候选模块的可信 verifier 产生），落在 T23。

## 后果

- T05/T06/T07 已有验收结论保持有效，但必须在证据/文档中带 `candidate_trust=cooperative`
  限定语；禁止在对抗威胁模型下引用它们作为"不可伪造"的证据。
- 候选因 policy 门被拒是合法、可记录的结果（计入分母），不是基础设施失败。
- 可信 MVP（T23）落地后，本 ADR 的第 1、2 条降级为纵深防御的一层，威胁模型升级为
  对抗型并重新跑 T05–T10/T12/T16/T21 的 GPU 验收。
