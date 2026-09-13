import type { RunSnapshot } from "../../api";
import { CopyButton } from "../../components/CopyButton";
import { Badge, Card, CardContent } from "../../components/ui";
import { useState } from "react";

/**
 * Run 级横幅（spec §4.3.1）：infra_error / journal_corrupt 专用。
 * 必须注明「基础设施错误 ≠ 候选失败」；error_tail 尾部 300 字直接可见，
 * 可展开全部（<=1500 字符，后端已截断）并可复制。
 */
export function RunErrorBanner({ snapshot }: { snapshot: RunSnapshot }) {
  const [expanded, setExpanded] = useState(false);
  const corrupt = snapshot.state === "journal_corrupt";
  const tail = snapshot.error_tail ?? "";

  return (
    <div
      role="alert"
      className="rounded-lg border border-error/40 bg-error-soft px-4 py-3 text-sm"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-medium text-error">
          {corrupt
            ? "journal 损坏：无法解析 run 目录中的 journal.jsonl，以下为服务端返回的原始错误状态。"
            : "基础设施错误（infra_error）：run 进程层面失败。基础设施错误 ≠ 候选失败，下方内容为 run 目录 error.txt 的尾部。"}
        </p>
        <div className="flex items-center gap-2">
          <Badge variant="error">{corrupt ? "journal_corrupt" : "infra_error"}</Badge>
          {tail && <CopyButton text={tail} label="复制错误尾部" />}
        </div>
      </div>
      {tail && (
        <>
          <pre className="mt-2 max-h-40 overflow-auto rounded-md border border-error/30 bg-bg/70 p-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted">
            {expanded ? tail : tail.slice(-300)}
          </pre>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="mt-1.5 text-xs text-primary hover:text-primary-hover"
          >
            {expanded ? "收起（仅显示尾部 300 字）" : "展开全部（服务端已截断至 1500 字符）"}
          </button>
        </>
      )}
    </div>
  );
}

/**
 * 终态解释卡（spec §4.3.3）：no_improvement / budget_exhausted 是
 * 合法结果，不是错误；completed 强调 cooperative 审查义务。
 */
export function TerminalNote({ snapshot }: { snapshot: RunSnapshot }) {
  if (snapshot.state === "no_improvement") {
    return (
      <Card className="border-muted-s/30 bg-neutral-soft">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium">无改进（合法结果）：</span>
          所有候选均未在确认阶段超过 baseline，baseline 保留为当前最优实现。
          这不是错误，也不代表候选全部失败 —— 各候选的具体原因见「候选对比」。
        </CardContent>
      </Card>
    );
  }
  if (snapshot.state === "budget_exhausted") {
    return (
      <Card className="border-warning/30 bg-warning-soft">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium">预算耗尽（合法结果）：</span>
          GPU 秒或 token 预算用尽而正常终止，未到终局确认。缺失的阶段指标一律标
          <span className="mx-1 inline-flex translate-y-[-1px] items-center rounded-full border border-muted-s/25 bg-neutral-soft px-1.5 py-0 font-mono text-[10px] text-muted-s">
            NOT_RUN
          </span>
          ，不按零处理；已结算部分见预算面板（journal 台账）。
        </CardContent>
      </Card>
    );
  }
  if (snapshot.state === "completed") {
    return (
      <Card className="border-champion/30 bg-warning-soft/60">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium text-champion">运行完成：</span>
          已产生 champion（见下）。candidate_trust ={" "}
          <span className="font-mono">{snapshot.candidate_trust ?? "未报告"}</span>{" "}
          —— champion 需人工审查后方可采信，本页不将完成渲染为“已验证可信”。
        </CardContent>
      </Card>
    );
  }
  return null;
}
