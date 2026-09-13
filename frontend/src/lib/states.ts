import type { RunReport, RunSnapshot, RunState } from "../api";

/**
 * 终态判定（spec §5）：后端 TERMINAL_STATES + UI 补充的
 * journal_corrupt / unknown；终态后停止轮询。
 * 前端不得自行推断"成功"，仅用于停止刷新。
 */
export const TERMINAL_RUN_STATES: readonly RunState[] = [
  "completed",
  "no_improvement",
  "budget_exhausted",
  "infra_error",
  "journal_corrupt",
  "unknown",
];

export function isTerminalRunState(state: RunState): boolean {
  return (TERMINAL_RUN_STATES as readonly string[]).includes(state);
}

/**
 * 轮询是否应继续。注意 "unknown" 是服务端的"暂不可知"（如 job.json 已落盘、
 * journal.jsonl/report.json 尚未生成，或读取与 job 线程收尾赛跑），不是终局：
 * 若按终态停轮询，Run 详情会永远停在「未知 + 预算 NOT_RUN」。因此 unknown
 * 必须继续轮询，让终态到达后的下一拍（≤1 个轮询间隔）必然收敛。
 * journal_corrupt 不会自行恢复，仍视为终局。
 */
export function shouldKeepPolling(state: RunState): boolean {
  return state === "unknown" || !isTerminalRunState(state);
}

/** 服务端快照枚举中"真正的终局"（unknown/journal_corrupt 之外的四个） */
const SETTLED_SERVER_STATES: readonly string[] = [
  "completed",
  "no_improvement",
  "budget_exhausted",
  "infra_error",
];

/**
 * 快照 state=unknown 时的 report 兜底合并（修复终态收敛）：
 * 服务端 run_snapshot 在 job 线程收尾/启动窗口可能返回 state="unknown" 且
 * budget=null，而 report.json 已（或即将）落盘。此时以 report.json 这一
 * 同源服务端产物补齐终态显示；快照自身的字段永远优先（report 只兜空缺）。
 * 仅接受 report 携带的服务端终局状态，绝不把 report.state=running/unknown
 * 升级成任何"看起来更好"的状态。
 */
export function mergeRunWithReport(snapshot: RunSnapshot, report: RunReport | null): RunSnapshot {
  if (snapshot.state !== "unknown" || !report) return snapshot;
  const reportState = report.state;
  const state =
    typeof reportState === "string" && (SETTLED_SERVER_STATES as readonly string[]).includes(reportState)
      ? reportState
      : snapshot.state;
  return {
    ...snapshot,
    state,
    budget: snapshot.budget ?? report.budget ?? null,
    champion: snapshot.champion ?? report.champion ?? null,
    candidates:
      (snapshot.candidates?.length ?? 0) > 0 ? snapshot.candidates : (report.candidates ?? []),
    candidate_trust: snapshot.candidate_trust ?? report.candidate_trust,
    adversarially_secure:
      snapshot.adversarially_secure ?? report.adversarially_secure,
  };
}
