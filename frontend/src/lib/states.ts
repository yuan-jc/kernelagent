import type { RunState } from "../api";

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
