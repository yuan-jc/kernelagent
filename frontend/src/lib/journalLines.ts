import type { JournalEntry } from "../api";
import type { LogLine } from "../components/LogViewer";
import { shortHash } from "./format";

/**
 * journal 事件 -> 日志行（tone 语义对齐 statusMap）。
 * journal 无时间戳字段（项目事实）：行首只给行号/seq，不显示时间。
 */

const KIND_TONE: Record<string, LogLine["tone"]> = {
  budget_reserved: "warn",
  budget_settled: "warn",
  action_started: "run",
  action_finished: "ok",
  action_interrupted: "bad",
  experiment_done: "accent",
  report_ready: "ok",
  stage_started: "run",
  stage_finished: "ok",
};

export function journalToLines(entries: JournalEntry[]): LogLine[] {
  return entries.map((e, i) => {
    const seq = typeof e.seq === "number" ? e.seq : i + 1;
    const parts: string[] = [`#${String(seq).padStart(3, "0")}`];
    parts.push(e.kind);
    if (e.action_id) parts.push(e.action_id);
    if (e.stage) parts.push(`stage=${e.stage}`);
    if (typeof e.gpu_seconds === "number") parts.push(`gpu=${e.gpu_seconds}s`);
    if (typeof e.tokens === "number" && e.tokens > 0) parts.push(`tokens=${e.tokens}`);
    if (typeof e.settled_gpu_seconds === "number") parts.push(`settled_gpu=${e.settled_gpu_seconds}s`);
    if (typeof e.settled_tokens === "number") parts.push(`settled_tokens=${e.settled_tokens}`);
    if (e.result_ref) parts.push(`result=${e.result_ref}`);
    if (e.input_hash) parts.push(`in=${shortHash(e.input_hash)}`);
    if (e.entry_hash) parts.push(`hash=${shortHash(e.entry_hash)}`);
    return { text: parts.join("  "), tone: KIND_TONE[e.kind] ?? "plain" };
  });
}

/** workspace 文本 -> 日志行（原样，不解释） */
export function textToLines(text: string): LogLine[] {
  return text.split("\n").map((line) => ({ text: line, tone: "plain" as const }));
}
