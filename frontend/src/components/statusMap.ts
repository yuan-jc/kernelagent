import type { RunState } from "../api";
import type { BadgeVariant } from "./ui/Badge";

/**
 * 全部枚举 → {中文标签, 徽章色} 的唯一出处（spec §4.2 / §2.2）。
 * 颜色语义对齐设计规范，不得在组件内散落映射。
 */

export interface StatusMeta {
  variant: BadgeVariant;
  label: string;
  /** running 态徽章带呼吸圆点 */
  dot: boolean;
}

// -- Run 状态（server.py run_snapshot 枚举；映射对齐 spec §2.2 表）-----------

export const RUN_STATE_META: Record<RunState, StatusMeta> = {
  running: { variant: "running", label: "运行中", dot: true },
  completed: { variant: "success", label: "完成（已晋升）", dot: false },
  no_improvement: { variant: "neutral", label: "无改进", dot: false },
  budget_exhausted: { variant: "warning", label: "预算耗尽", dot: false },
  infra_error: { variant: "error", label: "基础设施错误", dot: false },
  journal_corrupt: { variant: "error", label: "journal 损坏", dot: false },
  unknown: { variant: "neutral", label: "未知", dot: false },
};

export function runStateMeta(state: RunState): StatusMeta {
  return RUN_STATE_META[state] ?? RUN_STATE_META.unknown;
}

// -- 候选状态（report.candidates[].status；spec §2.2）------------------------

export function candidateStatusMeta(status: string | undefined): StatusMeta {
  switch (status) {
    case "promoted":
      return { variant: "success", label: "promoted", dot: false };
    case "retained":
      return { variant: "neutral", label: "retained", dot: false };
    case "rejected":
      return { variant: "warning", label: "rejected", dot: false };
    case "failed":
      return { variant: "error", label: "failed", dot: false };
    case "measured":
      // timing 记录的终态（baseline-eager / 未晋升候选）：已测量，非运行中
      return { variant: "success", label: "measured（已测）", dot: false };
    default:
      return { variant: "neutral", label: status ?? "unknown", dot: false };
  }
}

// -- 阶段（progress stage 枚举；spec §4.1 六阶段，一期仅用于展示）------------

export const STAGES = [
  { key: "generate", label: "生成" },
  { key: "policy", label: "策略门" },
  { key: "evaluate", label: "容器评测" },
  { key: "correctness_pro", label: "增强正确性" },
  { key: "timing", label: "正式计时" },
  { key: "confirm", label: "晋升确认" },
] as const;

/** record 中 provider 失败标 stage:"generation"，归并到 generate 展示 */
export function normalizeStage(stage: string | undefined): string {
  return stage === "generation" ? "generate" : (stage ?? "");
}
