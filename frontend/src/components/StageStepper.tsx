import { STAGES } from "./statusMap";
import type { StageNodeState } from "../lib/lanes";

/**
 * 单候选六阶段横向 stepper（spec §4.1.1）。
 *
 * 节点五态：done(running→ok) / running(run+呼吸) / failed(bad+✕) /
 * skipped(warn+skip) / pending(空心) / unknown(灰+?)。
 * 阶段耗时无时间戳依据时显示 "—"，不得编造（spec §4.1.1 一期约束）。
 */

const NODE_STYLE: Record<StageNodeState, { ring: string; fill: string; glyph: string; glyphColor: string }> = {
  done: {
    ring: "border-success/60",
    fill: "bg-success-soft",
    glyph: "✓",
    glyphColor: "text-success",
  },
  running: {
    ring: "border-running/70",
    fill: "bg-running-soft",
    glyph: "●",
    glyphColor: "text-running animate-pulse-dot",
  },
  failed: {
    ring: "border-error/70",
    fill: "bg-error-soft",
    glyph: "✕",
    glyphColor: "text-error",
  },
  skipped: {
    ring: "border-warning/60",
    fill: "bg-warning-soft",
    glyph: "skip",
    glyphColor: "text-warning",
  },
  pending: {
    ring: "border-border",
    fill: "bg-surface-2",
    glyph: "",
    glyphColor: "text-text-3",
  },
  unknown: {
    ring: "border-border-strong",
    fill: "bg-neutral-soft",
    glyph: "?",
    glyphColor: "text-muted-s",
  },
};

export interface StageStepperProps {
  stages: StageNodeState[];
  /** 每阶段耗时（毫秒）；缺省（无时间戳依据）一律显示 "—" */
  durationsMs?: Array<number | null>;
  /** 点击失败节点时的回调（展开错误） */
  onFailedClick?: (stageIndex: number) => void;
  className?: string;
}

export function StageStepper({
  stages,
  durationsMs,
  onFailedClick,
  className = "",
}: StageStepperProps) {
  return (
    <div className={`flex items-stretch gap-0 overflow-x-auto ${className}`}>
      {STAGES.map((stage, i) => {
        const state = stages[i] ?? "pending";
        const style = NODE_STYLE[state];
        const duration = durationsMs?.[i];
        const clickable = state === "failed" && onFailedClick !== undefined;
        return (
          <div key={stage.key} className="flex min-w-[88px] flex-1 items-center">
            <div className="flex flex-col items-center gap-1">
              <button
                type="button"
                disabled={!clickable}
                onClick={clickable ? () => onFailedClick?.(i) : undefined}
                title={`${stage.key} · ${stage.label}${
                  state === "failed" ? "（点击查看详情）" : ""
                }`}
                className={`flex h-8 w-8 items-center justify-center rounded-full border text-xs font-medium transition-colors ${style.ring} ${style.fill} ${style.glyphColor} ${
                  clickable ? "cursor-pointer hover:brightness-125" : "cursor-default"
                }`}
              >
                <span className={state === "skipped" ? "text-[9px] font-semibold" : ""}>
                  {style.glyph || ""}
                </span>
              </button>
              <div className="text-center leading-tight">
                <p
                  className={`text-[11px] font-medium ${
                    state === "pending" ? "text-text-3" : "text-fg"
                  }`}
                >
                  {stage.label}
                </p>
                <p className="font-mono text-[10px] text-text-3" title="单阶段精确耗时需要 journal 时间戳（P3 二期），缺失时不编造">
                  {typeof duration === "number" && Number.isFinite(duration)
                    ? `${(duration / 1000).toFixed(1)}s`
                    : "—"}
                </p>
              </div>
            </div>
            {i < STAGES.length - 1 && (
              <div
                aria-hidden="true"
                className={`mx-1 mb-5 h-px flex-1 ${
                  (stages[i] ?? "pending") === "done" ? "bg-success/50" : "bg-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
