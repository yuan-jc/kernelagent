import type { Lane } from "../lib/lanes";
import { STAGES } from "./statusMap";
import { NotRunBadge } from "./ui/NotRunBadge";
import { Badge } from "./ui/Badge";
import { candidateStatusMeta } from "./statusMap";

/**
 * 多候选泳道图（spec §4.1.2）：每候选一行，行主体为六段水平条。
 * 已完成段 ok、进行中段 run+流光、失败段 bad、skipped 段 warn、
 * 未到达段暗底。行右侧关键指标：加速比 CI 或 NOT_RUN 徽章
 * （缺失指标不画 0）。当前活跃行置顶（由 buildLanes 排序保证）并高亮左边框。
 */

const SEGMENT_CLASS: Record<string, string> = {
  done: "bg-success/70",
  running: "bg-running/80 animate-stage-flow",
  failed: "bg-error/80",
  skipped: "bg-warning/70",
  pending: "bg-surface-3",
  unknown: "bg-muted-s/40",
};

function ratioText(ci: [number, number]): string {
  return `${ci[0].toFixed(2)}× – ${ci[1].toFixed(2)}×`;
}

export interface StageSwimlanesProps {
  lanes: Lane[];
  /** 点击失败泳道展开 detail */
  onLaneDetail?: (lane: Lane) => void;
}

export function StageSwimlanes({ lanes, onLaneDetail }: StageSwimlanesProps) {
  return (
    <div className="space-y-1.5">
      {lanes.map((lane) => {
        const meta = candidateStatusMeta(lane.status);
        const isRunning = lane.active || lane.stages.includes("running");
        return (
          <div
            key={lane.id}
            className={`flex flex-col gap-1.5 rounded-md border px-3 py-2 md:flex-row md:items-center ${
              lane.active
                ? "border-running/40 border-l-2 border-l-running bg-running-soft/40"
                : "border-border bg-surface-2/50"
            }`}
          >
            <div className="flex w-44 shrink-0 items-center gap-2 md:w-52">
              <span className="truncate font-mono text-xs text-fg" title={lane.id}>
                {lane.id}
              </span>
              {lane.status ? (
                <Badge variant={meta.variant} dot={isRunning && !lane.status}>
                  {isRunning && !lane.status ? "运行中" : meta.label}
                </Badge>
              ) : isRunning ? (
                <Badge variant="running" dot>
                  运行中
                </Badge>
              ) : (
                // 无 status 且无运行依据（如终态 run 里 journal 有轨迹但无
                // final 记录的动作）：诚实 NOT_RUN，不猜成败，也不留空
                <NotRunBadge what={`${lane.id} 的 final 记录`} />
              )}
            </div>
            <div className="flex min-w-0 flex-1 items-center gap-2">
              <div className="flex h-3.5 min-w-0 flex-1 gap-px overflow-hidden rounded-sm">
                {lane.stages.map((state, i) => (
                  <div
                    key={STAGES[i]!.key}
                    title={`${STAGES[i]!.label}: ${state}`}
                    className={`h-full flex-1 ${SEGMENT_CLASS[state] ?? SEGMENT_CLASS.pending}`}
                  />
                ))}
              </div>
              <div className="w-40 shrink-0 text-right md:w-56">
                {lane.ratioCi ? (
                  <span
                    className="font-mono text-xs tabular-nums text-ok"
                    title="confirm 阶段速度比 95% 置信区间"
                  >
                    {ratioText(lane.ratioCi)}
                  </span>
                ) : lane.detail ? (
                  <button
                    type="button"
                    onClick={() => onLaneDetail?.(lane)}
                    className="max-w-full truncate font-mono text-[11px] text-error hover:underline"
                    title="点击展开完整 detail"
                  >
                    {lane.detail}
                  </button>
                ) : (
                  <NotRunBadge what="速度比 95% CI" />
                )}
              </div>
            </div>
          </div>
        );
      })}
      <div className="flex justify-end gap-px pr-44 md:pr-56" aria-hidden="true">
        {STAGES.map((s) => (
          <span key={s.key} className="w-1/6 min-w-[52px] text-center font-mono text-[10px] text-text-3">
            {s.key}
          </span>
        ))}
      </div>
    </div>
  );
}
