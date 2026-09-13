import { useEffect, useRef, useState } from "react";

/**
 * 日志查看器（spec §1.3/§4）：等宽、分 tone 着色、智能跟随滚动 +
 * auto-scroll 开关（借鉴 UCAgent Task 页范式）。
 *
 * 智能跟随：默认跟随最新；用户上滚后停住保持位置；滚回底部自动恢复。
 */

export type LogTone = "plain" | "ok" | "warn" | "bad" | "run" | "dim" | "accent";

export interface LogLine {
  text: string;
  tone?: LogTone;
}

const TONE_CLASS: Record<LogTone, string> = {
  plain: "text-fg/90",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  run: "text-run",
  dim: "text-text-3",
  accent: "text-accent",
};

export interface LogViewerProps {
  lines: LogLine[];
  /** 标题右侧的附加节点（如数据来源徽章） */
  toolbar?: React.ReactNode;
  /** 空态文案 */
  emptyText?: string;
  /** auto-scroll 默认值（来自本地偏好） */
  autoscrollDefault?: boolean;
  /** 自动滚动偏好变化时回调（用于持久化） */
  onAutoscrollChange?: (value: boolean) => void;
  maxHeight?: string;
}

export function LogViewer({
  lines,
  toolbar,
  emptyText = "暂无日志。",
  autoscrollDefault = true,
  onAutoscrollChange,
  maxHeight = "480px",
}: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(autoscrollDefault);
  const [autoscroll, setAutoscroll] = useState(autoscrollDefault);
  const pinnedToBottom = useRef(true);

  // 数据更新时：跟随模式且开关打开 -> 滚到底
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (autoscroll && follow && pinnedToBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines, autoscroll, follow]);

  function handleScroll() {
    const el = containerRef.current;
    if (!el) return;
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    pinnedToBottom.current = gap < 24;
    if (gap >= 24 && follow) setFollow(false);
    else if (gap < 24 && !follow) setFollow(true);
  }

  function toggleAutoscroll(next: boolean) {
    setAutoscroll(next);
    onAutoscrollChange?.(next);
    if (next) {
      pinnedToBottom.current = true;
      setFollow(true);
    }
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-log-bg">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted">日志流</span>
          {!follow && (
            <span className="rounded-full border border-warning/25 bg-warning-soft px-2 py-0.5 text-[10px] text-warn">
              已暂停跟随（滚回底部恢复）
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {toolbar}
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted">
            <input
              type="checkbox"
              checked={autoscroll}
              onChange={(e) => toggleAutoscroll(e.target.checked)}
              className="h-3.5 w-3.5 accent-[var(--color-accent)]"
            />
            自动滚动
          </label>
        </div>
      </div>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{ maxHeight }}
        className="overflow-auto px-3 py-2 font-mono text-[12.5px] leading-relaxed"
      >
        {lines.length === 0 ? (
          <p className="py-6 text-center text-xs text-text-3">{emptyText}</p>
        ) : (
          <ol className="space-y-px">
            {lines.map((line, i) => (
              <li key={i} className={`whitespace-pre-wrap break-all ${TONE_CLASS[line.tone ?? "plain"]}`}>
                {line.text}
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
