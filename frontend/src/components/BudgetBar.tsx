import type { RunBudget } from "../api";
import { formatCount, formatDuration } from "../lib/format";

/**
 * 预算双层条（spec §2.5/§3.4）：settled 实色 + reserved 叠加半透明。
 * 缺失 limit 时显示"无上限记录"，不画 0。
 */

interface BudgetBarProps {
  label: string;
  unit: "gpu" | "token";
  budget: RunBudget | null;
}

function fmt(unit: "gpu" | "token", v: number): string {
  return unit === "gpu" ? formatDuration(v) : formatCount(v);
}

export function BudgetBar({ label, unit, budget }: BudgetBarProps) {
  const limit = unit === "gpu" ? budget?.gpu_seconds_limit : budget?.tokens_limit;
  const settled = unit === "gpu" ? budget?.settled_gpu_seconds : budget?.settled_tokens;
  const reserved = unit === "gpu" ? budget?.reserved_gpu_seconds : budget?.reserved_tokens;

  const hasLimit = typeof limit === "number" && limit > 0;
  const settledValue = typeof settled === "number" && settled >= 0 ? settled : null;
  const reservedValue = typeof reserved === "number" && reserved > 0 ? reserved : 0;

  const settledPct = hasLimit && settledValue !== null ? Math.min((settledValue / limit) * 100, 100) : 0;
  const reservedPct = hasLimit ? Math.min((reservedValue / limit) * 100, 100 - settledPct) : 0;
  const nearLimit = hasLimit && settledPct + reservedPct >= 80;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="text-muted">{label}</span>
        <span className="font-mono tabular-nums text-fg">
          {settledValue === null ? "—" : fmt(unit, settledValue)}
          {reservedValue > 0 ? ` +${fmt(unit, reservedValue)} 预留` : ""}
          {hasLimit ? ` / ${fmt(unit, limit)}` : " / 无上限记录"}
        </span>
      </div>
      {settledValue === null && !hasLimit ? (
        <div className="flex h-2 w-full items-center rounded-full bg-surface-raised">
          <span className="px-2 text-[10px] text-text-3">预算账本尚未报告（NOT_RUN，不是 0）</span>
        </div>
      ) : (
        <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-raised">
          <div
            className={`h-full ${nearLimit ? "bg-warning" : "bg-primary"}`}
            style={{ width: `${settledPct}%` }}
          />
          {reservedPct > 0 && (
            <div
              className="h-full bg-primary/40"
              style={{ width: `${reservedPct}%` }}
              title="预留（未结算）"
            />
          )}
        </div>
      )}
      {settledValue !== null && reservedValue > 0 && (
        <p className="mt-1 text-[11px] text-muted">
          另有 {fmt(unit, reservedValue)} 在途预留（reserved，未结算）
        </p>
      )}
    </div>
  );
}
