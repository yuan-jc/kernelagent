/**
 * 缺失指标统一呈现：NOT_RUN / "—"（项目不变量：缺失指标不是零，绝不画 0）。
 */
export function NotRunBadge({ what }: { what?: string }) {
  return (
    <span
      title={what ? `${what ?? "指标"}未运行/未报告，不是 0` : "未运行/未报告，不是 0"}
      className="inline-flex items-center rounded-full border border-muted-s/25 bg-neutral-soft px-2 py-0.5 font-mono text-[11px] text-muted-s"
    >
      NOT_RUN
    </span>
  );
}
