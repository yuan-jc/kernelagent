import type { ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";

/** Run 详情的 tab 条；tab 状态同步到 ?tab= query（spec §8.3） */
export interface TabDef {
  key: string;
  label: string;
  /** 徽章内容（如 NOT_RUN / 数量） */
  badge?: ReactNode;
}

export function TabBar({ tabs, active, onChange }: { tabs: TabDef[]; active: string; onChange: (key: string) => void }) {
  return (
    <div className="flex gap-1 overflow-x-auto border-b border-border" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          role="tab"
          type="button"
          aria-selected={active === tab.key}
          onClick={() => onChange(tab.key)}
          className={`-mb-px flex items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-2 text-sm transition-colors ${
            active === tab.key
              ? "border-accent font-medium text-fg"
              : "border-transparent text-muted hover:text-fg"
          }`}
        >
          {tab.label}
          {tab.badge}
        </button>
      ))}
    </div>
  );
}

/** 便捷 hook：?tab= 与 setState 同步 */
export function useTabParam(defaultTab: string, valid: string[]): [string, (key: string) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab") ?? defaultTab;
  const active = valid.includes(raw) ? raw : defaultTab;
  const setTab = (key: string) => {
    const next = new URLSearchParams(params);
    next.set("tab", key);
    setParams(next, { replace: true });
  };
  return [active, setTab];
}

/** Profile 页返回详情的链接样式统一 */
export function BackToRunLink({ runId }: { runId: string }) {
  return (
    <Link to={`/runs/${runId}`} className="text-xs text-muted hover:text-fg">
      ← 返回 Run 详情
    </Link>
  );
}
