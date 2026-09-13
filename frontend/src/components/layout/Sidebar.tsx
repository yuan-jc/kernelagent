import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import {
  IconChip,
  IconDashboard,
  IconGauge,
  IconRuns,
  IconSliders,
} from "../icons";

interface NavItem {
  to: string;
  label: string;
  end?: boolean;
  icon: (props: { className?: string }) => ReactNode;
}

/** 侧栏导航（spec §3.1：总览 / Runs / 新建运行 / Benchmarks / 设置） */
const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "总览", end: true, icon: IconDashboard },
  { to: "/runs", label: "Runs", icon: IconRuns },
  { to: "/launch", label: "新建运行", icon: IconChip },
  { to: "/benchmarks", label: "Benchmarks", icon: IconGauge },
  { to: "/settings", label: "设置", icon: IconSliders },
];

function NavList({ items }: { items: NavItem[] }) {
  return (
    <nav className="flex flex-col gap-1">
      {items.map((item) => {
        const ItemIcon = item.icon;
        return (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-accent-subtle font-medium text-accent"
                  : "text-muted hover:bg-surface-2 hover:text-fg"
              }`
            }
          >
            <ItemIcon className="h-4 w-4 shrink-0" />
            <span>{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}

export function Sidebar() {
  return (
    <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-border bg-surface md:flex">
      <div className="flex items-center gap-2.5 border-b border-border px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent font-mono text-sm font-bold text-primary-fg">
          KA
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold text-fg">kernelagent</p>
          <p className="text-[11px] text-muted">算子优化控制台</p>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <NavList items={NAV_ITEMS} />
      </div>
      <div className="border-t border-border px-4 py-3 text-[11px] leading-relaxed text-muted">
        单 GPU = 单活跃 Run。
        <br />
        状态一律来自 run 目录。
      </div>
    </aside>
  );
}
