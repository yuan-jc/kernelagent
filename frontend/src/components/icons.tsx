import type { ReactNode } from "react";

/**
 * 极简内联图标集（24x24 线框，stroke 继承 currentColor），
 * 避免为脚手架引入图标依赖；后续可整体替换。
 */

interface IconProps {
  className?: string;
}

function IconBase({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className ?? "h-4 w-4 shrink-0"}
    >
      {children}
    </svg>
  );
}

/** 总览：四宫格 */
export function IconDashboard({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.2" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.2" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.2" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.2" />
    </IconBase>
  );
}

/** Runs：播放圆环 */
export function IconRuns({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <circle cx="12" cy="12" r="9" />
      <path d="M10 8.8v6.4l5.2-3.2z" fill="currentColor" stroke="none" />
    </IconBase>
  );
}

/** API/模型配置：芯片 */
export function IconChip({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M10.2 10.5h3.6v3h-3.6z" />
      <path d="M4 10h3M4 14h3M17 10h3M17 14h3M10 4v3M14 4v3M10 17v3M14 17v3" />
    </IconBase>
  );
}

/** Benchmark：仪表盘 */
export function IconGauge({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <path d="M4.5 14.5a7.5 7.5 0 1 1 15 0" />
      <path d="M12 14.5l3.6-3.6" />
      <path d="M4.5 14.5h2M17.5 14.5h2M12 6.5v2" />
      <path d="M3.5 19h17" />
    </IconBase>
  );
}

/** Profile：脉冲曲线 */
export function IconPulse({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <path d="M3 12h4l2.5-6.5 4.5 13 2.5-6.5H21" />
    </IconBase>
  );
}

/** 设置：滑杆 */
export function IconSliders({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <path d="M4 7h9M17.5 7H20M4 12h3M11.5 12H20M4 17h11M19 17h1" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="12" r="2" />
      <circle cx="17" cy="17" r="2" />
    </IconBase>
  );
}

/** 刷新 */
export function IconRefresh({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <path d="M20.5 12a8.5 8.5 0 1 1-2.5-6" />
      <path d="M20.5 3.5V9H15" />
    </IconBase>
  );
}

/** 空状态：收件箱 */
export function IconInbox({ className }: IconProps) {
  return (
    <IconBase className={className}>
      <path d="M4 13.5V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v7.5" />
      <path d="M4 13.5h4.5l1.5 2.5h4l1.5-2.5H20V18a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
    </IconBase>
  );
}
