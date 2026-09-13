import type { ReactNode } from "react";

export type BadgeVariant =
  | "neutral"
  | "primary"
  | "success"
  | "warning"
  | "error"
  | "running";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  neutral: "bg-neutral-soft text-muted-s border-muted-s/25",
  primary: "bg-primary-soft text-primary border-primary/25",
  success: "bg-success-soft text-success border-success/25",
  warning: "bg-warning-soft text-warning border-warning/25",
  error: "bg-error-soft text-error border-error/25",
  running: "bg-running-soft text-running border-running/25",
};

export interface BadgeProps {
  variant?: BadgeVariant;
  /** 在文字前显示状态圆点（running 时自动加脉冲动画） */
  dot?: boolean;
  className?: string;
  /** 原生 title 提示 */
  title?: string;
  children?: ReactNode;
}

const DOT_COLOR: Record<BadgeVariant, string> = {
  neutral: "bg-muted",
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  error: "bg-error",
  running: "bg-running",
};

export function Badge({
  variant = "neutral",
  dot = false,
  className = "",
  title,
  children,
}: BadgeProps) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap ${VARIANT_CLASSES[variant]} ${className}`}
    >
      {dot && (
        <span
          aria-hidden="true"
          className={`h-1.5 w-1.5 rounded-full ${DOT_COLOR[variant]} ${
            variant === "running" ? "animate-pulse-dot" : ""
          }`}
        />
      )}
      {children}
    </span>
  );
}
