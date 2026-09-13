import type { HTMLAttributes } from "react";

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: "xs" | "sm" | "md" | "lg";
}

const SIZE_CLASSES = {
  xs: "h-3 w-3 border",
  sm: "h-4 w-4 border-2",
  md: "h-6 w-6 border-2",
  lg: "h-8 w-8 border-[3px]",
} as const;

export function Spinner({
  size = "md",
  className = "",
  ...rest
}: SpinnerProps) {
  return (
    <div
      role="status"
      aria-label="加载中"
      className={`inline-block animate-spin rounded-full border-current border-t-transparent text-primary ${SIZE_CLASSES[size]} ${className}`}
      {...rest}
    />
  );
}

/** 居中的整块加载占位 */
export function LoadingBlock({ label = "加载中…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted">
      <Spinner size="sm" />
      <span>{label}</span>
    </div>
  );
}
