import type { ReactNode } from "react";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  /** 可选的主操作（如"刷新"按钮） */
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-surface px-6 py-12 text-center ${className}`}
    >
      {icon && <div className="text-muted">{icon}</div>}
      <p className="text-sm font-medium text-fg">{title}</p>
      {description && (
        <p className="max-w-md text-xs leading-relaxed text-muted">
          {description}
        </p>
      )}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
