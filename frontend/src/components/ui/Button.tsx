import type { ButtonHTMLAttributes } from "react";
import { Spinner } from "./Spinner";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** 为 true 时禁用并显示 spinner（用于提交等异步动作） */
  loading?: boolean;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-fg hover:bg-primary-hover border border-transparent",
  secondary:
    "bg-surface-raised text-fg border border-border hover:border-border-strong",
  ghost: "bg-transparent text-muted border border-transparent hover:text-fg hover:bg-surface-raised",
  danger: "bg-error-soft text-error border border-error/30 hover:bg-error hover:text-white",
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-xs gap-1",
  md: "h-8 px-3 text-sm gap-1.5",
};

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  disabled,
  className = "",
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT_CLASSES[variant]} ${SIZE_CLASSES[size]} ${className}`}
      {...rest}
    >
      {loading && <Spinner size="xs" />}
      {children}
    </button>
  );
}
