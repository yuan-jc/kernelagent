import type { HTMLAttributes } from "react";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** 内边距由使用方通过 CardContent 控制，容器本身不带 padding */
}

export function Card({ className = "", ...rest }: CardProps) {
  return (
    <div
      className={`rounded-lg border border-border bg-surface shadow-sm ${className}`}
      {...rest}
    />
  );
}

export function CardHeader({
  className = "",
  ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3 ${className}`}
      {...rest}
    />
  );
}

export function CardTitle({
  className = "",
  ...rest
}: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={`text-sm font-semibold text-fg ${className}`} {...rest} />
  );
}

export function CardDescription({
  className = "",
  ...rest
}: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={`text-xs text-muted ${className}`} {...rest} />;
}

export function CardContent({
  className = "",
  ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={`px-4 py-3 ${className}`} {...rest} />;
}
