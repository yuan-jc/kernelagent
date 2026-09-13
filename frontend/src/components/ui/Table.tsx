import type {
  HTMLAttributes,
  ReactNode,
  TableHTMLAttributes,
  TdHTMLAttributes,
  ThHTMLAttributes,
} from "react";

export function Table({
  className = "",
  ...rest
}: TableHTMLAttributes<HTMLTableElement>) {
  return (
    <table
      className={`w-full border-collapse text-left text-sm ${className}`}
      {...rest}
    />
  );
}

export function THead({
  className = "",
  ...rest
}: HTMLAttributes<HTMLTableSectionElement>) {
  return (
    <thead
      className={`border-b border-border text-xs text-muted uppercase ${className}`}
      {...rest}
    />
  );
}

export function TBody({
  className = "",
  ...rest
}: HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={className} {...rest} />;
}

export function TR({
  className = "",
  ...rest
}: HTMLAttributes<HTMLTableRowElement>) {
  return (
    <tr className={`border-b border-border last:border-0 ${className}`} {...rest} />
  );
}

export function TH({
  className = "",
  ...rest
}: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th className={`px-3 py-2 font-medium tracking-wide ${className}`} {...rest} />
  );
}

export function TD({
  className = "",
  ...rest
}: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={`px-3 py-2 align-middle ${className}`} {...rest} />;
}

/** 常见的"表格壳"：外层卡片 + 可横向滚动 */
export function TableWrapper({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface">
      {children}
    </div>
  );
}
