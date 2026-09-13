import { CopyButton } from "./CopyButton";

/**
 * 只读源码/文本查看：等宽 + 行号 + 日志底色。
 * 有意不引入 highlight.js 依赖（mock/产物均为短文本；后续可整体替换）。
 */
export function CodeView({
  content,
  path,
  maxHeight = "420px",
}: {
  content: string;
  path?: string;
  maxHeight?: string;
}) {
  const lines = content.split("\n");
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-log-bg">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-1.5">
        <span className="truncate font-mono text-xs text-muted" title={path}>
          {path ?? "file"}
        </span>
        <CopyButton text={content} label="复制内容" />
      </div>
      <div style={{ maxHeight }} className="overflow-auto px-2 py-2 font-mono text-[12.5px] leading-relaxed">
        <ol className="space-y-px">
          {lines.map((line, i) => (
            <li key={i} className="flex gap-3">
              <span className="w-10 shrink-0 select-none text-right text-text-3/70 tabular-nums">
                {i + 1}
              </span>
              <span className="whitespace-pre-wrap break-all text-fg/90">{line || " "}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
