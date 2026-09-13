import { useState } from "react";
import { Button } from "./ui/Button";

/** 复制到剪贴板的小按钮；失败时降级为选中提示（不静默） */
export function CopyButton({
  text,
  label = "复制",
  className = "",
}: {
  text: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用（非安全上下文等）：降级用 execCommand 复制
      const pre = document.createElement("pre");
      pre.textContent = text;
      document.body.appendChild(pre);
      window.getSelection()?.selectAllChildren(pre);
      document.execCommand("copy");
      document.body.removeChild(pre);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }

  return (
    <Button size="sm" variant="ghost" className={className} onClick={copy}>
      {copied ? "已复制" : label}
    </Button>
  );
}
