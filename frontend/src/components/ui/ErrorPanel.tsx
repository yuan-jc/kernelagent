import { toApiError } from "../../api";
import { Button } from "./Button";

export interface ErrorPanelProps {
  /** 标题；未提供时从错误推导 */
  title?: string;
  /** 任意抛出的错误对象（Error / ApiError / 字符串） */
  error?: unknown;
  /** 直接给定消息（与 error 二选一） */
  message?: string;
  /** 附加大段文本（如 run 的 error_tail），以等宽字体展示 */
  details?: string | null;
  /** 提供时显示"重试"按钮 */
  onRetry?: () => void;
  className?: string;
}

export function ErrorPanel({
  title,
  error,
  message,
  details,
  onRetry,
  className = "",
}: ErrorPanelProps) {
  const text = message ?? (error !== undefined ? toApiError(error).message : "发生未知错误");
  const heading = title ?? "出错了";
  return (
    <div
      role="alert"
      className={`rounded-lg border border-error/40 bg-error-soft px-4 py-3 text-sm ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-medium text-error">{heading}</p>
        {onRetry && (
          <Button size="sm" variant="secondary" onClick={onRetry}>
            重试
          </Button>
        )}
      </div>
      <p className="mt-1 break-words text-xs leading-relaxed text-fg/90">{text}</p>
      {details && (
        <pre className="mt-2 max-h-64 overflow-auto rounded-md border border-error/30 bg-bg/60 p-2 font-mono text-xs whitespace-pre-wrap text-muted">
          {details}
        </pre>
      )}
    </div>
  );
}
