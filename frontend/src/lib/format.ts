/** 展示用格式化工具（无业务语义） */

/** Unix 秒 -> 本地时间字符串；非法输入原样返回 */
export function formatDateTime(unixSeconds: number | undefined | null): string {
  if (typeof unixSeconds !== "number" || !Number.isFinite(unixSeconds)) {
    return "-";
  }
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleString("zh-CN", { hour12: false });
}

/** 秒 -> 紧凑时长，如 1h30m / 5m00s / 42s */
export function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return "-";
  }
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

/** token 数 -> 12.3k / 1.2M */
export function formatCount(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

/** 长哈希 -> 前 8 位省略号 */
export function shortHash(hash: string | null | undefined): string {
  if (!hash) return "-";
  return hash.length > 12 ? `${hash.slice(0, 8)}…` : hash;
}

/** Unix 秒 -> 相对时间（"3m ago" 风格）；非法输入返回 null（调用方显示 "—"） */
export function relativeTime(unixSeconds: number | undefined | null): string | null {
  if (typeof unixSeconds !== "number" || !Number.isFinite(unixSeconds)) return null;
  const diff = Date.now() / 1000 - unixSeconds;
  if (diff < 0) return null;
  if (diff < 60) return `${Math.max(Math.floor(diff), 0)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/**
 * run_id 形如 "20260913-102458"（服务端 time.strftime 生成）。
 * 从中解析启动时间；解析失败返回 null（调用方显示 "—"，不编造）。
 */
export function parseRunIdTime(runId: string): number | null {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s] = m.map(Number) as unknown as number[];
  const ts = new Date(y, mo - 1, d, h, mi, s).getTime() / 1000;
  return Number.isFinite(ts) ? ts : null;
}

/** 毫秒样本统计；空数组返回 null（缺失指标不是零） */
export function sampleStats(values: number[] | undefined | null): {
  n: number;
  min: number;
  mean: number;
  p95: number;
  max: number;
} | null {
  if (!values || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const idx = (q: number) => sorted[Math.min(Math.floor(q * (sorted.length - 1)), sorted.length - 1)]!;
  return {
    n: sorted.length,
    min: sorted[0]!,
    mean: sum / sorted.length,
    p95: idx(0.95),
    max: sorted[sorted.length - 1]!,
  };
}

/** 文本截断（候选 detail 400 字符等）；返回 null 表示无需截断 */
export function truncate(text: string, max: number): string | null {
  return text.length > max ? text.slice(0, max) : null;
}
