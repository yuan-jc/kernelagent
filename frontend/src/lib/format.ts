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
