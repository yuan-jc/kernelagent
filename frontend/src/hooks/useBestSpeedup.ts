/**
 * Dashboard「最佳加速比」瓦片的数据源：为最近 ≤10 条 runs 逐个拉
 * GET /api/runs/<id>/report（P2），提取 champion 候选的 ratio_ci_95。
 *
 * 防请求风暴的三道闸：
 * 1. 数量闸：最多取 runs 列表前 MAX_RUNS 条（列表新在前）；
 * 2. 并发闸：任意时刻最多 CONCURRENCY 个在途 report 请求；
 * 3. 触发闸：effect 以 runs id 列表的稳定字符串为依赖——runs 列表轮询
 *   （5s/15s）产生的新对象不会触发重拉，只有列表内容（id 集合）变化才重拉；
 *   已成功的 run 结果缓存在 ref 中，同一次挂载内绝不重复请求。
 *
 * 诚实性：只认 report.champion 与候选记录 candidate_sha256 匹配且
 * ratio_ci_95 完整的记录；一个都拿不到时保持 NOT_RUN（缺失不是 0）。
 */

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { RunReport } from "../api";

const MAX_RUNS = 10;
const CONCURRENCY = 3;

export interface BestSpeedup {
  /** champion ratio_ci_95 的下界（择优依据，与页内 CI 展示一致） */
  low: number;
  high: number;
  runId: string;
}

interface CacheEntry {
  /** 该 run 的 champion CI；无 champion/CI 时为 null */
  best: BestSpeedup | null;
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await fn(items[index]!);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, () => worker()),
  );
  return results;
}

/** 从单条 report 提取 champion 候选的加速比 CI；不完整一律 null */
export function championSpeedupOf(runId: string, report: RunReport): BestSpeedup | null {
  const sha = report.champion?.candidate_sha256;
  if (!sha) return null;
  const match = (report.candidates ?? []).find(
    (c) => c.candidate_sha256 === sha && Array.isArray(c.ratio_ci_95),
  );
  const ci = match?.ratio_ci_95;
  if (!ci || typeof ci[0] !== "number" || typeof ci[1] !== "number") return null;
  return { low: ci[0], high: ci[1], runId };
}

/**
 * @param runs runs 列表（新在前）；null/undefined 时不动作
 * @returns best = 全部最近 run 中 CI 下界最高者；null = 无任何数据（显示 NOT_RUN）
 */
export function useBestSpeedup(
  runs: Array<{ run_id: string }> | null | undefined,
): { best: BestSpeedup | null; loading: boolean } {
  const [best, setBest] = useState<BestSpeedup | null>(null);
  const [loading, setLoading] = useState(false);
  // 同一挂载内已取过的 run 不再请求（runs 列表轮询不触发重拉）
  const cacheRef = useRef<Map<string, CacheEntry>>(new Map());

  // 稳定依赖：id 串。列表对象每轮轮询都换新引用，id 集合不变就不重拉。
  const idsKey = (runs ?? [])
    .slice(0, MAX_RUNS)
    .map((r) => r.run_id)
    .join(",");

  useEffect(() => {
    const ids = idsKey ? idsKey.split(",") : [];
    if (ids.length === 0) {
      setBest(null);
      setLoading(false);
      return;
    }
    const missing = ids.filter((id) => !cacheRef.current.has(id));
    // 先用缓存即时给出结果，再补拉缺失的
    const applyBest = () => {
      const all = ids
        .map((id) => cacheRef.current.get(id)?.best ?? null)
        .filter((v): v is BestSpeedup => v !== null);
      setBest(all.length > 0 ? all.reduce((a, b) => (b.low > a.low ? b : a)) : null);
    };
    applyBest();
    if (missing.length === 0) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    void mapLimit(missing, CONCURRENCY, async (runId): Promise<void> => {
      let entry: CacheEntry;
      try {
        const report = await api.getReport(runId);
        entry = { best: championSpeedupOf(runId, report) };
      } catch {
        // 404（运行中/无 report）或其他错误：该 run 记为无数据，不冒充 0
        entry = { best: null };
      }
      if (cancelled) return;
      cacheRef.current.set(runId, entry);
    }).then(() => {
      if (cancelled) return;
      applyBest();
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [idsKey]);

  return { best, loading };
}
