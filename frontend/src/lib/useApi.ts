/**
 * 轻量数据获取 hook：不引入额外依赖，覆盖脚手架期的
 * 加载/错误/刷新/轮询需求；后续可平滑替换为 TanStack Query
 * （设计规范 §5 的 refetchInterval/退避策略落点）。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { toApiError } from "../api";

export interface UseApiOptions<T> {
  /** 轮询间隔（毫秒）；0/undefined 表示不轮询 */
  pollMs?: number;
  /**
   * 轮询谓词：返回 false 时跳过下一次请求（如 run 已到终态停止轮询，
   * spec §5）。不在间隔变化时重新请求，只 gate 掉 tick。
   */
  shouldPoll?: (data: T | null) => boolean;
}

export interface AsyncState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  /** 手动刷新（重新执行 fetcher） */
  reload: () => void;
}

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: readonly unknown[],
  options?: UseApiOptions<T>,
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  // fetcher/options 每次渲染都是新引用，用 ref 保存避免把调用方拖进依赖数组
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const dataRef = useRef<T | null>(data);
  dataRef.current = data;
  const shouldPollRef = useRef(options?.shouldPoll);
  shouldPollRef.current = options?.shouldPoll;

  const reload = useCallback(() => setTick((t) => t + 1), []);

  // deps 长度由调用点静态决定，展开是安全的
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetcherRef
      .current()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(toApiError(cause));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);

  const pollMs = options?.pollMs;
  useEffect(() => {
    if (!pollMs) return;
    const id = window.setInterval(() => {
      const current = dataRef.current;
      if (shouldPollRef.current && !shouldPollRef.current(current)) return;
      setTick((t) => t + 1);
    }, pollMs);
    return () => window.clearInterval(id);
  }, [pollMs]);

  return { data, error, loading, reload };
}
