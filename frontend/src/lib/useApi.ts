/**
 * 轻量数据获取 hook：不引入额外依赖，覆盖
 * 加载 / 错误 / 刷新 / 轮询 / 终态停止 / 页面隐藏暂停 / 失败退避。
 * 接口形态接近 TanStack Query 的 useQuery，后续可平滑替换（设计规范 §5）。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { toApiError } from "../api";

/** 连续失败达到该次数后退避到 BACKOFF_MS 并置 degraded（不静默假装在刷新） */
const FAILURE_THRESHOLD = 3;
const BACKOFF_MS = 10_000;

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
  /** 仅首载（尚无任何数据）时为 true，避免轮询导致的闪烁 */
  loading: boolean;
  /** 连续失败 >= 3 次后退避中；UI 应显示"连接失败，重试中" */
  degraded: boolean;
  /** 手动刷新（重新执行 fetcher 并重置退避） */
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
  const [degraded, setDegraded] = useState(false);
  const [tick, setTick] = useState(0);

  // fetcher/options 每次渲染都是新引用，用 ref 保存避免把调用方拖进依赖数组
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const dataRef = useRef<T | null>(data);
  dataRef.current = data;
  const shouldPollRef = useRef(options?.shouldPoll);
  shouldPollRef.current = options?.shouldPoll;
  const failuresRef = useRef(0);

  const reload = useCallback(() => {
    failuresRef.current = 0;
    setDegraded(false);
    setTick((t) => t + 1);
  }, []);

  // deps 长度由调用点静态决定，展开是安全的
  useEffect(() => {
    let cancelled = false;
    if (dataRef.current === null) setLoading(true);
    fetcherRef
      .current()
      .then((result) => {
        if (cancelled) return;
        failuresRef.current = 0;
        setDegraded(false);
        setData(result);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        failuresRef.current += 1;
        if (failuresRef.current >= FAILURE_THRESHOLD) setDegraded(true);
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
    let hidden = document.visibilityState === "hidden";
    const onVisibility = () => {
      const wasHidden = hidden;
      hidden = document.visibilityState === "hidden";
      // 回前台立即刷一次（spec §5）
      if (wasHidden && !hidden) {
        if (shouldPollRef.current && !shouldPollRef.current(dataRef.current)) return;
        setTick((t) => t + 1);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    const id = window.setInterval(() => {
      if (hidden) return; // 页面隐藏时暂停轮询
      const backoff = failuresRef.current >= FAILURE_THRESHOLD ? BACKOFF_MS : pollMs;
      if (backoff !== pollMs) return; // 退避由下面的 timeout 处理
      if (shouldPollRef.current && !shouldPollRef.current(dataRef.current)) return;
      setTick((t) => t + 1);
    }, pollMs);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.clearInterval(id);
    };
  }, [pollMs]);

  // 退避定时器：连续失败后降频到 10s 重试一次，成功后由 then 分支重置
  useEffect(() => {
    if (!degraded || !pollMs) return;
    const id = window.setTimeout(() => {
      if (shouldPollRef.current && !shouldPollRef.current(dataRef.current)) return;
      setTick((t) => t + 1);
    }, BACKOFF_MS);
    return () => window.clearTimeout(id);
  }, [degraded, pollMs, tick]);

  return { data, error, loading, degraded, reload };
}
