/**
 * Run 产物读取 hooks（P2/P3/P4/P5，带诚实降级）。
 *
 * 降级规则：调用方已确认 run 存在（run 详情快照加载成功）之后，
 * 子路径 404 一律视为「端点未实现/未落盘」（C3 未上线时的真实后端行为），
 * 返回 unavailable=true 而不是把错误冒充成数据。其余错误（网络/500）
 * 正常抛给 UI 呈现。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import type {
  ActionRecord,
  JournalEntry,
  JournalResponse,
  RunReport,
  WorkspaceEntry,
  WorkspaceFileResponse,
} from "../api";

export type ArtifactSource = "live" | "unavailable";

export interface ArtifactData<T> {
  data: T | null;
  source: ArtifactSource;
  /** 404：端点未实现或产物未生成（如运行中尚无 report） */
  unavailable: boolean;
  unavailableReason: string | null;
  error: Error | null;
  loading: boolean;
}

export interface ArtifactState<T> extends ArtifactData<T> {
  reload: () => void;
}

function initial<T>(): ArtifactData<T> {
  return {
    data: null,
    source: "live",
    unavailable: false,
    unavailableReason: null,
    error: null,
    loading: true,
  };
}

function toUnavailable<T>(reason: string): ArtifactData<T> {
  return {
    data: null,
    source: "live",
    unavailable: true,
    unavailableReason: reason,
    error: null,
    loading: false,
  };
}

// -- P3 journal（增量优先，未实现时全量兜底，再不行标记 unavailable）----------

export type JournalSource = "live-incremental" | "live-full";

export interface JournalState {
  entries: JournalEntry[];
  nextAfter: number;
  source: ArtifactSource;
  mode: JournalSource | null;
  unavailableReason: string | null;
  error: Error | null;
}

export function useJournal(
  runId: string,
  opts: { pollMs?: number; enabled?: boolean } = {},
): JournalState & { reload: () => void } {
  const { pollMs = 0, enabled = true } = opts;
  const [state, setState] = useState<JournalState>({
    entries: [],
    nextAfter: 0,
    source: "live",
    mode: null,
    unavailableReason: null,
    error: null,
  });
  const [tick, setTick] = useState(0);
  const nextAfterRef = useRef(0);
  const entriesRef = useRef<JournalEntry[]>([]);
  const fullFallbackRef = useRef(false);

  const reset = useCallback(() => {
    nextAfterRef.current = 0;
    entriesRef.current = [];
    fullFallbackRef.current = false;
    setState({
      entries: [],
      nextAfter: 0,
      source: "live",
      mode: null,
      unavailableReason: null,
      error: null,
    });
  }, []);

  const reload = useCallback(() => {
    reset();
    setTick((t) => t + 1);
  }, [reset]);

  // runId 变化时重置游标
  useEffect(() => {
    reset();
  }, [runId, reset]);

  useEffect(() => {
    let cancelled = false;
    const UNAVAILABLE_NOTE =
      "P3 journal 端点未实现（GET /api/runs/<id>/journal 404）。运行中进度来自快照 in_flight/progress；journal 事件流待后端上线。";

    function applyFull(full: JournalResponse) {
      entriesRef.current = full.entries ?? [];
      setState({
        entries: entriesRef.current,
        nextAfter: full.next_after ?? entriesRef.current.length,
        source: "live",
        mode: "live-full",
        unavailableReason: null,
        error: null,
      });
    }

    async function fetchOnce() {
      try {
        if (fullFallbackRef.current) {
          // 端点只支持全量：每轮拉全量（兜底路径）
          applyFull(await api.getJournal(runId));
          return;
        }
        const inc = await api.getJournal(runId, nextAfterRef.current);
        const chunk = inc.entries ?? [];
        if (nextAfterRef.current === 0) {
          entriesRef.current = chunk;
        } else {
          // 追加未持有的新条目（按 next_after 对齐，避免重复）
          const known = entriesRef.current.length;
          entriesRef.current =
            inc.next_after === known + chunk.length
              ? [...entriesRef.current, ...chunk]
              : chunk; // 服务端游标不一致时以最新分片为准
        }
        nextAfterRef.current = inc.next_after ?? nextAfterRef.current + chunk.length;
        setState({
          entries: entriesRef.current,
          nextAfter: nextAfterRef.current,
          source: "live",
          mode: "live-incremental",
          unavailableReason: null,
          error: null,
        });
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof ApiError && cause.status === 404) {
          if (!fullFallbackRef.current) {
            // 先试一次全量兜底（端点未实现增量参数的情况）
            fullFallbackRef.current = true;
            try {
              applyFull(await api.getJournal(runId));
              return;
            } catch (cause2) {
              if (cause2 instanceof ApiError && cause2.status === 404) {
                setState({
                  entries: [],
                  nextAfter: 0,
                  source: "unavailable",
                  mode: null,
                  unavailableReason: UNAVAILABLE_NOTE,
                  error: null,
                });
                return;
              }
            }
          } else {
            setState((prev) => ({
              ...prev,
              source: "unavailable",
              mode: null,
              unavailableReason: UNAVAILABLE_NOTE,
              error: null,
            }));
            return;
          }
        }
        setState((prev) => ({
          ...prev,
          error: cause instanceof Error ? cause : new Error(String(cause)),
        }));
      }
    }
    void fetchOnce();
    return () => {
      cancelled = true;
    };
  }, [runId, tick]);

  useEffect(() => {
    if (!pollMs || !enabled) return;
    const id = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      setTick((t) => t + 1);
    }, pollMs);
    return () => window.clearInterval(id);
  }, [pollMs, enabled]);

  return { ...state, reload };
}

// -- 通用一次性产物 hook（workspace / report / record）------------------------

function useArtifact<T>(
  runId: string,
  fetcher: (runId: string) => Promise<T>,
  deps: readonly unknown[] = [],
  missingNote: string,
): ArtifactState<T> {
  const [state, setState] = useState<ArtifactData<T>>(initial<T>);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState(initial<T>());
    (async () => {
      try {
        const data = await fetcher(runId);
        if (cancelled) return;
        setState({ data, source: "live", unavailable: false, unavailableReason: null, error: null, loading: false });
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof ApiError && cause.status === 404) {
          // 404 语义：优先采用服务端的具体消息（如 "has no record 'x'"），
          // 纯 "not found" 视为端点未实现，用调用方的通用说明。
          const msg = cause.message?.trim();
          setState(
            toUnavailable<T>(
              msg && msg !== "not found" ? `服务端返回：${msg}` : missingNote,
            ),
          );
          return;
        }
        setState({
          data: null,
          source: "live",
          unavailable: false,
          unavailableReason: null,
          error: cause instanceof Error ? cause : new Error(String(cause)),
          loading: false,
        });
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, tick, ...deps]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

export function useWorkspace(runId: string): ArtifactState<{ entries: WorkspaceEntry[] }> {
  return useArtifact(
    runId,
    async (id) => {
      const ws = await api.getWorkspace(id);
      return { entries: ws.entries ?? [] };
    },
    [],
    "P5 workspace 端点未实现（404）：工作区文件清单与容器日志需要后端只读产物端点上线后展示。",
  );
}

export function useWorkspaceFile(
  runId: string,
  filePath: string | null,
): ArtifactState<WorkspaceFileResponse> {
  return useArtifact(
    runId,
    async (id) => api.getWorkspaceFile(id, filePath ?? ""),
    [filePath],
    "P5 workspace 端点未实现（404）：无法读取文件内容。",
  );
}

export function useRunReport(runId: string): ArtifactState<RunReport> {
  return useArtifact(
    runId,
    (id) => api.getReport(id),
    [],
    "P2 report 端点未实现（404）：report.json 的身份链（commit/protocol/problem_sha256）待后端上线后展示；下方为快照中已有的证据字段。",
  );
}

/**
 * 快照 state=unknown 时的 report 兜底：仅当 trigger 非 null（调用方传入
 * 「处于 unknown 的快照」）时拉取 report.json，且 trigger 每次变化（新一轮
 * 轮询得到新快照）都会重拉，保证终态落盘后的下一拍内能读到。
 * 404（report 尚未生成）不是错误：安静地保持 null，等待下一拍。
 */
export function useReportFallback(
  runId: string,
  trigger: unknown,
): { report: RunReport | null; error: Error | null } {
  const [state, setState] = useState<{ report: RunReport | null; error: Error | null }>({
    report: null,
    error: null,
  });

  useEffect(() => {
    if (!trigger) return;
    let cancelled = false;
    api
      .getReport(runId)
      .then((report) => {
        if (!cancelled) setState({ report, error: null });
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        // 404 = report 还没写（与快照一致的中间态）；其余错误记录但不打断轮询
        setState({
          report: null,
          error: cause instanceof ApiError && cause.status === 404 ? null : (cause instanceof Error ? cause : new Error(String(cause))),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [runId, trigger]);

  if (!trigger) return { report: null, error: null };
  return state;
}

export function useRecord(
  runId: string,
  actionId: string | null,
): ArtifactState<ActionRecord> {
  return useArtifact(
    runId,
    (id) => api.getRecord(id, actionId ?? ""),
    [actionId],
    "P4 records 端点未实现（404）：单条 action 记录（batches_ms / async_leak）待后端上线后展示。",
  );
}

/** records 只读清单（C3 后端）；用于枚举有哪些 final 记录可读 */
export function useRecordsList(runId: string): ArtifactState<string[]> {
  return useArtifact(
    runId,
    async (id) => {
      const res = await api.getRecords(id);
      const ids = (res.records ?? [])
        .filter((r) => r.variant === "final")
        .map((r) => r.record);
      return [...new Set(ids)];
    },
    [],
    "P4 records 端点未实现（404）：无法枚举 records/ 目录中的记录。",
  );
}
