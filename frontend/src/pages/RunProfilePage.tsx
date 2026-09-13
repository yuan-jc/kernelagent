import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import type { ActionRecord } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { LogViewer } from "../components/LogViewer";
import { BackToRunLink } from "../components/Tabs";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorPanel,
  LoadingBlock,
  NotRunBadge,
} from "../components/ui";
import { textToLines } from "../lib/journalLines";
import { isTerminalRunState } from "../lib/states";
import { sampleStats } from "../lib/format";
import { useApi } from "../lib/useApi";
import { useRecordsList, useWorkspace } from "../hooks/useRunArtifacts";
import { prefs } from "../lib/prefs";

/**
 * Profile 报告页（spec §3.7）：timing 记录的批次延迟分布、样本统计、
 * async_leak 标注与容器日志（P4/P5）。约束：
 * - profiling 与正式 timing 分开，本页仅展示 run 目录中的持久化事实；
 * - 无数据时显示 NOT_RUN 空态，绝不画 0、不画空图冒充；
 * - SOL 不证明算法最优，本页不做因果结论。
 */

interface RecordSlot {
  state: "ok" | "missing" | "unavailable" | "error";
  record?: ActionRecord;
  message?: string;
}

type RecordMap = Record<string, RecordSlot>;

/** 逐条读取（单条 404/失败不影响其他记录）；全部 404 且无服务端具体消息视为端点未实现 */
function useRecords(
  runId: string,
  ids: string[],
): { map: RecordMap; loading: boolean; unavailable: boolean; reload: () => void } {
  const [map, setMap] = useState<RecordMap>({});
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [tick, setTick] = useState(0);
  const idsKey = ids.join(",");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const list = idsKey ? idsKey.split(",") : [];
    if (list.length === 0) {
      setMap({});
      setUnavailable(false);
      setLoading(false);
      return;
    }
    void Promise.all(
      list.map(async (id): Promise<readonly [string, RecordSlot]> => {
        try {
          return [id, { state: "ok", record: await api.getRecord(runId, id) }];
        } catch (cause) {
          const status = (cause as { status?: number }).status;
          const message = (cause as { message?: string }).message;
          if (status === 404) return [id, { state: "missing", message }];
          return [id, { state: "error", message: (cause as Error).message }];
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      const next: RecordMap = {};
      for (const [id, slot] of entries) next[id] = slot;
      setMap(next);
      // 全部 404 且服务端没有具体消息（纯 "not found"）=> P4 端点未实现；
      // 否则是该 run 确实没有记录（NOT_RUN，不是端点问题）。
      const endpointMissing = entries.every(
        ([, slot]) =>
          slot.state === "missing" &&
          (!slot.message || slot.message.trim().toLowerCase() === "not found"),
      );
      setUnavailable(endpointMissing);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [runId, idsKey, tick]);

  return { map, loading, unavailable, reload: () => setTick((t) => t + 1) };
}

/** 批次延迟散点图（手写 SVG；缺失数据不调用本组件） */
function BatchLatencyChart({
  series,
}: {
  series: Array<{ name: string; values: number[]; color: string }>;
}) {
  const W = 640;
  const H = 140;
  const PAD_L = 46;
  const PAD_B = 22;
  const all = series.flatMap((s) => s.values);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const yOf = (v: number) => 8 + (1 - (v - min) / span) * (H - PAD_B - 16);
  const maxN = Math.max(...series.map((s) => s.values.length));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="批次延迟分布">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = 8 + f * (H - PAD_B - 16);
        const v = max - f * span;
        return (
          <g key={f}>
            <line x1={PAD_L} x2={W - 8} y1={y} y2={y} stroke="var(--color-border)" strokeWidth="1" />
            <text x={PAD_L - 6} y={y + 3} textAnchor="end" fontSize="9" fill="var(--color-text-3)">
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}
      {series.map((s, si) =>
        s.values.map((v, i) => {
          const x = PAD_L + 12 + (i / Math.max(maxN - 1, 1)) * (W - PAD_L - 40);
          return (
            <circle
              key={`${si}-${i}`}
              cx={x}
              cy={yOf(v)}
              r={3}
              fill={s.color}
              opacity={0.85}
            >
              <title>{`${s.name} batch#${i + 1}: ${v.toFixed(3)} ms`}</title>
            </circle>
          );
        }),
      )}
      <text x={PAD_L} y={H - 6} fontSize="9" fill="var(--color-text-3)">
        batch 1..{maxN}（每点 = 单批次耗时 ms；样本见统计表）
      </text>
    </svg>
  );
}

function StatsTable({ series }: { series: Array<{ name: string; values: number[] }> }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-left text-xs">
        <thead className="border-b border-border text-[11px] uppercase text-muted">
          <tr>
            <th className="px-2.5 py-1.5 font-medium">记录</th>
            <th className="px-2.5 py-1.5 font-medium">样本数</th>
            <th className="px-2.5 py-1.5 font-medium">min</th>
            <th className="px-2.5 py-1.5 font-medium">mean</th>
            <th className="px-2.5 py-1.5 font-medium">p95</th>
            <th className="px-2.5 py-1.5 font-medium">max</th>
          </tr>
        </thead>
        <tbody>
          {series.map((s) => {
            const stats = sampleStats(s.values);
            return (
              <tr key={s.name} className="border-b border-border/60 last:border-0">
                <td className="px-2.5 py-1.5 font-mono">{s.name}</td>
                <td className="px-2.5 py-1.5 font-mono tabular-nums">
                  {stats ? stats.n : <NotRunBadge what="样本数" />}
                </td>
                <td className="px-2.5 py-1.5 font-mono tabular-nums">
                  {stats ? stats.min.toFixed(3) : "—"}
                </td>
                <td className="px-2.5 py-1.5 font-mono tabular-nums">
                  {stats ? stats.mean.toFixed(3) : "—"}
                </td>
                <td className="px-2.5 py-1.5 font-mono tabular-nums">
                  {stats ? stats.p95.toFixed(3) : "—"}
                </td>
                <td className="px-2.5 py-1.5 font-mono tabular-nums">
                  {stats ? stats.max.toFixed(3) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RunProfilePage() {
  const { runId = "" } = useParams();
  const snapshotState = useApi(() => api.getRun(runId), [runId], { pollMs: 0 });
  const running = snapshotState.data ? !isTerminalRunState(snapshotState.data.state) : false;

  // records 清单（C3 后端 /records）驱动：哪些 final 记录可读；
  // 端点未实现时回退为 baseline + report 候选的猜测列表（同样诚实降级）。
  const recordsList = useRecordsList(runId);
  const recordIds = useMemo(() => {
    if (recordsList.data && recordsList.data.length > 0) {
      return recordsList.data.slice(0, 24); // 防超长 run 拉爆：图表取前 24 条
    }
    const ids = new Set<string>(["baseline-eager"]);
    for (const c of snapshotState.data?.candidates ?? []) {
      if (c.candidate) ids.add(c.candidate);
    }
    return [...ids];
  }, [recordsList.data, snapshotState.data]);

  const records = useRecords(runId, recordIds);
  const workspace = useWorkspace(runId);
  const [logFile, setLogFile] = useState<string | null>(null);

  const measured = Object.entries(records.map)
    .filter(([, slot]) => slot.state === "ok" && Array.isArray(slot.record?.batches_ms) && (slot.record?.batches_ms?.length ?? 0) > 0)
    .map(([id, slot]) => ({
      name: id,
      values: slot.record!.batches_ms!,
      color: id === "baseline-eager" ? "var(--color-accent)" : "var(--color-ok)",
    }));
  const hasAnyData = measured.length > 0;
  const asyncLeaks = Object.entries(records.map)
    .filter(([, slot]) => slot.state === "ok" && typeof slot.record?.async_leak === "boolean")
    .map(([id, slot]) => ({ id, leak: slot.record!.async_leak! }));

  const containerDirs = (workspace.data?.entries ?? []).filter(
    (e) => e.path.startsWith("container-") && e.files?.some((f) => f === "stdout.log" || f === "stderr.log"),
  );

  return (
    <div className="space-y-5">
      <PageHeader
        title={
          <span className="inline-flex flex-wrap items-center gap-3">
            <BackToRunLink runId={runId} />
            <span>Profile 报告</span>
            <span className="font-mono text-sm font-normal text-muted">{runId}</span>
          </span>
        }
        description="profiling 与正式 timing 分开；本页仅呈现 run 目录中的持久化事实（P4 records / P5 workspace），不做因果推断，不用于晋升判定。"
        actions={
          <div className="flex items-center gap-2">
            {running && <Badge variant="running" dot>运行中（数据陆续落盘）</Badge>}
            {!hasAnyData && <NotRunBadge what="timing batches_ms 数据" />}
            <Button size="sm" variant="secondary" onClick={records.reload}>
              重新读取
            </Button>
          </div>
        }
      />

      {snapshotState.error && (
        <ErrorPanel title="run 快照读取失败" error={snapshotState.error} onRetry={snapshotState.reload} />
      )}

      {/* 批次延迟与样本统计 */}
      <Card>
        <CardHeader>
          <CardTitle>批次延迟（timing records · batches_ms）</CardTitle>
          <span className="text-[11px] text-muted">accent = baseline-eager；绿 = 候选</span>
        </CardHeader>
        <CardContent className="space-y-3">
          {records.loading ? (
            <LoadingBlock label="读取 records（P4）…" />
          ) : records.unavailable ? (
            <EmptyState
              title="P4 records 端点未实现（404）"
              description="批次延迟数据来自 records/<action_id>.json（batches_ms），需要后端 P4 端点上线；在此之前本区保持 NOT_RUN 空态而非空图。"
            />
          ) : hasAnyData ? (
            <>
              <BatchLatencyChart series={measured} />
              <StatsTable series={measured} />
            </>
          ) : (
            <EmptyState
              title="暂无 timing 样本（NOT_RUN）"
              description="该 run 尚无任何 status=measured 的 timing 记录（可能未到 timing 阶段、候选全部失败或预算耗尽）。缺失指标不是零，不绘制占位数据。"
            />
          )}
          {asyncLeaks.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs text-muted">async_leak：</span>
              {asyncLeaks.map(({ id, leak }) => (
                <Badge key={id} variant={leak ? "error" : "success"}>
                  {id}: {leak ? "检测到异步泄漏" : "未检测到"}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 容器日志 */}
      <Card>
        <CardHeader>
          <CardTitle>容器日志（workspace · 只读）</CardTitle>
          {workspace.unavailable && <NotRunBadge what="workspace 端点（P5）" />}
        </CardHeader>
        <CardContent className="space-y-2">
          {workspace.unavailable ? (
            <EmptyState
              title="P5 workspace 端点未实现（404）"
              description="容器 stdout/stderr 需要 GET /api/runs/<id>/workspace 只读端点上线后展示。"
            />
          ) : workspace.loading ? (
            <LoadingBlock />
          ) : containerDirs.length === 0 ? (
            <EmptyState
              title="暂无容器日志"
              description="容器产物目录尚未生成（通常在 evaluate 阶段之后出现）。"
            />
          ) : (
            <>
              <div className="flex flex-wrap gap-1.5">
                {containerDirs.flatMap((dir) =>
                  (dir.files ?? [])
                    .filter((f) => f === "stdout.log" || f === "stderr.log")
                    .map((f) => {
                      const full = `${dir.path}/${f}`;
                      return (
                        <button
                          key={full}
                          type="button"
                          onClick={() => setLogFile(full)}
                          className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                            logFile === full
                              ? "border-accent bg-accent-subtle text-accent"
                              : "border-border bg-surface-2 text-muted hover:text-fg"
                          }`}
                        >
                          {f} · {dir.path.slice(0, 14)}…
                        </button>
                      );
                    }),
                )}
              </div>
              {logFile && <ContainerFile runId={runId} path={logFile} />}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="text-[11px] leading-relaxed text-muted">
          约束提示：SOL 不证明算法最优；本页任何机制/因果结论都需要对应 journal /
          report 证据支撑，前端不会推导状态。加速比 CI 与晋升结论以 Run 详情页
          （服务端枚举）为准。
        </CardContent>
      </Card>
    </div>
  );
}

function ContainerFile({ runId, path }: { runId: string; path: string }) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getWorkspaceFile(runId, path)
      .then((res) => {
        if (!cancelled) setContent(res.content);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause : new Error(String(cause)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, path, tick]);

  if (loading) return <LoadingBlock />;
  if (error) return <ErrorPanel title="文件读取失败" error={error} onRetry={() => setTick((t) => t + 1)} />;
  return (
    <LogViewer
      lines={textToLines(content ?? "")}
      toolbar={<span className="font-mono text-[10px] text-text-3">{path}</span>}
      autoscrollDefault={prefs.getLogAutoscroll()}
      onAutoscrollChange={prefs.setLogAutoscroll}
      maxHeight="360px"
    />
  );
}
