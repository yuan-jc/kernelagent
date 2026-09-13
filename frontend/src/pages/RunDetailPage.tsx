import { Link, useParams } from "react-router-dom";
import { useMemo, useState } from "react";
import { api } from "../api";
import type { JournalEntry, RunSnapshot } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import { StageStepper } from "../components/StageStepper";
import { StageSwimlanes } from "../components/StageSwimlanes";
import { BudgetBar } from "../components/BudgetBar";
import { CandidateTable } from "../components/CandidateTable";
import { ChampionCard } from "../components/ChampionCard";
import { LogViewer } from "../components/LogViewer";
import { CodeView } from "../components/CodeView";
import { CopyButton } from "../components/CopyButton";
import { TabBar, useTabParam } from "../components/Tabs";
import { RunErrorBanner, TerminalNote } from "./rundetail/RunBanners";
import {
  useJournal,
  useRecord,
  useReportFallback,
  useRunReport,
  useWorkspace,
  useWorkspaceFile,
} from "../hooks/useRunArtifacts";
import { buildLanes, primaryLane } from "../lib/lanes";
import { journalToLines } from "../lib/journalLines";
import { isTerminalRunState, mergeRunWithReport, shouldKeepPolling } from "../lib/states";
import { prefs } from "../lib/prefs";
import { ApiError } from "../api";
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
} from "../components/ui";
import { formatDateTime, relativeTime } from "../lib/format";
import { useApi } from "../lib/useApi";

const TAB_KEYS = ["overview", "candidates", "logs", "evidence"] as const;

function KeyValue({ label, value, mono = true }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5 py-1.5 sm:flex-row sm:items-baseline sm:gap-3">
      <span className="w-40 shrink-0 text-xs text-muted">{label}</span>
      <span className={`min-w-0 break-words text-xs ${mono ? "font-mono" : ""} text-fg`}>
        {value ?? "—"}
      </span>
    </div>
  );
}

function unknownOr(value: string | null | undefined): React.ReactNode {
  return value ? value : <span className="text-text-3">—</span>;
}

// -- 概览 ---------------------------------------------------------------------

function OverviewTab({
  snapshot,
  running,
  journalEntries,
  journalMode,
  baselineRecord,
}: {
  snapshot: RunSnapshot;
  running: boolean;
  journalEntries: JournalEntry[] | null;
  journalMode: string | null;
  baselineRecord: ReturnType<typeof useRecord> | null;
}) {
  const lanes = useMemo(
    () =>
      buildLanes(
        snapshot,
        journalEntries,
        baselineRecord?.data && !baselineRecord.unavailable ? baselineRecord.data : null,
      ),
    [snapshot, journalEntries, baselineRecord],
  );
  const primary = primaryLane(lanes);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>预算（journal 结算台账）</CardTitle>
            {running && (
              <Badge variant="running" dot>
                实时
              </Badge>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            <BudgetBar label="GPU 秒" unit="gpu" budget={snapshot.budget} />
            <BudgetBar label="Token" unit="token" budget={snapshot.budget} />
            <p className="text-[11px] leading-relaxed text-muted">
              settled = 已结算；reserved = 在途预留（未结算）。报告未生成时数值来自
              journal 的持久化结算；缺失显示 —，不按零处理。
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>任务配置（job.json，不含 key）</CardTitle>
          </CardHeader>
          <CardContent className="divide-y divide-border/60">
            <KeyValue label="模式" value={unknownOr(snapshot.job?.mode)} />
            <KeyValue label="问题 spec" value={unknownOr(snapshot.job?.config?.problem)} />
            <KeyValue label="backend" value={unknownOr(snapshot.job?.config?.backend)} />
            <KeyValue label="模型" value={unknownOr(snapshot.job?.config?.model_id)} />
            <KeyValue label="base_url" value={unknownOr(snapshot.job?.config?.base_url)} />
            <KeyValue label="候选数 / 修复轮上限" value={
              snapshot.job?.config
                ? `${snapshot.job.config.max_candidates ?? "—"} / ${snapshot.job.config.max_repair_rounds ?? "—"}`
                : "—"
            } />
            <KeyValue label="开始时间" value={
              snapshot.job?.started_at
                ? `${formatDateTime(snapshot.job.started_at)}（${relativeTime(snapshot.job.started_at) ?? "—"}）`
                : "—"
            } />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>候选流水（泳道）</CardTitle>
          <span className="text-[11px] text-muted">
            {journalEntries
              ? journalMode === "live-incremental"
                ? "由 journal 事件流驱动（P3 增量）"
                : "由 journal 事件流驱动（全量兜底）"
              : "由快照 in_flight/progress 驱动（P3 未上线）"}
            ；单阶段耗时无时间戳依据显示 "—"
          </span>
        </CardHeader>
        <CardContent className="space-y-4">
          {lanes.length === 0 ? (
            <EmptyState
              title="暂无候选活动"
              description="运行中的实时进度来自快照 in_flight/progress；终态候选来自 report.json。两者都为空说明该 run 尚无候选动作。"
            />
          ) : (
            <StageSwimlanes lanes={lanes} />
          )}
          {primary && (
            <div className="rounded-lg border border-border bg-surface-2/40 px-3 pt-3 pb-2">
              <div className="mb-2 flex items-center gap-2">
                <span className="font-mono text-xs text-muted">
                  当前/最后活跃候选：
                </span>
                <span className="font-mono text-xs text-fg">{primary.id}</span>
              </div>
              <StageStepper stages={primary.stages} />
            </div>
          )}
        </CardContent>
      </Card>

      <ChampionCard snapshot={snapshot} />
    </div>
  );
}

// -- 候选对比 ------------------------------------------------------------------

function CandidatesTab({
  snapshot,
  baseline,
}: {
  snapshot: RunSnapshot;
  baseline: ReturnType<typeof useRecord>;
}) {
  // baseline 记录需要 P4（records/baseline-eager）；404 时诚实降级。
  // 记录由页面级 hook 统一获取（概览泳道与对比表共用，避免重复请求）。
  const baselineRecord =
    baseline.data && !baseline.unavailable
      ? {
          candidate: "baseline-eager",
          stage: baseline.data.stage,
          status: baseline.data.status,
          detail: baseline.data.detail,
          ratio_ci_95: undefined,
        }
      : null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>候选对比（champion vs baseline vs 全部）</CardTitle>
          <span className="text-[11px] text-muted">
            终态候选来自 report.json；加速比 CI 缺失标 NOT_RUN，不显示 0×
          </span>
        </CardHeader>
        <CardContent className="space-y-2">
          <CandidateTable
            snapshot={snapshot}
            baseline={baselineRecord}
            championSha={snapshot.champion?.candidate_sha256}
          />
          {baseline.unavailable && !baseline.unavailableReason?.includes("no record") && (
            <p className="text-[11px] leading-relaxed text-muted">
              {baseline.unavailableReason}
            </p>
          )}        </CardContent>
      </Card>
      <ChampionCard snapshot={snapshot} />
    </div>
  );
}

// -- 日志与产物 ----------------------------------------------------------------

function LogsTab({
  snapshot,
  running,
  journal,
}: {
  snapshot: RunSnapshot;
  running: boolean;
  journal: ReturnType<typeof useJournal>;
}) {
  const [view, setView] = useState<"journal" | "workspace">("journal");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const workspace = useWorkspace(snapshot.run_id);
  const file = useWorkspaceFile(snapshot.run_id, view === "workspace" ? selectedFile : null);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="flex rounded-md border border-border p-0.5">
          {(
            [
              { key: "journal", label: "journal 事件流" },
              { key: "workspace", label: "workspace 产物" },
            ] as const
          ).map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setView(item.key)}
              className={`rounded px-2.5 py-1 text-xs transition-colors ${
                view === item.key ? "bg-accent-subtle font-medium text-accent" : "text-muted hover:text-fg"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
        {running && view === "journal" && journal.source === "live" && (
          <Badge variant="running" dot>
            实时
          </Badge>
        )}
      </div>

      {view === "journal" ? (
        journal.source === "unavailable" ? (
          <EmptyState
            title="journal 事件流不可用"
            description={journal.unavailableReason ?? "P3 端点未实现。"}
          />
        ) : journal.error ? (
          <ErrorPanel title="journal 读取失败" error={journal.error} onRetry={journal.reload} />
        ) : (
          <LogViewer
            lines={journalToLines(journal.entries)}
            emptyText="journal 尚无事件（该 run 可能没有 journal.jsonl）。"
            toolbar={
              <Badge variant={journal.mode === "live-incremental" ? "success" : "warning"}>
                {journal.mode === "live-incremental"
                  ? "P3 增量"
                  : journal.mode === "live-full"
                    ? "全量兜底"
                    : "加载中"}
              </Badge>
            }
            autoscrollDefault={prefs.getLogAutoscroll()}
            onAutoscrollChange={prefs.setLogAutoscroll}
          />
        )
      ) : workspace.unavailable ? (
        <EmptyState
          title="workspace 产物不可用"
          description={workspace.unavailableReason ?? "P5 端点未实现。"}
        />
      ) : workspace.error ? (
        <ErrorPanel title="workspace 读取失败" error={workspace.error} onRetry={workspace.reload} />
      ) : (workspace.data?.entries.length ?? 0) === 0 ? (
        <EmptyState
          title="工作区为空"
          description="容器/计时产物目录尚未生成（通常在 evaluate/timing 阶段之后出现）。"
        />
      ) : (
        <div className="grid gap-3 lg:grid-cols-[260px_1fr]">
          <div className="space-y-2.5">
            {workspace.data!.entries.map((entry) => (
              <div key={entry.path} className="rounded-md border border-border bg-surface p-2">
                <p className="truncate font-mono text-[11px] text-fg" title={entry.path}>
                  {entry.type === "dir" ? "▸ " : ""}
                  {entry.path}
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {entry.files?.map((f) => {
                    const full = `${entry.path}/${f}`;
                    return (
                      <button
                        key={full}
                        type="button"
                        onClick={() => setSelectedFile(full)}
                        className={`rounded border px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                          selectedFile === full
                            ? "border-accent bg-accent-subtle text-accent"
                            : "border-border bg-surface-2 text-muted hover:text-fg"
                        }`}
                      >
                        {f}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
            <p className="text-[11px] leading-relaxed text-muted">
              只读视图（P5 提案）；后端负责 path 白名单校验，前端仅按服务端返回渲染。
            </p>
          </div>
          <div>
            {!selectedFile ? (
              <EmptyState title="选择左侧文件查看内容" description="stdout/stderr、候选源码与计时驱动脚本（只读）。" />
            ) : file.loading ? (
              <LoadingBlock />
            ) : file.unavailable ? (
              <EmptyState title="文件不可读" description={file.unavailableReason ?? undefined} />
            ) : file.error ? (
              <ErrorPanel title="文件读取失败" error={file.error} onRetry={file.reload} />
            ) : (
              <CodeView content={file.data!.content} path={file.data!.path} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// -- 证据 ----------------------------------------------------------------------

function EvidenceTab({ snapshot }: { snapshot: RunSnapshot }) {
  const report = useRunReport(snapshot.run_id);

  return (
    <div className="space-y-4">
      {report.unavailable && (
        <p className="rounded-md border border-border bg-surface px-3 py-2 text-[11px] leading-relaxed text-muted">
          {report.unavailableReason}
        </p>
      )}
      {report.error && <ErrorPanel title="report 读取失败" error={report.error} onRetry={report.reload} />}

      {report.data && !report.unavailable && (
        <Card>
          <CardHeader>
            <CardTitle>report.json 身份链（P2）</CardTitle>
            <Badge variant="success">P2 已上线</Badge>
          </CardHeader>
          <CardContent className="divide-y divide-border/60">
            <KeyValue label="protocol" value={unknownOr(report.data.protocol)} />
            <KeyValue label="commit（代码身份）" value={
              report.data.commit ? (
                <span className="inline-flex items-center gap-2">
                  {report.data.commit}
                  <CopyButton text={report.data.commit} label="复制" />
                </span>
              ) : "—"
            } />
            <KeyValue label="problem_sha256" value={unknownOr(report.data.problem_sha256)} />
            <KeyValue label="problem_path" value={unknownOr(report.data.problem_path)} />
            <KeyValue label="gpu_device（环境身份）" value={unknownOr(report.data.gpu_device)} />
            <KeyValue label="journal_entries" value={report.data.journal_entries ?? "—"} />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>快照证据（run_snapshot 持久化字段）</CardTitle>
        </CardHeader>
        <CardContent className="divide-y divide-border/60">
          <KeyValue label="run_id" value={snapshot.run_id} />
          <KeyValue label="state（服务端枚举）" value={snapshot.state} />
          <KeyValue label="candidate_trust" value={unknownOr(snapshot.candidate_trust)} />
          <KeyValue
            label="adversarially_secure"
            value={snapshot.adversarially_secure === undefined ? "未报告" : String(snapshot.adversarially_secure)}
          />
          <KeyValue
            label="champion sha256"
            value={snapshot.champion?.candidate_sha256 ?? "—"}
          />
          <KeyValue label="job.mode" value={unknownOr(snapshot.job?.mode)} />
          <KeyValue label="job.config" value={
            snapshot.job?.config ? (
              <pre className="max-h-40 overflow-auto rounded border border-border bg-log-bg p-2 text-[11px] text-muted">
                {JSON.stringify(snapshot.job.config, null, 2)}
              </pre>
            ) : "—"
          } />
        </CardContent>
      </Card>

      <p className="text-[11px] leading-relaxed text-muted">
        晋升至少绑定代码/环境/协议身份、正确性与独立性能确认；本页只呈现 run
        目录与服务端快照的持久化事实，不做任何前端推断。
      </p>
    </div>
  );
}

// -- 页面 ----------------------------------------------------------------------

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const [tab, setTab] = useTabParam("overview", [...TAB_KEYS]);
  const { data, error, loading, degraded, reload } = useApi(
    () => api.getRun(runId),
    [runId],
    // spec §5：running 时按本地偏好轮询（默认 1.5s）；终态停止。
    // "unknown" 是服务端"暂不可知"（job 线程收尾/启动窗口），不能停：
    // 否则页面会永远停在「未知 + 预算 NOT_RUN」，下一拍永远等不到。
    {
      pollMs: prefs.getPollInterval(),
      shouldPoll: (snapshot) =>
        snapshot === null || shouldKeepPolling((snapshot as RunSnapshot).state),
    },
  );
  // 快照 state=unknown 时按 P2 report 兜底（终态已落盘的场合当拍收敛）；
  // 非 unknown 时不发请求。trigger 用快照对象本身：每轮轮询新快照都会重试。
  const fallback = useReportFallback(runId, data?.state === "unknown" ? data : null);
  // 合并后的展示快照（仅 unknown 分支会被 report 补齐；其余原样透传）
  const merged = data ? mergeRunWithReport(data, fallback.report) : null;
  const convergedFromReport = !!data && !!fallback.report && merged!.state !== data.state;
  // baseline 记录（P4）：概览泳道终态徽章与候选对比表共用
  const baseline = useRecord(runId, "baseline-eager");
  const running = merged ? !isTerminalRunState(merged.state) : true;
  // journal 事件流：泳道（概览）与日志 tab 共用；终态后停止轮询
  const journal = useJournal(runId, { pollMs: 1500, enabled: running });

  if (!runId) {
    return <ErrorPanel title="缺少 run id" message="路径中未提供 run id。" />;
  }

  if (loading && !data) return <LoadingBlock />;

  if (error && !merged) {
    const notFound = error instanceof ApiError && error.status === 404;
    return (
      <div>
        <PageHeader title={<span className="font-mono">{runId}</span>} />
        <ErrorPanel
          title={notFound ? `Run ${runId} 不存在` : `无法读取 run ${runId}`}
          error={error}
          onRetry={reload}
        />
      </div>
    );
  }

  if (!merged) return null;

  const snapshot = merged;
  const startedAgo = snapshot.job?.started_at ? relativeTime(snapshot.job.started_at) : null;

  return (
    <div className="space-y-4">
      <PageHeader
        title={
          <span className="inline-flex flex-wrap items-center gap-3">
            <Link to="/runs" className="text-sm text-muted hover:text-fg">
              ←
            </Link>
            <span className="font-mono">{snapshot.run_id}</span>
            <RunStateBadge state={snapshot.state} />
          </span>
        }
        description={
          <>
            {[snapshot.job?.mode, snapshot.job?.config?.problem, snapshot.job?.config?.backend]
              .filter(Boolean)
              .join(" · ") || "配置未报告"}
            {startedAgo && ` · 开始 ${startedAgo}`}
          </>
        }
        actions={
          <>
            <Link
              to={`/runs/${runId}/profile`}
              className="text-xs text-primary hover:text-primary-hover"
            >
              Profile →
            </Link>
            <Button variant="secondary" size="sm" onClick={reload}>
              刷新
            </Button>
          </>
        }
      />

      {degraded && running && (
        <p className="rounded-md border border-warning/30 bg-warning-soft px-3 py-2 text-xs text-warn">
          连接失败，退避重试中（每 10s 一次）；显示的可能是过期快照。
        </p>
      )}

      {(snapshot.state === "infra_error" || snapshot.state === "journal_corrupt") && (
        <RunErrorBanner snapshot={snapshot} />
      )}
      <TerminalNote snapshot={snapshot} />
      {convergedFromReport && (
        <p className="rounded-md border border-border bg-surface px-3 py-2 text-[11px] leading-relaxed text-muted">
          快照 state=unknown（run 目录在 job 线程收尾窗口内暂不可知）；上方状态按同源的
          report.json 终态兜底显示，原始快照字段见「证据」页。轮询将继续，直到服务端
          快照给出终态。
        </p>
      )}

      <TabBar
        tabs={[
          { key: "overview", label: "概览" },
          { key: "candidates", label: "候选对比" },
          {
            key: "logs",
            label: "日志与产物",
            badge: running ? (
              <Badge variant="running" dot>
                live
              </Badge>
            ) : undefined,
          },
          { key: "evidence", label: "证据" },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === "overview" && (
        <OverviewTab
          snapshot={snapshot}
          running={running}
          journalEntries={journal.source === "live" ? journal.entries : null}
          journalMode={journal.mode}
          baselineRecord={baseline}
        />
      )}
      {tab === "candidates" && <CandidatesTab snapshot={snapshot} baseline={baseline} />}
      {tab === "logs" && <LogsTab snapshot={snapshot} running={running} journal={journal} />}
      {/* 证据页展示服务端原始快照（不吞并 report 兜底），保持可审计 */}
      {tab === "evidence" && <EvidenceTab snapshot={data!} />}
    </div>
  );
}
