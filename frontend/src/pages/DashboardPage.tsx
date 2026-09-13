import { Link } from "react-router-dom";
import { api } from "../api";
import type { RunSummary } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import { StageStepper } from "../components/StageStepper";
import { BudgetBar } from "../components/BudgetBar";
import { NotRunBadge } from "../components/ui/NotRunBadge";
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
  Table,
  TableWrapper,
  TBody,
  TD,
  TH,
  THead,
  TR,
} from "../components/ui";
import { parseRunIdTime, relativeTime } from "../lib/format";
import { buildLanes, primaryLane } from "../lib/lanes";
import { useApi } from "../lib/useApi";
import { MOCKS_ENABLED } from "../mocks";

function StatTile({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <Card>
      <CardContent className="py-4" title={title}>
        <p className="text-xs text-muted">{label}</p>
        <p className="mt-1 truncate font-mono text-lg font-semibold tabular-nums text-fg">
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

/** 活跃运行卡：实时阶段 stepper + 双预算条（spec §3.2） */
function ActiveRunCard({ runId }: { runId: string }) {
  const snapshot = useApi(
    () => api.getRun(runId),
    [runId],
    { pollMs: 1500 }, // 活跃 run 的 Dashboard 卡 1.5s（spec §5）
  );
  const data = snapshot.data;
  const lane = data ? primaryLane(buildLanes(data, null)) : null;

  return (
    <Card className="border-running/40">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          活跃运行
          {data && <RunStateBadge state={data.state} />}
        </CardTitle>
        <Link
          to={`/runs/${runId}`}
          className="font-mono text-xs text-primary hover:text-primary-hover"
        >
          {runId} →
        </Link>
      </CardHeader>
      <CardContent className="space-y-3">
        {snapshot.loading && !data ? (
          <LoadingBlock label="读取活跃 run 快照…" />
        ) : snapshot.error || !data ? (
          <p className="text-xs text-warning">
            活跃 run 快照读取失败：{snapshot.error?.message ?? "未知错误"}
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
              <Badge variant={data.job?.mode === "live" ? "primary" : "warning"}>
                {data.job?.mode === "live" ? "LIVE" : `${data.job?.mode ?? "—"}（演示）`}
              </Badge>
              <span className="font-mono">{data.job?.config?.problem ?? "—"}</span>
              <span className="font-mono">{data.job?.config?.backend ?? ""}</span>
            </div>
            {lane && <StageStepper stages={lane.stages} />}
            <div className="grid gap-3 sm:grid-cols-2">
              <BudgetBar label="GPU 秒" unit="gpu" budget={data.budget} />
              <BudgetBar label="Token" unit="token" budget={data.budget} />
            </div>
            {(data.in_flight?.length ?? 0) > 0 && (
              <p className="text-[11px] text-muted">
                在途动作：
                {data.in_flight.map((id) => `${id}(${data.progress[id] ?? "stage 未知"})`).join("、")}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function DashboardPage() {
  const health = useApi(() => api.getHealth(), [], { pollMs: 10_000 });
  const runs = useApi(() => api.listRuns(), [], {
    // 有活跃 run 5s，否则 15s（spec §5）
    pollMs: health.data?.active_run_id ? 5_000 : 15_000,
  });

  const runList: RunSummary[] = runs.data?.runs ?? [];
  const activeId = health.data?.active_run_id ?? runList.find((r) => r.state === "running")?.run_id ?? null;
  const completedCount = runList.filter((r) => r.state === "completed").length;
  const championCount = runList.filter((r) => r.champion).length;
  const completionRate =
    runList.length > 0 ? `${Math.round((completedCount / runList.length) * 100)}%` : "—";
  const recent = runList.slice(0, 8);

  return (
    <div className="space-y-5">
      <PageHeader
        title="总览"
        description="后端健康、GPU 占用、活跃运行与最近 runs 的一屏摘要；所有状态均来自 run 目录，不由前端推断。"
        actions={
          <div className="flex items-center gap-2">
            {MOCKS_ENABLED && <Badge variant="primary">MOCK 数据源</Badge>}
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                health.reload();
                runs.reload();
              }}
            >
              刷新
            </Button>
          </div>
        }
      />

      {health.error && (
        <ErrorPanel
          title="无法读取后端健康状态"
          error={health.error}
          onRetry={health.reload}
        />
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Run 总数" value={runs.loading && !runs.data ? "…" : String(runList.length)} />
        <StatTile
          label="完成率（completed）"
          value={completionRate}
          title="completed = 有 champion 且通过确认的 run"
        />
        <StatTile
          label="最佳加速比"
          value={<NotRunBadge what="加速比（需逐 run 读取 report/records）" />}
          title="列表端点不含 CI 数据；待 P2/P4 端点后统计。缺失不显示 0。"
        />
        <StatTile
          label="有 champion 的 runs"
          value={runs.loading && !runs.data ? "…" : String(championCount)}
          title="champion 需人工审查（candidate_trust=cooperative）"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>系统健康</CardTitle>
            {health.data ? (
              <Badge variant="success" dot>
                ok
              </Badge>
            ) : (
              <Badge variant="error">不可达</Badge>
            )}
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-muted">GPU 设备</span>
              <span className="min-w-0 truncate text-right font-mono text-fg" title={health.data?.gpu_device}>
                {health.data?.gpu_device ?? "—"}
              </span>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-muted">活跃 run</span>
              <span className="font-mono text-fg">{health.data?.active_run_id ?? "无（GPU 空闲）"}</span>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-muted">后端</span>
              <span className="font-mono text-fg">GET /api/health（10s 轮询）</span>
            </div>
            <p className="pt-1 text-[11px] leading-relaxed text-muted">
              candidate_trust = cooperative：champion 需人工审查后采信（顶栏常驻警示）。
            </p>
          </CardContent>
        </Card>

        <div>
          {activeId ? (
            <ActiveRunCard runId={activeId} />
          ) : (
            <EmptyState
              title="当前没有活跃运行"
              description="GPU 空闲。可前往「新建运行」启动一次优化；或等待 /api/health 报告活跃 run。"
              action={
                <Link
                  to="/launch"
                  className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-fg hover:bg-primary-hover"
                >
                  新建运行 →
                </Link>
              }
            />
          )}
        </div>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-fg">最近 Runs</h3>
          <Link to="/runs" className="text-xs text-primary hover:text-primary-hover">
            查看全部 →
          </Link>
        </div>
        {runs.loading && !runs.data ? (
          <LoadingBlock />
        ) : runs.error ? (
          <ErrorPanel error={runs.error} onRetry={runs.reload} />
        ) : recent.length === 0 ? (
          <EmptyState
            title="还没有任何 Run"
            description="前往「新建运行」启动第一次优化。"
          />
        ) : (
          <TableWrapper>
            <Table>
              <THead>
                <TR>
                  <TH>Run ID</TH>
                  <TH>状态</TH>
                  <TH>模式</TH>
                  <TH>题目</TH>
                  <TH>Champion</TH>
                  <TH>开始</TH>
                </TR>
              </THead>
              <TBody>
                {recent.map((run) => {
                  const startedAt =
                    typeof run.started_at === "number" && run.started_at > 0
                      ? run.started_at
                      : parseRunIdTime(run.run_id);
                  const ago = relativeTime(startedAt);
                  return (
                    <TR key={run.run_id} className="hover:bg-surface-raised/60">
                      <TD>
                        <Link
                          to={`/runs/${run.run_id}`}
                          className="font-mono text-xs text-primary hover:text-primary-hover"
                        >
                          {run.run_id}
                        </Link>
                      </TD>
                      <TD>
                        <RunStateBadge state={run.state} />
                      </TD>
                      <TD className="font-mono text-xs text-muted">
                        {run.mode === "live" ? (
                          "live"
                        ) : (
                          <span title="demo 模式，非 LIVE_MODEL">{run.mode ?? "—"}（演示）</span>
                        )}
                      </TD>
                      <TD className="max-w-[14rem] truncate font-mono text-xs">
                        {run.problem ?? "—"}
                      </TD>
                      <TD className="text-xs">
                        {run.champion ? (
                          <span className="text-champion" title="champion 需人工审查">
                            🏆 有
                          </span>
                        ) : (
                          <span className="text-text-3">—</span>
                        )}
                      </TD>
                      <TD
                        className="text-xs text-muted tabular-nums"
                        title={ago ? `run_id 编码的启动时间（${startedAt ? new Date(startedAt * 1000).toLocaleString("zh-CN", { hour12: false }) : ""}）` : "run_id 无法解析出时间"}
                      >
                        {ago ?? "—"}
                      </TD>
                    </TR>
                  );
                })}
              </TBody>
            </Table>
          </TableWrapper>
        )}
      </div>
    </div>
  );
}
