import type { ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { CandidateRecord, RunBudget, RunSnapshot } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import { candidateStatusMeta, STAGES } from "../components/statusMap";
import { isTerminalRunState } from "../lib/states";
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
  Table,
  TableWrapper,
  TBody,
  TD,
  TH,
  THead,
  TR,
} from "../components/ui";
import {
  formatCount,
  formatDateTime,
  formatDuration,
  shortHash,
} from "../lib/format";
import { useApi } from "../lib/useApi";

function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-1.5 sm:flex-row sm:items-baseline sm:gap-3">
      <span className="w-40 shrink-0 text-xs text-muted">{label}</span>
      <span className="min-w-0 font-mono text-xs break-words text-fg">
        {value ?? "-"}
      </span>
    </div>
  );
}

function BudgetBar({
  label,
  settled,
  reserved,
  limit,
  format,
}: {
  label: string;
  settled?: number;
  reserved?: number;
  limit?: number | null;
  format: (v: number) => string;
}) {
  const hasLimit = typeof limit === "number" && limit > 0;
  const settledValue = typeof settled === "number" ? settled : 0;
  const reservedValue = typeof reserved === "number" ? reserved : 0;
  const settledPct = hasLimit ? Math.min((settledValue / limit) * 100, 100) : 0;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="text-muted">{label}</span>
        <span className="font-mono text-fg">
          {format(settledValue)}
          {reservedValue > 0 ? ` (+${format(reservedValue)} 预留)` : ""}
          {hasLimit ? ` / ${format(limit)}` : " / 无上限记录"}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-raised">
        <div className="h-full bg-primary" style={{ width: `${settledPct}%` }} />
      </div>
      {reservedValue > 0 && (
        <p className="mt-1 text-[11px] text-muted">
          另有 {format(reservedValue)} 在途预留（未结算）
        </p>
      )}
    </div>
  );
}

function budgetView(budget: RunBudget | null) {
  return {
    gpuLimit: budget?.gpu_seconds_limit ?? null,
    tokenLimit: budget?.tokens_limit ?? null,
    gpuSettled: budget?.settled_gpu_seconds,
    tokenSettled: budget?.settled_tokens,
    gpuReserved: budget?.reserved_gpu_seconds,
    tokenReserved: budget?.reserved_tokens,
  };
}

function CandidateRow({ candidate }: { candidate: CandidateRecord }) {
  return (
    <TR>
      <TD className="font-mono text-xs">{candidate.candidate ?? "-"}</TD>
      <TD className="font-mono text-xs text-muted">
        {shortHash(candidate.candidate_sha256)}
      </TD>
      <TD>
        <Badge variant={candidateStatusMeta(candidate.status).variant}>
          {candidateStatusMeta(candidate.status).label}
        </Badge>
      </TD>
      <TD className="font-mono text-xs text-muted">{candidate.stage ?? "-"}</TD>
      <TD className="font-mono text-xs">
        {candidate.ratio_ci_95 ? (
          `[${candidate.ratio_ci_95[0].toFixed(3)}, ${candidate.ratio_ci_95[1].toFixed(3)}]`
        ) : (
          <NotRunBadge what="速度比 95% CI（未到 confirm/timing 阶段）" />
        )}
      </TD>
      <TD
        className="max-w-[18rem] truncate text-xs text-muted"
        title={candidate.detail ?? undefined}
      >
        {candidate.detail ?? "-"}
      </TD>
    </TR>
  );
}

/** 终态解释卡（spec §4.3.3）：no_improvement/budget_exhausted 是合法结果 */
function TerminalNote({ state }: { state: RunSnapshot["state"] }) {
  if (state === "no_improvement") {
    return (
      <Card className="border-muted-s/30 bg-neutral-soft">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium">无改进（合法结果）：</span>
          未找到超过基线的候选，baseline 保留为 champion。这不是错误。
        </CardContent>
      </Card>
    );
  }
  if (state === "budget_exhausted") {
    return (
      <Card className="border-warning/30 bg-warning-soft">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium">预算耗尽（合法结果）：</span>
          GPU 秒或 token 预算用尽而终止；缺失的阶段指标标为 NOT_RUN，不按零处理。
        </CardContent>
      </Card>
    );
  }
  if (state === "completed") {
    return (
      <Card className="border-champion/30 bg-warning-soft">
        <CardContent className="text-xs leading-relaxed text-fg/90">
          <span className="font-medium text-champion">
            candidate_trust = cooperative：
          </span>
          champion 需人工审查后方可采信；本页不将完成状态渲染为“已验证可信”。
        </CardContent>
      </Card>
    );
  }
  return null;
}

function SnapshotView({ data }: { data: RunSnapshot }) {
  const config = data.job?.config;
  const b = budgetView(data.budget);
  const inFlight = data.in_flight ?? [];
  const candidates = data.candidates ?? [];

  return (
    <div className="space-y-5">
      <TerminalNote state={data.state} />
      {data.error_tail && (
        <ErrorPanel
          title="error.txt（基础设施/模型错误尾部）"
          message="服务端未虚构状态：该内容是 run 目录中 error.txt 的最后 1500 字符。"
          details={data.error_tail}
        />
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>任务配置</CardTitle>
            <span className="font-mono text-xs text-muted">
              {data.job?.mode ?? "-"} · 开始于 {formatDateTime(data.job?.started_at)}
            </span>
          </CardHeader>
          <CardContent className="divide-y divide-border/60">
            <KeyValue label="问题 spec" value={config?.problem} />
            <KeyValue label="backend" value={config?.backend} />
            <KeyValue label="模型" value={config?.model_id} />
            <KeyValue label="base_url" value={config?.base_url || "-"} />
            <KeyValue label="GPU 设备" value={config?.gpu_device} />
            <KeyValue
              label="候选数上限"
              value={config?.max_candidates?.toString()}
            />
            <KeyValue
              label="修复轮数上限"
              value={config?.max_repair_rounds?.toString()}
            />
            <KeyValue
              label="candidate_trust"
              value={data.candidate_trust ?? "未报告"}
            />
            <KeyValue
              label="adversarially_secure"
              value={
                data.adversarially_secure === undefined
                  ? "未报告"
                  : data.adversarially_secure
                    ? "true"
                    : "false"
              }
            />
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>预算（journal 结算值）</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <BudgetBar
                label="GPU 秒"
                settled={b.gpuSettled}
                reserved={b.gpuReserved}
                limit={b.gpuLimit}
                format={(v) => formatDuration(v)}
              />
              <BudgetBar
                label="Token"
                settled={b.tokenSettled}
                reserved={b.tokenReserved}
                limit={b.tokenLimit}
                format={(v) => formatCount(v)}
              />
              <p className="text-[11px] leading-relaxed text-muted">
                缺失指标不按零处理：报告未生成时预算来自 journal 的持久化结算。
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>进行中动作</CardTitle>
              <Badge variant={inFlight.length > 0 ? "running" : "neutral"} dot>
                {inFlight.length > 0 ? `${inFlight.length} 个` : "无"}
              </Badge>
            </CardHeader>
            <CardContent>
              {inFlight.length === 0 ? (
                <p className="text-xs text-muted">当前没有在途动作。</p>
              ) : (
                <ul className="space-y-2">
                  {inFlight.map((actionId) => (
                    <li
                      key={actionId}
                      className="flex items-center justify-between gap-2 rounded-md border border-border bg-surface-raised px-3 py-2"
                    >
                      <span className="truncate font-mono text-xs text-fg">
                        {actionId}
                      </span>
                      <span className="font-mono text-[11px] text-muted">
                        {data.progress[actionId] ?? "stage 未知"}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>候选记录</CardTitle>
          <span className="text-xs text-muted">
            {data.champion
              ? `champion: ${shortHash(data.champion.candidate_sha256)}`
              : "暂无 champion"}
          </span>
        </CardHeader>
        <CardContent>
          {candidates.length === 0 ? (
            <EmptyState
              title="尚无候选记录"
              description="候选评估结果会出现在 run 目录的 report.json 中；缺失不代表零分。"
            />
          ) : (
            <TableWrapper>
              <Table className="bg-surface">
                <THead>
                  <TR>
                    <TH>候选</TH>
                    <TH>sha256</TH>
                    <TH>状态</TH>
                    <TH>阶段</TH>
                    <TH>速度比 95% CI</TH>
                    <TH>详情</TH>
                  </TR>
                </THead>
                <TBody>
                  {candidates.map((candidate, index) => (
                    <CandidateRow
                      key={candidate.candidate_sha256 ?? candidate.candidate ?? index}
                      candidate={candidate}
                    />
                  ))}
                </TBody>
              </Table>
            </TableWrapper>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>阶段时间线</CardTitle>
          <Badge variant="neutral">占位</Badge>
        </CardHeader>
        <CardContent>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {STAGES.map((stage) => (
              <Badge key={stage.key} variant="neutral">
                <span className="font-mono text-[11px]">{stage.key}</span>
                <span className="text-[11px]">{stage.label}</span>
              </Badge>
            ))}
          </div>
          <p className="text-xs leading-relaxed text-muted">
            按 journal.jsonl 事件流渲染的六阶段 stepper / 候选泳道图将在后续
            工作包接入（含 provider 失败的 generation→generate 归并）；当前仅
            展示服务端快照中的 in-flight 动作与其最近上报的 stage。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const { data, error, loading, reload } = useApi(
    () => api.getRun(runId),
    [runId],
    // spec §5：running 时 1.5s 轮询；终态（含 journal_corrupt/unknown）停止
    {
      pollMs: 1500,
      shouldPoll: (snapshot) =>
        snapshot === null || !isTerminalRunState(snapshot.state),
    },
  );

  if (!runId) {
    return <ErrorPanel title="缺少 run id" message="路径中未提供 run id。" />;
  }

  if (loading && !data) return <LoadingBlock />;

  if (error) {
    return (
      <div>
        <PageHeader title={<span className="font-mono">{runId}</span>} />
        <ErrorPanel title={`无法读取 run ${runId}`} error={error} onRetry={reload} />
      </div>
    );
  }

  if (!data) return null;

  return (
    <div>
      <PageHeader
        title={
          <span className="inline-flex items-center gap-3">
            <span className="font-mono">{data.run_id}</span>
            <RunStateBadge state={data.state} />
          </span>
        }
        description={data.job?.config?.problem ?? undefined}
        actions={
          <>
            <Link to="/runs" className="text-xs text-muted hover:text-fg">
              ← 返回列表
            </Link>
            <Link
              to={`/runs/${runId}/profile`}
              className="text-xs text-primary hover:text-primary-hover"
            >
              Profile（建设中）→
            </Link>
            <Button variant="secondary" size="sm" onClick={reload}>
              刷新
            </Button>
          </>
        }
      />
      <SnapshotView data={data} />
    </div>
  );
}
