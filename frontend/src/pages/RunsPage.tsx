import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { RunMode, RunState } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import { IconInbox } from "../components/icons";
import { RUN_STATES } from "../api";
import {
  Button,
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
import { MOCKS_ENABLED } from "../mocks";
import { parseRunIdTime, relativeTime } from "../lib/format";
import { prefs } from "../lib/prefs";
import { useApi } from "../lib/useApi";

/**
 * Runs 列表（spec §3.3）：筛选（状态/模式/关键字，localStorage 持久化）
 * + 状态徽章 + 客户端分页。时间列优先用 /api/runs 的 started_at（C3 后端），
 * 缺失时回退 run_id 编码时间，再缺失显示 "—"，不编造。
 */

const PAGE_SIZE = 15;

const MODES: RunMode[] = ["live", "demo-correct", "demo-wrong"];

interface FilterState {
  state: "all" | RunState;
  mode: "all" | RunMode;
  query: string;
  page: number;
}

function loadFilter(): FilterState {
  return prefs.getRunsFilter<FilterState>(
    { state: "all", mode: "all", query: "", page: 0 },
    (raw) => {
      try {
        const obj = JSON.parse(raw) as Partial<FilterState>;
        if (typeof obj !== "object" || obj === null) return null;
        return {
          state: obj.state ?? "all",
          mode: obj.mode ?? "all",
          query: typeof obj.query === "string" ? obj.query : "",
          page: 0,
        };
      } catch {
        return null;
      }
    },
  );
}

const selectClass =
  "h-8 rounded-md border border-border bg-surface px-2 text-xs text-fg focus:border-primary focus:outline-none";

export function RunsPage() {
  const initial = useMemo(() => loadFilter(), []);
  const [filter, setFilter] = useState<FilterState>(initial);
  const { data, error, loading, degraded, reload } = useApi(() => api.listRuns(), [], {
    pollMs: 15_000,
  });

  function update(patch: Partial<FilterState>) {
    setFilter((prev) => {
      const next = { ...prev, ...patch };
      prefs.setRunsFilter({ state: next.state, mode: next.mode, query: next.query });
      return next;
    });
  }

  const runs = data?.runs ?? [];
  const needle = filter.query.trim().toLowerCase();
  const filtered = runs.filter((run) => {
    if (filter.state !== "all" && run.state !== filter.state) return false;
    if (filter.mode !== "all" && run.mode !== filter.mode) return false;
    if (
      needle &&
      !`${run.run_id} ${run.problem ?? ""} ${run.mode ?? ""}`.toLowerCase().includes(needle)
    ) {
      return false;
    }
    return true;
  });

  const pageCount = Math.max(Math.ceil(filtered.length / PAGE_SIZE), 1);
  const page = Math.min(filter.page, pageCount - 1);
  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div>
      <PageHeader
        title="Runs"
        description="历史运行列表（新在前）。候选不决定自身正确性；状态与结果一律来自服务端持久化的 run 目录。"
        actions={
          <div className="flex items-center gap-2">
            {MOCKS_ENABLED && (
              <span className="text-[11px] text-accent">MOCK 数据源</span>
            )}
            {degraded && (
              <span className="text-[11px] text-warning">连接失败，退避重试中…</span>
            )}
            <Button variant="secondary" size="sm" onClick={reload}>
              刷新
            </Button>
          </div>
        }
      />

      {/* 筛选条 */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <select
          aria-label="按状态筛选"
          className={selectClass}
          value={filter.state}
          onChange={(e) => update({ state: e.target.value as FilterState["state"], page: 0 })}
        >
          <option value="all">状态：全部</option>
          {RUN_STATES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select
          aria-label="按模式筛选"
          className={selectClass}
          value={filter.mode}
          onChange={(e) => update({ mode: e.target.value as FilterState["mode"], page: 0 })}
        >
          <option value="all">模式：全部</option>
          {MODES.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <input
          type="search"
          value={filter.query}
          onChange={(e) => update({ query: e.target.value, page: 0 })}
          placeholder="按 run_id / 题目搜索…"
          className="h-8 w-52 rounded-md border border-border bg-surface px-2.5 text-xs text-fg placeholder:text-muted/60 focus:border-primary focus:outline-none"
        />
        <span className="text-[11px] text-text-3">
          {filtered.length} / {runs.length} 条
        </span>
      </div>

      {loading && !data ? (
        <LoadingBlock />
      ) : error ? (
        <ErrorPanel error={error} onRetry={reload} />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<IconInbox className="h-8 w-8" />}
          title={runs.length === 0 ? "还没有任何 Run" : "没有匹配筛选条件的 Run"}
          description={
            runs.length === 0
              ? "可前往「新建运行」启动第一次优化（POST /api/runs）。"
              : "调整上方筛选条件后重试。"
          }
        />
      ) : (
        <>
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
                  <TH />
                </TR>
              </THead>
              <TBody>
                {pageRows.map((run) => {
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
                      <TD className="max-w-[16rem] truncate font-mono text-xs">
                        {run.problem ?? "—"}
                      </TD>
                      <TD className="font-mono text-xs">
                        {run.champion ? (
                          <span className="text-champion" title={run.champion}>
                            🏆 有
                          </span>
                        ) : (
                          <span className="text-text-3">—</span>
                        )}
                      </TD>
                      <TD
                        className="text-xs text-muted tabular-nums"
                        title={
                          ago
                            ? startedAt
                              ? `启动时间 ${new Date(startedAt * 1000).toLocaleString("zh-CN", { hour12: false })}`
                              : "run_id 编码的启动时间"
                            : "无启动时间数据，不编造"
                        }
                      >
                        {ago ?? "—"}
                      </TD>
                      <TD>
                        <Link
                          to={`/runs/${run.run_id}`}
                          className="text-xs text-primary hover:text-primary-hover"
                        >
                          详情 →
                        </Link>
                      </TD>
                    </TR>
                  );
                })}
              </TBody>
            </Table>
          </TableWrapper>
          {pageCount > 1 && (
            <div className="mt-3 flex items-center justify-between text-xs text-muted">
              <span>
                第 {page + 1} / {pageCount} 页（每页 {PAGE_SIZE} 条）
              </span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={page === 0}
                  onClick={() => update({ page: page - 1 })}
                >
                  上一页
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={page >= pageCount - 1}
                  onClick={() => update({ page: page + 1 })}
                >
                  下一页
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
