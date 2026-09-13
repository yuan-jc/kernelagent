import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorPanel,
  LoadingBlock,
} from "../components/ui";
import { MOCKS_ENABLED } from "../mocks";
import { useApi } from "../lib/useApi";

/**
 * Benchmark 管理（spec §3.6）：KernelBench 题目树（GET /api/problems，
 * 现有数据）+ 尚未接入的 benchmark 适配器（MKB / rk / user-bench）以
 * "Planned" 卡诚实展示 —— 端点未实现，不提供假的题目数据。
 */

/** 尚未接入 optimize 的 benchmark 库（对齐 /api/problems 的 note 字段语义） */
const PLANNED_BENCHES = [
  {
    name: "MKB",
    desc: "矩阵算子 benchmark 适配器：未接入（/api/problems 仅返回 kernelbench）。",
  },
  {
    name: "rk",
    desc: "rk（随机 kernel 集）适配器：未接入；接入后经 /api/problems 暴露题树。",
  },
  {
    name: "user-bench",
    desc: "用户自定义 benchmark 适配器：未接入；设计上经 adapter 组合接口接入，不另建平行题库。",
  },
] as const;

export function BenchmarksPage() {
  const { data, error, loading, reload } = useApi(() => api.getProblems(), []);
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set());

  const levels = data?.levels ?? [];
  const needle = filter.trim().toLowerCase();
  const visibleLevels = levels
    .map((level) => ({
      ...level,
      problems: needle
        ? level.problems.filter(
            (problem) =>
              problem.label.toLowerCase().includes(needle) ||
              problem.spec.toLowerCase().includes(needle) ||
              String(problem.id) === needle,
          )
        : level.problems,
    }))
    .filter((level) => level.problems.length > 0);

  const totalCount = levels.reduce((sum, level) => sum + level.problems.length, 0);

  function toggle(level: number) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      return next;
    });
  }

  return (
    <div>
      <PageHeader
        title="Benchmarks"
        description="已接入的 KernelBench 题目树（GET /api/problems）与适配器接入状态。清单与容差不会为过测而改动。"
        actions={
          <div className="flex items-center gap-2">
            {MOCKS_ENABLED && <Badge variant="primary">MOCK 数据源</Badge>}
            <input
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="按 label / spec / id 过滤…"
              className="h-8 w-56 rounded-md border border-border bg-surface px-3 text-xs text-fg placeholder:text-muted/60 focus:border-primary focus:outline-none"
            />
          </div>
        }
      />

      {/* 适配器接入状态卡 */}
      <div className="mb-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="border-success/30">
          <CardContent className="py-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-fg">KernelBench</p>
              <Badge variant="success">已接入</Badge>
            </div>
            <p className="mt-1 text-[11px] text-muted">
              pinned snapshot；{loading ? "…" : `${totalCount} 题 / ${levels.length} 个 level`}
            </p>
          </CardContent>
        </Card>
        {PLANNED_BENCHES.map((bench) => (
          <Card key={bench.name} className="border-dashed">
            <CardContent className="py-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-semibold text-fg">{bench.name}</p>
                <Badge variant="neutral">Planned</Badge>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-muted">{bench.desc}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {loading && !data ? (
        <LoadingBlock />
      ) : error ? (
        <ErrorPanel error={error} onRetry={reload} />
      ) : levels.length === 0 ? (
        <EmptyState
          title="未找到题目"
          description={
            data?.note ??
            "snapshot_root 下没有 KernelBench level 目录；请确认服务端启动参数。"
          }
        />
      ) : (
        <div className="space-y-4">
          <p className="text-xs text-muted">
            {data?.bench ?? "kernelbench"} · {levels.length} 个 level · 共 {totalCount} 题
            {needle && ` · 过滤后 ${visibleLevels.length} 个 level`}
          </p>
          {visibleLevels.map((level) => {
            const isCollapsed = collapsed.has(level.level) && !needle;
            return (
              <Card key={level.level}>
                <CardHeader
                  className="cursor-pointer select-none"
                  onClick={() => toggle(level.level)}
                  title={isCollapsed ? "展开" : "折叠"}
                >
                  <CardTitle className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-text-3">
                      {isCollapsed ? "▸" : "▾"}
                    </span>
                    Level {level.level}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Badge variant="neutral">{level.problems.length} 题</Badge>
                    {needle && <Badge variant="primary">过滤中</Badge>}
                  </div>
                </CardHeader>
                {!isCollapsed && (
                  <CardContent>
                    <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
                      {level.problems.map((problem) => (
                        <div
                          key={problem.spec}
                          className="flex items-center justify-between gap-2 rounded-md border border-border bg-surface-raised px-2.5 py-2"
                          title={`spec: ${problem.spec}`}
                        >
                          <div className="min-w-0">
                            <p className="truncate font-mono text-xs text-fg">{problem.label}</p>
                            <p className="mt-0.5 font-mono text-[11px] text-muted">{problem.spec}</p>
                          </div>
                          <Link
                            to={`/launch?problem=${encodeURIComponent(problem.spec)}`}
                            className="shrink-0 rounded border border-primary/40 px-1.5 py-0.5 text-[11px] text-primary hover:bg-primary-soft"
                            title="在新建运行页预填该题目"
                          >
                            去运行
                          </Link>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                )}
                {level.level === levels.length && data?.note && (
                  <CardContent className="border-t border-border pt-3">
                    <CardDescription>{data.note}</CardDescription>
                  </CardContent>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
