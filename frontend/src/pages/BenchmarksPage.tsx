import { useState } from "react";
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
import { useApi } from "../lib/useApi";

export function BenchmarksPage() {
  const { data, error, loading, reload } = useApi(() => api.getProblems(), []);
  const [filter, setFilter] = useState("");

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

  const totalCount = levels.reduce(
    (sum, level) => sum + level.problems.length,
    0,
  );

  return (
    <div>
      <PageHeader
        title="Benchmarks"
        description="pinned KernelBench snapshot 的题目树（GET /api/problems）。题目与启动表单（/launch）打通在下一工作包接入；清单与容差不会为过测而改动。"
        actions={
          <input
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="按 label / spec / id 过滤…"
            className="h-8 w-56 rounded-md border border-border bg-surface px-3 text-xs text-fg placeholder:text-muted/60 focus:border-primary focus:outline-none"
          />
        }
      />

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
            {data?.bench ?? "kernelbench"} · {levels.length} 个 level · 共{" "}
            {totalCount} 题
            {needle && ` · 过滤后 ${visibleLevels.length} 个 level`}
          </p>
          {visibleLevels.map((level) => (
            <Card key={level.level}>
              <CardHeader>
                <CardTitle>Level {level.level}</CardTitle>
                <Badge variant="neutral">{level.problems.length} 题</Badge>
              </CardHeader>
              <CardContent>
                <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
                  {level.problems.map((problem) => (
                    <div
                      key={problem.spec}
                      className="rounded-md border border-border bg-surface-raised px-2.5 py-2"
                      title={`spec: ${problem.spec}`}
                    >
                      <p className="truncate font-mono text-xs text-fg">
                        {problem.label}
                      </p>
                      <p className="mt-0.5 font-mono text-[11px] text-muted">
                        {problem.spec}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
              {level.level === levels.length && data?.note && (
                <CardContent className="border-t border-border pt-3">
                  <CardDescription>{data.note}</CardDescription>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
