import { Link } from "react-router-dom";
import { api } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import {
  Button,
  Card,
  CardContent,
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
import { useApi } from "../lib/useApi";

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <p className="text-xs text-muted">{label}</p>
        <p className="mt-1 truncate font-mono text-lg font-semibold text-fg">
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

export function DashboardPage() {
  const health = useApi(() => api.getHealth(), [], { pollMs: 15_000 });
  const runs = useApi(() => api.listRuns(), [], { pollMs: 15_000 });
  const problems = useApi(() => api.getProblems(), []);

  const runList = runs.data?.runs ?? [];
  const problemCount =
    problems.data?.levels.reduce((sum, level) => sum + level.problems.length, 0) ??
    null;

  return (
    <div>
      <PageHeader
        title="总览"
        description="后端健康、GPU 占用与最近运行的一屏摘要；所有状态均来自 run 目录，不由前端推断。"
        actions={
          <Button variant="secondary" size="sm" onClick={() => { health.reload(); runs.reload(); }}>
            刷新
          </Button>
        }
      />

      {health.error && (
        <ErrorPanel
          className="mb-4"
          title="无法读取后端健康状态"
          error={health.error}
          onRetry={health.reload}
        />
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="GPU 设备" value={health.data?.gpu_device ?? "—"} />
        <StatCard label="活跃 Run" value={health.data?.active_run_id ?? "无"} />
        <StatCard label="Run 总数" value={String(runList.length)} />
        <StatCard
          label="KernelBench 题目"
          value={problemCount === null ? "—" : String(problemCount)}
        />
      </div>

      <div className="mt-6">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-fg">最近 Runs</h3>
          <Link
            to="/runs"
            className="text-xs text-primary hover:text-primary-hover"
          >
            查看全部 →
          </Link>
        </div>

        {runs.loading && !runs.data ? (
          <LoadingBlock />
        ) : runs.error ? (
          <ErrorPanel error={runs.error} onRetry={runs.reload} />
        ) : runList.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted">
              还没有任何 Run。可前往{" "}
              <Link to="/launch" className="text-primary hover:text-primary-hover">
                新建运行
              </Link>{" "}
              （启动表单在后续工作包接入）。
            </CardContent>
          </Card>
        ) : (
          <TableWrapper>
            <Table>
              <THead>
                <TR>
                  <TH>Run ID</TH>
                  <TH>状态</TH>
                  <TH>模式</TH>
                  <TH>题目</TH>
                  <TH>启动时间</TH>
                </TR>
              </THead>
              <TBody>
                {runList.slice(0, 5).map((run) => (
                  <TR key={run.run_id}>
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
                      {run.mode ?? "-"}
                    </TD>
                    <TD className="max-w-[16rem] truncate font-mono text-xs">
                      {run.problem ?? "-"}
                    </TD>
                    <TD className="text-xs text-muted" title="启动时间在 Run 详情的 job 记录中">
                      -
                    </TD>
                  </TR>
                ))}
              </TBody>
            </Table>
          </TableWrapper>
        )}
      </div>
    </div>
  );
}
