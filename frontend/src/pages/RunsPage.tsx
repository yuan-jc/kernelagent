import { Link } from "react-router-dom";
import { api } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { RunStateBadge } from "../components/RunStateBadge";
import { IconInbox } from "../components/icons";
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
import { shortHash } from "../lib/format";
import { useApi } from "../lib/useApi";

export function RunsPage() {
  const { data, error, loading, reload } = useApi(() => api.listRuns(), [], {
    pollMs: 15_000,
  });
  const runs = data?.runs ?? [];

  return (
    <div>
      <PageHeader
        title="Runs"
        description="历史运行列表（新在前）。候选不决定自身正确性；状态与结果一律来自服务端持久化的 run 目录。"
        actions={
          <Button variant="secondary" size="sm" onClick={reload}>
            刷新
          </Button>
        }
      />

      {loading && !data ? (
        <LoadingBlock />
      ) : error ? (
        <ErrorPanel error={error} onRetry={reload} />
      ) : runs.length === 0 ? (
        <EmptyState
          icon={<IconInbox className="h-8 w-8" />}
          title="还没有任何 Run"
          description="可前往「新建运行」启动（启动表单在后续工作包接入，POST /api/runs）。"
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
                <TH>Champion (sha256)</TH>
                <TH />
              </TR>
            </THead>
            <TBody>
              {runs.map((run) => (
                <TR key={run.run_id} className="hover:bg-surface-raised/60">
                  <TD className="font-mono text-xs">{run.run_id}</TD>
                  <TD>
                    <RunStateBadge state={run.state} />
                  </TD>
                  <TD className="font-mono text-xs text-muted">
                    {run.mode ?? "-"}
                  </TD>
                  <TD className="max-w-[16rem] truncate font-mono text-xs">
                    {run.problem ?? "-"}
                  </TD>
                  <TD className="font-mono text-xs text-muted">
                    {shortHash(run.champion)}
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
              ))}
            </TBody>
          </Table>
        </TableWrapper>
      )}
    </div>
  );
}
