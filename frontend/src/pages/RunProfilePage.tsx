import { useParams } from "react-router-dom";
import { PageHeader } from "../components/layout/PageHeader";
import { Card, CardContent } from "../components/ui";
import { EmptyState, NotRunBadge } from "../components/ui";
import { IconPulse } from "../components/icons";

/**
 * Profile 报告页（spec §3.7）：路由 /runs/:runId/profile。
 * 数据依赖设计规范提案 P4（records）/P5（workspace 日志），为后续工作包。
 */
export function RunProfilePage() {
  const { runId = "" } = useParams();

  return (
    <div className="space-y-5">
      <PageHeader
        title={
          <span className="inline-flex items-center gap-3">
            Profile
            {runId && (
              <span className="font-mono text-sm font-normal text-muted">
                {runId}
              </span>
            )}
          </span>
        }
        description="profiling 与正式 timing 分开；profile 结果用于机制解释，不用于晋升判定。"
        actions={<NotRunBadge what="profiling 报告" />}
      />
      <EmptyState
        icon={<IconPulse className="h-8 w-8" />}
        title="Profiling 视图建设中"
        description="后续将按 run 目录中的 timing 记录（batches_ms 分布）与容器 stdout/stderr 渲染：批次延迟图、样本统计、ANSI 日志查看器（auto-scroll）；无数据时整页保持 NOT_RUN 状态而非空图。"
      />
      <Card>
        <CardContent className="text-xs leading-relaxed text-muted">
          约束提示：SOL 不证明算法最优；profile 面板的任何因果结论都需要对应
          journal / report 证据支撑，本页不会由前端推导状态。
        </CardContent>
      </Card>
    </div>
  );
}
