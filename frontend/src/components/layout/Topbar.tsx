import { useLocation } from "react-router-dom";
import { api } from "../../api";
import { useApi } from "../../lib/useApi";
import { MOCKS_ENABLED } from "../../mocks";
import { Badge } from "../ui/Badge";
import { Spinner } from "../ui/Spinner";

/**
 * 顶栏（spec §3.1）：左侧当前区域标题，右侧 GPU 徽章、活跃 run 徽章、
 * candidate_trust=cooperative 常驻警示徽章（诚实性声明，不可省略）；
 * mock 数据源模式下追加 MOCK 徽章（诚实标注演示数据）。
 */

const SECTION_BY_PREFIX: Array<{ prefix: string; label: string }> = [
  { prefix: "/runs", label: "Runs" },
  { prefix: "/launch", label: "新建运行" },
  { prefix: "/benchmarks", label: "Benchmarks" },
  { prefix: "/settings", label: "设置" },
];

function sectionLabel(pathname: string): string {
  const hit = SECTION_BY_PREFIX.find((s) => pathname.startsWith(s.prefix));
  return hit ? hit.label : "总览";
}

function HealthChip() {
  const { data, error, loading, reload } = useApi(() => api.getHealth(), [], {
    pollMs: 10_000,
  });

  if (loading && !data) {
    return (
      <span className="flex items-center gap-2 text-xs text-muted">
        <Spinner size="xs" /> 连接后端…
      </span>
    );
  }

  if (error || !data) {
    return (
      <button
        type="button"
        onClick={reload}
        title={error ? error.message : undefined}
        className="flex items-center gap-1.5 rounded-full border border-bad/25 bg-error-soft px-2.5 py-1 text-xs text-bad"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-bad" />
        后端离线（点击重试）
      </button>
    );
  }

  return (
    <span className="flex items-center gap-2 text-xs text-muted">
      <Badge variant="success" dot>
        在线
      </Badge>
      <span className="hidden max-w-[16rem] truncate font-mono lg:inline" title={data.gpu_device}>
        {data.gpu_device}
      </span>
      {data.active_run_id ? (
        <Badge variant="running" dot>
          活跃 {data.active_run_id}
        </Badge>
      ) : (
        <span>GPU 空闲</span>
      )}
    </span>
  );
}

export function Topbar() {
  const { pathname } = useLocation();
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between gap-3 border-b border-border bg-bg/85 px-4 backdrop-blur md:px-6">
      <h2 className="text-sm font-medium text-fg">{sectionLabel(pathname)}</h2>
      <div className="flex items-center gap-2 lg:gap-3">
        {/* 常驻诚实性警示：candidate_trust=cooperative（不可移除/弱化） */}
        <Badge
          variant="warning"
          className="max-w-[13rem] whitespace-normal leading-tight sm:max-w-none"
          title="candidate_trust = cooperative：自动评估为协作式，champion 需人工审查后采信"
        >
          candidate_trust=cooperative · champion 需人工审查
        </Badge>
        {MOCKS_ENABLED && (
          <Badge
            variant="primary"
            title="前端当前使用 frontend/src/mocks 内存数据源（可在设置页切换）"
          >
            MOCK
          </Badge>
        )}
        <HealthChip />
      </div>
    </header>
  );
}
