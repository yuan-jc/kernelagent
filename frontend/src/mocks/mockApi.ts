/**
 * Mock 请求路由：与 src/api/client.ts 的 request<T>(path, options)
 * 同签名。MOCKS_ENABLED 时由 client 统一转入这里。
 *
 * 覆盖端点：
 * - 现有 6 端点（health/problems/runs/run detail/POST runs/models）
 * - 设计规范提案 P2（report）/ P3（journal）/ P4（records）/ P5（workspace）
 *
 * 安全：mock 层不读取、不存储 api_key（startRun 的 key 字段直接丢弃）。
 */

import { MockHttpError } from "./http";
import {
  MOCK_MODELS,
  MOCK_PROBLEMS,
  staticRunFixtures,
} from "./fixtures";
import {
  createDynamicRun,
  dynamicJournal,
  dynamicRecord,
  dynamicReport,
  dynamicSnapshot,
  dynamicState,
  dynamicWorkspace,
  dynamicWorkspaceFile,
  type DynamicRun,
} from "./world";

interface MockRequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
}

/** 动态（时间线驱动）runs：预置一个"刚刚启动"的演示 run */
const dynamicRuns: DynamicRun[] = [createDynamicRun({
  mode: "demo-correct",
  problem: "kernelbench:l1:40",
  backend: "triton",
  model: "glm-4.5",
  base_url: "https://api.deepseek.com",
  max_candidates: 5,
  max_repair_rounds: 2,
  gpu_budget_seconds: 1800,
  token_budget: 200000,
})];

function findDynamic(runId: string): DynamicRun | undefined {
  return dynamicRuns.find((r) => r.runId === runId);
}

function activeDynamic(): DynamicRun | undefined {
  return dynamicRuns.find((r) => dynamicState(r) === "running");
}

function allRunIds(): string[] {
  return [...dynamicRuns.map((r) => r.runId), ...staticRunFixtures().keys()];
}

function requireRun(runId: string): void {
  if (!allRunIds().includes(runId)) {
    throw new MockHttpError(404, `unknown run '${runId}' (mock)`);
  }
}

function handleStartRun(body: unknown): { run_id: string; state: string } {
  const payload = (body ?? {}) as Record<string, unknown>;
  const mode = String(payload.mode ?? "live");
  if (!["live", "demo-correct", "demo-wrong"].includes(mode)) {
    throw new MockHttpError(400, `mode must be one of live/demo-correct/demo-wrong; got ${JSON.stringify(mode)} (mock)`);
  }
  const problem = String(payload.problem ?? "");
  if (!/^kernelbench:l\d+:\d+$/.test(problem)) {
    throw new MockHttpError(400, `problem 无法解析: ${JSON.stringify(problem)} (mock)`);
  }
  const busy = activeDynamic();
  if (busy) {
    throw new MockHttpError(409, `run ${busy.runId} is still using the GPU (mock)`);
  }
  // api_key 有意不读取：mock 不接触任何凭据。
  const run = createDynamicRun({
    mode: mode as DynamicRun["config"]["mode"],
    problem,
    backend: String(payload.backend ?? "triton"),
    model: String(payload.model ?? "glm-4.5"),
    base_url: String(payload.base_url ?? ""),
    max_candidates: Number(payload.max_candidates ?? 5),
    max_repair_rounds: Number(payload.max_repair_rounds ?? 2),
    gpu_budget_seconds: Number(payload.gpu_budget_seconds ?? 1800),
    token_budget: Number(payload.token_budget ?? 200000),
  });
  dynamicRuns.push(run);
  return { run_id: run.runId, state: "running" };
}

/** GET /api/runs — 摘要（新在前），复用快照推导 state/champion/started_at */
function listRuns() {
  const summaries = allRunIds().map((runId) => {
    const dyn = findDynamic(runId);
    const snap = dyn ? dynamicSnapshot(dyn) : staticRunFixtures().get(runId)!.snapshot();
    return {
      run_id: snap.run_id,
      state: snap.state,
      mode: snap.job?.mode ?? null,
      problem: snap.job?.config?.problem ?? null,
      champion: snap.champion?.candidate_sha256 ?? null,
      started_at: snap.job?.started_at ?? null,
    };
  });
  summaries.sort((a, b) => (a.run_id < b.run_id ? 1 : -1));
  return { runs: summaries };
}

/** mock 的 final 记录清单（对齐 C3 GET /records 的形状） */
function recordIdsOf(runId: string): string[] {
  const dyn = findDynamic(runId);
  const candidates = dyn ? ["baseline-eager", "candidate-000", "candidate-001"] : [];
  if (dyn) {
    return candidates.filter((id) => {
      try {
        dynamicRecord(dyn, id);
        return true;
      } catch {
        return false;
      }
    });
  }
  const fixture = staticRunFixtures().get(runId)!;
  const ids: string[] = [];
  for (const id of ["baseline-eager", "candidate-000", "candidate-001", "candidate-002"]) {
    try {
      fixture.record(id);
      ids.push(id);
    } catch {
      // 该 fixture 无此记录
    }
  }
  return ids;
}

export async function mockRequest<T>(path: string, options: MockRequestOptions = {}): Promise<T> {
  const method = options.method ?? "GET";
  const [pathOnly, queryString] = path.split("?");
  const query = new URLSearchParams(queryString ?? "");
  // 模拟真实网络延迟可调；50ms 让 loading 态可见但不拖慢演示
  await new Promise((resolve) => setTimeout(resolve, 40));

  if (method === "POST") {
    if (pathOnly === "/api/runs") {
      return handleStartRun(options.body) as T;
    }
    if (pathOnly === "/api/models") {
      return { models: MOCK_MODELS } as T;
    }
    throw new MockHttpError(404, "not found (mock)");
  }

  switch (pathOnly) {
    case "/api/health": {
      const active = activeDynamic();
      return {
        ok: true,
        gpu_device: "nvidia.com/gpu=GPU-mock-0000-1111-2222（mock 设备）",
        active_run_id: active?.runId ?? null,
      } as T;
    }
    case "/api/problems":
      return MOCK_PROBLEMS as T;
    case "/api/runs":
      return listRuns() as T;
    default:
      break;
  }

  const runMatch = pathOnly.match(/^\/api\/runs\/([^/]+)(\/.*)?$/);
  if (runMatch) {
    const runId = decodeURIComponent(runMatch[1]);
    const sub = runMatch[2] ?? "";
    requireRun(runId);
    const dyn = findDynamic(runId);
    const fixture = staticRunFixtures().get(runId);

    if (sub === "" ) {
      const snap = dyn ? dynamicSnapshot(dyn) : fixture!.snapshot();
      return snap as T;
    }
    if (sub === "/journal") {
      const afterRaw = query.get("after");
      const after = afterRaw === null ? undefined : Number(afterRaw);
      const entries = dyn
        ? dynamicJournal(dyn, after)
        : (() => {
            const all = fixture!.journal();
            const start = typeof after === "number" && after > 0 ? after : 0;
            return { run_id: runId, next_after: all.length, entries: all.slice(start) };
          })();
      return entries as T;
    }
    if (sub === "/records") {
      const ids = recordIdsOf(runId);
      return {
        run_id: runId,
        records: ids.map((id) => ({
          record: id,
          variant: "final",
          file: `${id}.json`,
          size: 128,
        })),
      } as T;
    }
    if (sub === "/workspace") {
      return (dyn ? dynamicWorkspace(dyn) : fixture!.workspace()) as T;
    }
    if (sub === "/workspace/file") {
      const filePath = query.get("path") ?? "";
      if (!filePath) throw new MockHttpError(400, "missing ?path= (mock)");
      return (dyn
        ? dynamicWorkspaceFile(dyn, filePath)
        : fixture!.file(filePath)) as T;
    }
    if (sub === "/report") {
      return (dyn ? dynamicReport(dyn) : fixture!.report()) as T;
    }
    const recordMatch = sub.match(/^\/records\/([^/]+)$/);
    if (recordMatch) {
      const actionId = decodeURIComponent(recordMatch[1]);
      return (dyn ? dynamicRecord(dyn, actionId) : fixture!.record(actionId)) as T;
    }
  }

  throw new MockHttpError(404, `not found: ${pathOnly} (mock)`);
}
