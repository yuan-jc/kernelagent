/**
 * Mock 固定数据：历史终态 runs（completed / no_improvement /
 * budget_exhausted / infra_error）+ 题目树 + 模型列表。
 *
 * completed / no_improvement 两个 run 直接复用 world.ts 的事件时间线
 * （把开始时间回拨到数小时前），保证快照/journal/records/workspace/
 * report 五类数据彼此一致；budget_exhausted / infra_error 为手工构造。
 */

import type {
  ActionRecord,
  JournalEntry,
  ProblemsResponse,
  RunReport,
  RunSnapshot,
  WorkspaceFileResponse,
  WorkspaceResponse,
} from "../api";
import { MockHttpError } from "./http";
import {
  createDynamicRun,
  makeRunId,
  dynamicJournal,
  dynamicRecord,
  dynamicReport,
  dynamicSnapshot,
  dynamicWorkspace,
  dynamicWorkspaceFile,
  type DynamicRun,
} from "./world";

const MOCK_GPU = "nvidia.com/gpu=GPU-mock-0000-1111-2222（mock 设备）";

function backdatedRun(
  agoSec: number,
  mode: "live" | "demo-correct" | "demo-wrong",
): DynamicRun {
  const run = createDynamicRun({
    mode,
    problem: "kernelbench:l1:40",
    backend: "triton",
    model: "glm-4.5",
    base_url: "https://api.deepseek.com",
    max_candidates: 5,
    max_repair_rounds: 2,
    gpu_budget_seconds: 1800,
    token_budget: 200000,
  });
  run.runId = makeRunId(agoSec);
  run.startedAtMs = Date.now() - agoSec * 1000;
  return run;
}

/** 历史 completed run：约 9 小时前启动（promoted 时间线早已走完） */
const completedRun = backdatedRun(9 * 3600, "live");
/** 历史 no_improvement run：约 3.5 小时前启动 */
const noImprovementRun = backdatedRun(3.5 * 3600 + 1800, "demo-wrong");

// -- budget_exhausted（手工构造） --------------------------------------------

function budgetExhaustedSnapshot(runId: string): RunSnapshot {
  return {
    run_id: runId,
    state: "budget_exhausted",
    budget: {
      gpu_seconds_limit: 1800.0,
      tokens_limit: 200000,
      settled_gpu_seconds: 1800.0,
      settled_tokens: 137_000,
      reserved_gpu_seconds: 0.0,
      reserved_tokens: 0,
    },
    in_flight: [],
    progress: {},
    candidates: [
      { candidate: "baseline-eager", stage: "timing", status: "measured" },
      {
        candidate: "candidate-000",
        stage: "confirm",
        status: "retained",
        ratio_ci_95: [1.03, 1.12],
      },
      {
        candidate: "candidate-001",
        stage: "evaluate",
        status: "failed",
        detail: "预算耗尽：GPU 秒预留不足，动作在容器评测前被预算账本拒绝（mock 数据）",
      },
    ],
    champion: { candidate_sha256: null, path: null },
    candidate_trust: "cooperative",
    adversarially_secure: false,
    error_tail: null,
    job: {
      run_id: runId,
      mode: "live",
      config: {
        problem: "kernelbench:l1:23",
        backend: "triton",
        model_id: "glm-4.5",
        base_url: "https://api.deepseek.com",
        max_candidates: 5,
        max_repair_rounds: 2,
        gpu_budget_seconds: 1800.0,
        token_budget: 200000,
        gpu_device: MOCK_GPU,
      },
      started_at: Date.now() / 1000 - 2.2 * 3600,
    },
  };
}

const BUDGET_RUN_ID = makeRunId(2.2 * 3600);

const budgetJournal: JournalEntry[] = [
  { seq: 1, kind: "budget_reserved", action_id: "baseline-eager", gpu_seconds: 300, tokens: 0, entry_hash: "aa01", prev_hash: "genesis" },
  { seq: 2, kind: "action_started", action_id: "baseline-eager", entry_hash: "aa02", prev_hash: "aa01" },
  { seq: 3, kind: "budget_settled", action_id: "baseline-eager", gpu_seconds: 300, tokens: 0, entry_hash: "aa03", prev_hash: "aa02" },
  { seq: 4, kind: "action_finished", action_id: "baseline-eager", result_ref: "baseline-eager", entry_hash: "aa04", prev_hash: "aa03" },
  { seq: 5, kind: "budget_reserved", action_id: "candidate-000", gpu_seconds: 600, tokens: 60000, entry_hash: "aa05", prev_hash: "aa04" },
  { seq: 6, kind: "action_started", action_id: "candidate-000", entry_hash: "aa06", prev_hash: "aa05" },
  { seq: 7, kind: "budget_settled", action_id: "candidate-000", gpu_seconds: 600, tokens: 60000, entry_hash: "aa07", prev_hash: "aa06" },
  { seq: 8, kind: "action_finished", action_id: "candidate-000", result_ref: "candidate-000:retained", entry_hash: "aa08", prev_hash: "aa07" },
  { seq: 9, kind: "budget_reserved", action_id: "candidate-001", gpu_seconds: 900, tokens: 77000, entry_hash: "aa09", prev_hash: "aa08" },
  { seq: 10, kind: "action_started", action_id: "candidate-001", entry_hash: "aa0a", prev_hash: "aa09" },
  { seq: 11, kind: "budget_settled", action_id: "candidate-001", gpu_seconds: 900, tokens: 77000, entry_hash: "aa0b", prev_hash: "aa0a" },
  { seq: 12, kind: "action_interrupted", action_id: "candidate-001", result_ref: "budget-exhausted", entry_hash: "aa0c", prev_hash: "aa0b" },
  { seq: 13, kind: "experiment_done", settled_gpu_seconds: 1800, settled_tokens: 137000, entry_hash: "aa0d", prev_hash: "aa0c" },
];

const budgetRecords: Record<string, ActionRecord> = {
  "baseline-eager": {
    candidate: "baseline-eager",
    stage: "timing",
    status: "measured",
    batches_ms: [11.42, 11.08, 11.63, 11.21, 10.98, 11.34, 11.77, 11.12],
    async_leak: false,
  },
  "candidate-000": {
    candidate: "candidate-000",
    stage: "timing",
    status: "measured",
    batches_ms: [10.9, 10.72, 11.01, 10.65, 10.88, 11.2, 10.77, 10.95],
    async_leak: false,
  },
};

const budgetWorkspace: WorkspaceResponse = {
  entries: [
    { path: "container-77c1d9e2b4a83f60", type: "dir", files: ["container-record.json", "stdout.log", "stderr.log"] },
    { path: "timing-inputs-baseline-eager-timing", type: "dir", files: ["candidate.py", "timing_driver.py", "case.json"] },
    { path: "timing-inputs-candidate-000-timing", type: "dir", files: ["candidate.py", "timing_driver.py", "case.json"] },
  ],
};

// -- infra_error（手工构造） --------------------------------------------------

const INFRA_RUN_ID = makeRunId(55 * 60);

const infraSnapshot: RunSnapshot = {
  run_id: INFRA_RUN_ID,
  state: "infra_error",
  budget: {
    gpu_seconds_limit: 1800.0,
    tokens_limit: 200000,
    settled_gpu_seconds: 0,
    settled_tokens: 0,
    reserved_gpu_seconds: 0,
    reserved_tokens: 0,
  },
  in_flight: [],
  progress: {},
  candidates: [],
  champion: { candidate_sha256: null, path: null },
  candidate_trust: undefined,
  adversarially_secure: undefined,
  error_tail: `Traceback (most recent call last):
  File "src/kernelagent/webapp/server.py", line 144, in _run_job
    return {"run_id": run_id, "state": "running"}
  File "src/kernelagent/webapp/jobs.py", line 69, in build_generator
    return default_generator(base_url)
  File "src/kernelagent/config.py", line 44, in default_generator
    raise OptimizationConfigError(
kernelagent.optimization.OptimizationConfigError: environment variable MODEL_PROVIDER_API_KEY is not set: live generation requires provider credentials on the control plane

（mock 数据：结构与真实 error.txt 一致）`,
  job: {
    run_id: INFRA_RUN_ID,
    mode: "live",
    config: {
      problem: "kernelbench:l1:40",
      backend: "triton",
      model_id: "glm-4.5",
      base_url: "https://api.deepseek.com",
      max_candidates: 5,
      max_repair_rounds: 2,
      gpu_budget_seconds: 1800.0,
      token_budget: 200000,
      gpu_device: MOCK_GPU,
    },
    started_at: Date.now() / 1000 - 55 * 60,
  },
};

// -- 汇总 ---------------------------------------------------------------------

export interface StaticRunFixture {
  snapshot: () => RunSnapshot;
  journal: () => JournalEntry[];
  record: (actionId: string) => ActionRecord;
  workspace: () => WorkspaceResponse;
  file: (path: string) => WorkspaceFileResponse;
  report: () => RunReport;
}

function fromTimeline(run: DynamicRun): StaticRunFixture {
  return {
    snapshot: () => dynamicSnapshot(run),
    journal: () => dynamicJournal(run, 0).entries,
    record: (actionId) => dynamicRecord(run, actionId),
    workspace: () => dynamicWorkspace(run),
    file: (path) => dynamicWorkspaceFile(run, path),
    report: () => dynamicReport(run),
  };
}

const budgetExhaustedFixture: StaticRunFixture = {
  snapshot: () => budgetExhaustedSnapshot(BUDGET_RUN_ID),
  journal: () => budgetJournal,
  record: (actionId) => {
    if (budgetRecords[actionId]) return budgetRecords[actionId];
    if (actionId === "candidate-001") {
      return {
        candidate: "candidate-001",
        stage: "evaluate",
        status: "failed",
        detail: "预算耗尽：动作在容器评测前被预算账本拒绝（mock 数据）",
      };
    }
    throw new MockHttpError(404, `no record ${actionId} (mock)`);
  },
  workspace: () => budgetWorkspace,
  file: (path) => {
    throw new MockHttpError(404, `workspace file 未在 budget_exhausted fixture 中提供: ${path} (mock)`);
  },
  report: () => ({
    state: "budget_exhausted",
    commit: "6f036b579730536f71d01aed656fac05d32b2b64",
    protocol: "optimize-v1",
    problem_sha256: "c41d2f9a7e8b60513dfc6ba84c971e53a890f9d9c6f0a4fb828904941f25975",
    problem_path:
      "research/sources/ScalingIntelligence__KernelBench/KernelBench/level1/23 Softmax.py",
    gpu_device: MOCK_GPU,
    journal_entries: budgetJournal.length,
    config: budgetExhaustedSnapshot(BUDGET_RUN_ID).job?.config ?? {},
    champion: null,
    candidate_trust: "cooperative",
    adversarially_secure: false,
  }),
};

const infraFixture: StaticRunFixture = {
  snapshot: () => infraSnapshot,
  journal: () => [],
  record: (actionId) => {
    throw new MockHttpError(404, `no record ${actionId} (mock)`);
  },
  workspace: () => ({ entries: [] }),
  file: (path) => {
    throw new MockHttpError(404, `no such workspace file: ${path} (mock)`);
  },
  report: () => {
    throw new MockHttpError(404, "report.json not written (mock)");
  },
};

/** 全部静态 run：run_id -> fixture */
export function staticRunFixtures(): Map<string, StaticRunFixture> {
  return new Map([
    [completedRun.runId, fromTimeline(completedRun)],
    [noImprovementRun.runId, fromTimeline(noImprovementRun)],
    [BUDGET_RUN_ID, budgetExhaustedFixture],
    [INFRA_RUN_ID, infraFixture],
  ]);
}

export { completedRun, noImprovementRun };

// -- /api/problems（含 KernelBench；新库由 Benchmarks 页以 Planned 卡展示） ----

export const MOCK_PROBLEMS: ProblemsResponse = {
  bench: "kernelbench",
  note: "tritonbench / flashinfer-bench adapters exist but are not wired into optimize（mock 数据）",
  levels: [
    {
      level: 1,
      problems: [
        { id: 1, spec: "kernelbench:l1:1", label: "1_Square_matrix_multiplication_" },
        { id: 19, spec: "kernelbench:l1:19", label: "19_ReLU" },
        { id: 23, spec: "kernelbench:l1:23", label: "23_Softmax" },
        { id: 40, spec: "kernelbench:l1:40", label: "40_LayerNorm" },
        { id: 62, spec: "kernelbench:l1:62", label: "62_HingeLoss" },
      ],
    },
    {
      level: 2,
      problems: [
        { id: 1, spec: "kernelbench:l2:1", label: "1_Conv2d" },
        { id: 7, spec: "kernelbench:l2:7", label: "7_GEMM_split_k" },
      ],
    },
    {
      level: 3,
      problems: [
        { id: 3, spec: "kernelbench:l3:3", label: "3_LeNet" },
      ],
    },
    {
      level: 4,
      problems: [
        { id: 2, spec: "kernelbench:l4:2", label: "2_NWarp_Download" },
      ],
    },
  ],
};

export const MOCK_MODELS = [
  { id: "glm-4.5", owned_by: "zhipu (mock)" },
  { id: "glm-4.6", owned_by: "zhipu (mock)" },
  { id: "deepseek-chat", owned_by: "deepseek (mock)" },
];
