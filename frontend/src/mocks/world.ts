/**
 * Mock 世界：用"事件时间线"驱动一次模拟的优化 run。
 *
 * 设计：run 的全部可观测状态（快照/journal/records/workspace/report）
 * 都是同一条确定性事件时间线在"当前时刻"的投影，因此轮询时能看到
 * 阶段推进、预算结算、泳道前进、产物逐步出现 —— 与真实 run 的
 * 数据形态一致（journal 无时间戳、运行中 report.candidates 为空等）。
 *
 * 所有数值均为演示用 mock，页面在 mock 模式下有全局 MOCK 标注。
 */

import type {
  ActionRecord,
  JournalEntry,
  JournalResponse,
  RunReport,
  RunSnapshot,
  RunState,
  WorkspaceFileResponse,
  WorkspaceResponse,
} from "../api";
import { MockHttpError } from "./http";

export interface MockRunConfig {
  mode: "live" | "demo-correct" | "demo-wrong";
  problem: string;
  backend: string;
  model: string;
  base_url: string;
  max_candidates: number;
  max_repair_rounds: number;
  gpu_budget_seconds: number;
  token_budget: number;
}

type Outcome = "promoted" | "no_improvement";

/** 单个候选的六阶段推进计划（秒）——总和约 70s，便于演示 */
const PLAN: Array<{ stage: string; seconds: number }> = [
  { stage: "generate", seconds: 8 },
  { stage: "policy", seconds: 4 },
  { stage: "evaluate", seconds: 18 },
  { stage: "correctness_pro", seconds: 12 },
  { stage: "timing", seconds: 20 },
  { stage: "confirm", seconds: 8 },
];

const BASELINE_DONE_AT = 12;
const C0_START = 12;
const C0_FAIL_AT = 36;
const C1_START = 39;

interface Ev {
  t: number;
  kind: string;
  action?: string;
  stage?: string;
  gpu?: number;
  tokens?: number;
  result_ref?: string;
}

function planStageTimes(start: number): Array<{ stage: string; from: number; to: number }> {
  let cursor = start;
  return PLAN.map((p) => {
    const from = cursor;
    cursor += p.seconds;
    return { stage: p.stage, from, to: cursor };
  });
}

function buildTimeline(outcome: Outcome): Ev[] {
  const ev: Ev[] = [];
  // baseline-eager：只跑 timing
  ev.push(
    { t: 0, kind: "budget_reserved", action: "baseline-eager", gpu: 90, tokens: 0 },
    { t: 0, kind: "action_started", action: "baseline-eager" },
    { t: 0, kind: "stage_started", action: "baseline-eager", stage: "timing" },
    { t: BASELINE_DONE_AT, kind: "stage_finished", action: "baseline-eager", stage: "timing" },
    { t: BASELINE_DONE_AT, kind: "budget_settled", action: "baseline-eager", gpu: 90, tokens: 0 },
    {
      t: BASELINE_DONE_AT,
      kind: "action_finished",
      action: "baseline-eager",
      result_ref: "baseline-eager",
    },
  );
  // candidate-000：走到 correctness_pro 失败（候选级错误演示）
  ev.push(
    { t: C0_START, kind: "budget_reserved", action: "candidate-000", gpu: 30, tokens: 12000 },
    { t: C0_START, kind: "action_started", action: "candidate-000" },
  );
  for (const seg of planStageTimes(C0_START).slice(0, 4)) {
    if (seg.from >= C0_FAIL_AT) break;
    ev.push({ t: seg.from, kind: "stage_started", action: "candidate-000", stage: seg.stage });
    ev.push({
      t: Math.min(seg.to, C0_FAIL_AT),
      kind: "stage_finished",
      action: "candidate-000",
      stage: seg.stage,
    });
  }
  ev.push(
    { t: C0_FAIL_AT, kind: "budget_settled", action: "candidate-000", gpu: 30, tokens: 12000 },
    {
      t: C0_FAIL_AT,
      kind: "action_finished",
      action: "candidate-000",
      result_ref: "candidate-000:correctness-failed",
    },
  );
  // candidate-001
  ev.push(
    { t: C1_START, kind: "budget_reserved", action: "candidate-001", gpu: 30, tokens: 15000 },
    { t: C1_START, kind: "action_started", action: "candidate-001" },
  );
  if (outcome === "promoted") {
    for (const seg of planStageTimes(C1_START)) {
      ev.push({ t: seg.from, kind: "stage_started", action: "candidate-001", stage: seg.stage });
      ev.push({ t: seg.to, kind: "stage_finished", action: "candidate-001", stage: seg.stage });
    }
    const end = planStageTimes(C1_START).at(-1)!.to;
    ev.push(
      { t: end, kind: "budget_settled", action: "candidate-001", gpu: 30, tokens: 15000 },
      {
        t: end,
        kind: "action_finished",
        action: "candidate-001",
        result_ref: "candidate-001:promoted",
      },
      { t: end, kind: "experiment_done" },
      { t: end + 1, kind: "report_ready" },
    );
  } else {
    // no_improvement：candidate-001 在 correctness_pro 失败
    for (const seg of planStageTimes(C1_START).slice(0, 4)) {
      if (seg.from >= 78) break;
      ev.push({ t: seg.from, kind: "stage_started", action: "candidate-001", stage: seg.stage });
      ev.push({
        t: Math.min(seg.to, 78),
        kind: "stage_finished",
        action: "candidate-001",
        stage: seg.stage,
      });
    }
    ev.push(
      { t: 78, kind: "budget_settled", action: "candidate-001", gpu: 30, tokens: 15000 },
      {
        t: 78,
        kind: "action_finished",
        action: "candidate-001",
        result_ref: "candidate-001:correctness-failed",
      },
      { t: 78, kind: "experiment_done" },
      { t: 79, kind: "report_ready" },
    );
  }
  return ev.sort((a, b) => a.t - b.t);
}

export interface DynamicRun {
  runId: string;
  startedAtMs: number;
  config: MockRunConfig;
  outcome: Outcome;
  timeline: Ev[];
}

const MOCK_CHAMPION_SHA = "9f2c81d4a6b35e7c8d01f4e2c7b5a3d96e8f0c1a2b3d4e5f6a7b8c9d0e1f2a3b";
const C0_DETAIL =
  "correctness: max rel err 3.2e-02 超出容差 1e-02（fixture: l1:40/case-2，mock 数据）";

export function makeRunId(agoSeconds = 0): string {
  const d = new Date(Date.now() - agoSeconds * 1000);
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(
    d.getMinutes(),
  )}${p(d.getSeconds())}`;
}

function fakeHash(i: number): string {
  const hex = "0123456789abcdef";
  let s = "";
  let x = (i + 1) * 2654435761;
  for (let k = 0; k < 64; k++) {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    s += hex[(x >>> 8) & 0xf];
  }
  return s;
}

export function createDynamicRun(config: MockRunConfig): DynamicRun {
  const outcome: Outcome = config.mode === "demo-wrong" ? "no_improvement" : "promoted";
  return {
    runId: makeRunId(0),
    startedAtMs: Date.now(),
    config,
    outcome,
    timeline: buildTimeline(outcome),
  };
}

export function dynamicElapsed(run: DynamicRun): number {
  return (Date.now() - run.startedAtMs) / 1000;
}

export function dynamicState(run: DynamicRun): RunState {
  const t = dynamicElapsed(run);
  const hasReport = run.timeline.some((e) => e.kind === "report_ready" && e.t <= t);
  if (!hasReport) return "running";
  return run.outcome === "promoted" ? "completed" : "no_improvement";
}

/** 某时刻之前的所有事件 */
function eventsUpTo(run: DynamicRun, t: number): Ev[] {
  return run.timeline.filter((e) => e.t <= t);
}

function dynamicBudget(run: DynamicRun) {
  const t = dynamicElapsed(run);
  const evs = eventsUpTo(run, t);
  let settledGpu = 0;
  let settledTokens = 0;
  let reservedGpu = 0;
  let reservedTokens = 0;
  for (const e of evs) {
    if (e.kind === "budget_reserved") {
      reservedGpu += e.gpu ?? 0;
      reservedTokens += e.tokens ?? 0;
    } else if (e.kind === "budget_settled") {
      settledGpu += e.gpu ?? 0;
      settledTokens += e.tokens ?? 0;
      reservedGpu -= e.gpu ?? 0;
      reservedTokens -= e.tokens ?? 0;
    }
  }
  return {
    gpu_seconds_limit: run.config.gpu_budget_seconds,
    tokens_limit: run.config.token_budget,
    settled_gpu_seconds: settledGpu,
    settled_tokens: settledTokens,
    reserved_gpu_seconds: Math.max(reservedGpu, 0),
    reserved_tokens: Math.max(reservedTokens, 0),
  };
}

export function dynamicSnapshot(run: DynamicRun): RunSnapshot {
  const t = dynamicElapsed(run);
  const evs = eventsUpTo(run, t);
  const state = dynamicState(run);

  // in_flight / progress：action_started 未配对 finished；stage_started 未配对 finished
  const startedActions = new Set<string>();
  const finishedActions = new Set<string>();
  const openStage = new Map<string, string>();
  for (const e of evs) {
    if (e.kind === "action_started" && e.action) startedActions.add(e.action);
    if (e.kind === "action_finished" && e.action) {
      startedActions.delete(e.action);
      finishedActions.add(e.action);
    }
    if (e.action && e.kind === "stage_started") openStage.set(e.action, e.stage ?? "");
    if (e.action && e.kind === "stage_finished") openStage.delete(e.action);
  }
  const inFlight = [...startedActions];
  const progress: Record<string, string> = {};
  for (const id of inFlight) progress[id] = openStage.get(id) ?? "unknown";

  // candidates 只在 report_ready 之后出现（镜像真实行为：report.json 落盘才有）
  const candidates: RunSnapshot["candidates"] = [];
  const champion = {
    candidate_sha256: null as string | null,
    path: null as string | null,
  };
  if (state === "completed" || state === "no_improvement") {
    candidates.push(
      {
        candidate: "baseline-eager",
        stage: "timing",
        status: "measured",
      },
      {
        candidate: "candidate-000",
        stage: "correctness_pro",
        status: "failed",
        detail: C0_DETAIL,
      },
    );
    if (run.outcome === "promoted") {
      candidates.push({
        candidate: "candidate-001",
        stage: "confirm",
        status: "promoted",
        ratio_ci_95: [1.12, 1.31],
      });
      champion.candidate_sha256 = MOCK_CHAMPION_SHA;
      champion.path = "timing-inputs-candidate-001-timing/candidate.py";
    } else {
      candidates.push({
        candidate: "candidate-001",
        stage: "correctness_pro",
        status: "failed",
        detail:
          "correctness: 候选在异步拷贝路径上绕过了参考比较，被增强正确性门拒绝（mock 数据）",
      });
    }
  }

  return {
    run_id: run.runId,
    state,
    budget: dynamicBudget(run),
    in_flight: inFlight,
    progress,
    candidates,
    champion,
    candidate_trust: "cooperative",
    adversarially_secure: false,
    error_tail: null,
    job: {
      run_id: run.runId,
      mode: run.config.mode,
      config: {
        problem: run.config.problem,
        backend: run.config.backend,
        model_id: run.config.model,
        base_url: run.config.base_url,
        max_candidates: run.config.max_candidates,
        max_repair_rounds: run.config.max_repair_rounds,
        gpu_budget_seconds: run.config.gpu_budget_seconds,
        token_budget: run.config.token_budget,
        gpu_device: "nvidia.com/gpu=GPU-mock-0000-1111-2222（mock 设备）",
      },
      started_at: run.startedAtMs / 1000,
    },
  };
}

export function dynamicJournal(
  run: DynamicRun,
  after: number | undefined,
): JournalResponse {
  const t = dynamicElapsed(run);
  const evs = eventsUpTo(run, t);
  const all: JournalEntry[] = evs.map((e, i) => ({
    seq: i + 1,
    kind: e.kind,
    action_id: e.action,
    stage: e.stage,
    gpu_seconds: e.gpu,
    tokens: e.tokens,
    result_ref: e.result_ref,
    entry_hash: fakeHash(i),
    prev_hash: i === 0 ? "genesis" : fakeHash(i - 1),
  }));
  const start = typeof after === "number" && after > 0 ? after : 0;
  return {
    run_id: run.runId,
    next_after: all.length,
    entries: all.slice(start),
  };
}

const BASELINE_BATCHES = [
  7.11, 6.91, 7.01, 6.69, 6.42, 6.61, 6.52, 6.42, 6.42, 8.01, 8.29, 6.51,
];
const CANDIDATE_BATCHES = [
  5.98, 5.71, 5.66, 5.81, 5.55, 5.62, 5.77, 5.49, 5.83, 6.02, 5.68, 5.59,
];

export function dynamicRecord(run: DynamicRun, actionId: string): ActionRecord {
  const t = dynamicElapsed(run);
  const fail = (what: string): never => {
    throw new MockHttpError(404, `no record ${what} (mock)`);
  };
  if (actionId === "baseline-eager") {
    if (t < BASELINE_DONE_AT) fail("baseline-eager");
    return {
      candidate: "baseline-eager",
      stage: "timing",
      status: "measured",
      batches_ms: BASELINE_BATCHES,
      async_leak: false,
    };
  }
  if (actionId === "candidate-000") {
    if (t < C0_FAIL_AT) fail("candidate-000");
    return {
      candidate: "candidate-000",
      stage: "correctness_pro",
      status: "failed",
      detail: C0_DETAIL,
    };
  }
  if (actionId === "candidate-001") {
    if (run.outcome === "promoted") {
      const timingDone = planStageTimes(C1_START).find((s) => s.stage === "timing")!.to;
      if (t < timingDone) fail("candidate-001");
      return {
        candidate: "candidate-001",
        stage: "timing",
        status: "measured",
        batches_ms: CANDIDATE_BATCHES,
        async_leak: false,
      };
    }
    if (t < 78) fail("candidate-001");
    return {
      candidate: "candidate-001",
      stage: "correctness_pro",
      status: "failed",
      detail: "correctness: 候选在异步拷贝路径上绕过了参考比较（mock 数据）",
    };
  }
  throw new MockHttpError(404, `no record ${actionId} (mock)`);
}

export function dynamicWorkspace(run: DynamicRun): WorkspaceResponse {
  const t = dynamicElapsed(run);
  const entries: WorkspaceResponse["entries"] = [];
  if (t >= BASELINE_DONE_AT) {
    entries.push({
      path: "timing-inputs-baseline-eager-timing",
      type: "dir",
      files: ["candidate.py", "timing_driver.py", "case.json"],
    });
  }
  if (t >= 23) {
    entries.push({
      path: "container-3f9c2e81a7b6405d",
      type: "dir",
      files: ["container-record.json", "stdout.log", "stderr.log"],
    });
  }
  if (run.outcome === "promoted" && t >= 101) {
    entries.push({
      path: "timing-inputs-candidate-001-timing",
      type: "dir",
      files: ["candidate.py", "timing_driver.py", "case.json"],
    });
  }
  return { entries };
}

const FILE_CANDIDATE_BASELINE = `# mock 数据：仅用于前端演示（baseline eager 参考）
import torch

def candidate(x, weight, bias, gamma, beta, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return x_hat * gamma + bias
`;

const FILE_CANDIDATE_OPT = `# mock 数据：仅用于前端演示（fused triton 候选）
import torch, triton, triton.language as tl

@triton.jit
def _fused_ln(x_ptr, w_ptr, b_ptr, g_ptr, o_ptr, n, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    hat = (x - mean) / tl.sqrt(var + eps)
    g = tl.load(g_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, hat * g + b, mask=mask)
`;

const FILE_STDOUT = `==========
== CUDA ==
==========

CUDA Device Query (CUDA Runtime API) results:
Device 0: "NVIDIA GeForce RTX" (mock)

[eval] case 1/12 ... ok (max_rel_err 0.0e+00)
[eval] case 2/12 ... ok (max_rel_err 4.1e-07)
[eval] correctness_pro: 8 adversarial cases ... ok
[timing] 12 batches, warmup 3 ... measured
`;

const FILE_STDERR = `[warn] Triton JIT cache miss on first launch (expected)\n`;

const FILE_CONTAINER_RECORD = `{
  "image": "kernelagent-eval:mock",
  "exit_code": 0,
  "duration_s": 34.2,
  "gpu_device": "nvidia.com/gpu=GPU-mock-0000"
}`;

const FILE_DRIVER = `# mock 数据：仅用于前端演示
# timing_driver: 固定种子、预热 3 次、测 12 个 batch，输出 batches_ms[]
`;

const FILE_CASE = `{
  "case_id": "l1:40/case-2",
  "shape": [4096, 4096],
  "dtype": "float16",
  "seed": 20260913
}`;

export function dynamicWorkspaceFile(
  run: DynamicRun,
  filePath: string,
): WorkspaceFileResponse {
  const ws = dynamicWorkspace(run);
  const dir = ws.entries.find((e) => filePath.startsWith(`${e.path}/`));
  if (!dir || !dir.files?.includes(filePath.slice(dir.path.length + 1))) {
    throw new MockHttpError(404, `no such workspace file: ${filePath} (mock)`);
  }
  const base = filePath.slice(dir.path.length + 1);
  const table: Record<string, string> = {
    "stdout.log": FILE_STDOUT,
    "stderr.log": FILE_STDERR,
    "container-record.json": FILE_CONTAINER_RECORD,
    "case.json": FILE_CASE,
    "timing_driver.py": FILE_DRIVER,
  };
  let content: string;
  if (base === "candidate.py") {
    content = dir.path.includes("candidate-001")
      ? FILE_CANDIDATE_OPT
      : FILE_CANDIDATE_BASELINE;
  } else {
    content = table[base] ?? `(mock 空文件)`;
  }
  return { path: filePath, size: content.length, content };
}

export function dynamicReport(run: DynamicRun): RunReport {
  const snap = dynamicSnapshot(run);
  if (snap.state === "running") {
    throw new MockHttpError(404, "report.json not written yet (mock)");
  }
  return {
    state: snap.state,
    commit: "6f036b579730536f71d01aed656fac05d32b2b64",
    protocol: "optimize-v1",
    problem_sha256: "8d975d7cee6946513dcfb6ba84c971e53a890f9d9c6f0a4fb828904941f25975",
    problem_path:
      "research/sources/ScalingIntelligence__KernelBench/KernelBench/level1/40_LayerNorm.py",
    gpu_device: "nvidia.com/gpu=GPU-mock-0000-1111-2222（mock 设备）",
    journal_entries: dynamicJournal(run, 0).entries.length,
    config: snap.job?.config ?? {},
    champion: snap.champion,
    candidate_trust: "cooperative",
    adversarially_secure: false,
  };
}
