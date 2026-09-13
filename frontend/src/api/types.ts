/**
 * 与 src/kernelagent/webapp/server.py 对齐的 API 类型。
 *
 * 服务端是标准库 HTTP server（无 schema 校验），这里以 server.py /
 * jobs.py / run_snapshot() 的实现与 artifacts 下真实 report.json 为准建模；
 * 对报告中可能演化的字段保留索引签名，未知字段不会让前端崩溃。
 */

// -- 共享枚举 ---------------------------------------------------------------

/** 运行模式：live 是唯一可计为 LIVE_MODEL 的模式（jobs.py MODES） */
export type RunMode = "live" | "demo-correct" | "demo-wrong";

/** 运行状态（run_snapshot + TERMINAL_STATES） */
export type RunState =
  | "running"
  | "completed"
  | "no_improvement"
  | "budget_exhausted"
  | "infra_error"
  | "journal_corrupt"
  | "unknown";

export const RUN_STATES: readonly RunState[] = [
  "running",
  "completed",
  "no_improvement",
  "budget_exhausted",
  "infra_error",
  "journal_corrupt",
  "unknown",
];

export function isRunState(value: unknown): value is RunState {
  return typeof value === "string" && (RUN_STATES as readonly string[]).includes(value);
}

// -- GET /api/health ---------------------------------------------------------

export interface HealthResponse {
  ok: boolean;
  gpu_device: string;
  active_run_id: string | null;
}

// -- GET /api/problems -------------------------------------------------------

/** 单个 KernelBench 题目（jobs.py list_problems） */
export interface ProblemEntry {
  id: number;
  /** 形如 "kernelbench:l1:3" 的问题 spec，POST /api/runs 的 problem 字段 */
  spec: string;
  /** 形如 "3_convolution.py" 去扩展名的 label */
  label: string;
}

export interface ProblemLevel {
  level: number;
  problems: ProblemEntry[];
}

export interface ProblemsResponse {
  bench: string;
  note: string;
  levels: ProblemLevel[];
}

// -- GET /api/runs -----------------------------------------------------------

export interface RunSummary {
  run_id: string;
  state: RunState;
  mode: RunMode | null;
  problem: string | null;
  /** champion 的 candidate_sha256；无 champion 时为 null */
  champion: string | null;
}

export interface RunsResponse {
  runs: RunSummary[];
}

// -- GET /api/runs/<id> ------------------------------------------------------

/** 预算账本（journal reconstruct 或 report.budget）；字段可能逐步演化 */
export interface RunBudget {
  gpu_seconds_limit?: number | null;
  tokens_limit?: number | null;
  settled_gpu_seconds?: number;
  settled_tokens?: number;
  reserved_gpu_seconds?: number;
  reserved_tokens?: number;
  [key: string]: unknown;
}

export interface RunChampion {
  candidate_sha256?: string;
  path?: string;
  [key: string]: unknown;
}

/** report.json 中的单候选记录（stage/status/ratio_ci_95 等） */
export interface CandidateRecord {
  candidate?: string;
  candidate_sha256?: string;
  stage?: string;
  status?: string;
  detail?: string;
  /** 确认阶段的速度比 95% 置信区间 [low, high] */
  ratio_ci_95?: [number, number];
  champion_path?: string;
  [key: string]: unknown;
}

/** job.json 的 config 段（不含 api_key，服务端保证不落盘） */
export interface JobConfig {
  problem?: string;
  backend?: string;
  model_id?: string;
  base_url?: string;
  max_candidates?: number;
  max_repair_rounds?: number;
  gpu_budget_seconds?: number;
  token_budget?: number;
  gpu_device?: string;
  [key: string]: unknown;
}

export interface JobRecord {
  run_id?: string;
  mode?: RunMode;
  config?: JobConfig;
  /** Unix 秒 */
  started_at?: number;
  [key: string]: unknown;
}

/** GET /api/runs/<id> 的响应（run_snapshot）；404 时客户端抛 ApiError */
export interface RunSnapshot {
  run_id: string;
  state: RunState;
  budget: RunBudget | null;
  /** 仍在进行中的 action id 列表 */
  in_flight: string[];
  /** action id -> 当前 stage 字符串 */
  progress: Record<string, string>;
  candidates: CandidateRecord[];
  champion: RunChampion | null;
  /** error.txt 尾部（<=1500 字符） */
  error_tail: string | null;
  job: JobRecord | null;
  candidate_trust?: string;
  adversarially_secure?: boolean;
}

// -- POST /api/runs ----------------------------------------------------------

export interface StartRunRequest {
  mode: RunMode;
  /** 问题 spec，如 "kernelbench:l1:3" */
  problem: string;
  backend?: string;
  model?: string;
  base_url?: string;
  /** 仅随本次请求进入服务端内存，绝不落盘 */
  api_key?: string;
  max_candidates?: number;
  max_repair_rounds?: number;
  gpu_budget_seconds?: number;
  token_budget?: number;
  disable_thinking?: boolean;
}

export interface StartRunResponse {
  run_id: string;
  state: string;
}

// -- POST /api/models --------------------------------------------------------

/** openai_compat.list_models 返回的单个模型条目 */
export interface ModelInfo {
  id: string;
  owned_by: string | null;
}

export interface ListModelsRequest {
  base_url: string;
  /** 缺省时服务端回退到控制面环境变量（server.py: API_KEY_ENV） */
  api_key?: string;
}

export interface ListModelsResponse {
  models: ModelInfo[];
  /** 400/502 时服务端附带错误消息且 models 为空数组 */
  error?: string;
}

// -- 统一错误体（非 2xx：{"error": string}）----------------------------------

export interface ApiErrorBody {
  error: string;
}
