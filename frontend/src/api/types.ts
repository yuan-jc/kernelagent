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
  /** C3 后端新增：run 启动时间（job.json 或目录时间推导）；缺失时为 undefined */
  started_at?: number | null;
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
  /** 服务端对无 champion 会写 null（report.json 原样），类型如实反映 */
  candidate_sha256?: string | null;
  path?: string | null;
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

// -- GET /api/runs/<id>/journal（设计规范提案 P3；C3 未上线时前端降级）--------

/** journal.jsonl 单条事件（服务端按行返回；无时间戳字段，前端不编造） */
export interface JournalEntry {
  /** P3 规范中的行号；服务端未实现时前端以数组下标代替展示序号 */
  seq?: number;
  kind: string;
  action_id?: string;
  stage?: string;
  gpu_seconds?: number;
  tokens?: number;
  entry_hash?: string;
  prev_hash?: string;
  input_hash?: string;
  budget_reservation?: string;
  budget_settlement?: string;
  result_ref?: string;
  settled_gpu_seconds?: number;
  settled_tokens?: number;
  actions?: string[];
  [key: string]: unknown;
}

export interface JournalResponse {
  run_id: string;
  next_after: number;
  entries: JournalEntry[];
}

// -- GET /api/runs/<id>/records/<action_id>（提案 P4）------------------------

/** C3 后端的 records 只读清单（GET /api/runs/<id>/records） */
export interface RecordListItem {
  record: string;
  variant: "final" | "progress" | string;
  file: string;
  size: number;
}

export interface RecordsListResponse {
  run_id: string;
  records: RecordListItem[];
}

/** timing/评测单条 action 记录；timing 记录含 batches_ms 与 async_leak */
export interface ActionRecord {
  candidate?: string;
  stage?: string;
  status?: string;
  detail?: string;
  ratio_ci_95?: [number, number];
  /** 正式 timing 的批次耗时样本（ms）；缺失时 Profile 页显示 NOT_RUN */
  batches_ms?: number[];
  async_leak?: boolean;
  [key: string]: unknown;
}

// -- GET /api/runs/<id>/workspace（提案 P5，只读）----------------------------

export interface WorkspaceEntry {
  path: string;
  type: "dir" | "file";
  files?: string[];
}

export interface WorkspaceResponse {
  entries: WorkspaceEntry[];
}

export interface WorkspaceFileResponse {
  path: string;
  size: number;
  content: string;
}

// -- GET /api/runs/<id>/report（提案 P2，原始报告）---------------------------

/** report.json 的持久化身份链字段（P2 未上线时无法读取，UI 显式降级） */
export interface RunReport {
  state?: RunState;
  commit?: string;
  protocol?: string;
  problem_sha256?: string;
  problem_path?: string;
  gpu_device?: string;
  journal_entries?: number;
  config?: JobConfig;
  champion?: RunChampion | null;
  candidate_trust?: string;
  adversarially_secure?: boolean;
  [key: string]: unknown;
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
