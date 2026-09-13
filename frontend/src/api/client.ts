/**
 * 类型化 API client：唯一的前后端通信入口。
 *
 * - base URL 集中在这里管理：默认同源（开发期由 vite proxy 把 /api
 *   转发到 server.py 默认端口 8501），生产部署可通过
 *   VITE_API_BASE_URL 环境变量指向别处（值不写入代码）。
 * - 非 2xx 统一抛 ApiError（status + 服务端 {"error": ...} 消息）。
 */

import type {
  ApiErrorBody,
  HealthResponse,
  ListModelsRequest,
  ListModelsResponse,
  ProblemsResponse,
  RunSnapshot,
  StartRunRequest,
  StartRunResponse,
  RunsResponse,
} from "./types";

/** 服务端默认端口（server.py: serve(port=8501)） */
export const BACKEND_PORT = 8501;

function resolveBaseUrl(): string {
  const raw: unknown = import.meta.env.VITE_API_BASE_URL;
  if (typeof raw === "string" && raw.trim() !== "") {
    return raw.trim().replace(/\/+$/, "");
  }
  // 同源：开发期走 vite proxy（/api -> 127.0.0.1:8501），
  // 生产期由 server.py 直接同源托管。
  return "";
}

export const API_BASE_URL = resolveBaseUrl();

/** HTTP 层错误：携带状态码与服务端 error 消息 */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function toApiError(cause: unknown): ApiError {
  if (cause instanceof ApiError) return cause;
  if (cause instanceof Error) return new ApiError(0, cause.message);
  return new ApiError(0, String(cause));
}

interface RequestOptions {
  method?: "GET" | "POST";
  /** JSON 请求体（POST）；GET 忽略 */
  body?: unknown;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body } = options;
  const init: RequestInit = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (cause) {
    // fetch 只在网络层失败时 reject（后端未启动 / CORS / 断网）
    throw new ApiError(
      0,
      `无法连接后端（${API_BASE_URL || "同源"}${path}）：${
        cause instanceof Error ? cause.message : String(cause)
      }`,
    );
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text !== "") {
    try {
      payload = JSON.parse(text) as unknown;
    } catch {
      // 非 JSON 响应体保持 null，走下面的状态码分支
    }
  }

  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof (payload as ApiErrorBody).error === "string"
        ? (payload as ApiErrorBody).error
        : `HTTP ${response.status} ${response.statusText}`.trim();
    throw new ApiError(response.status, message);
  }

  return payload as T;
}

/** 全部端点的类型化封装 */
export const api = {
  /** GET /api/health — 存活 + GPU 设备 + 活跃 run */
  getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
  },

  /** GET /api/problems — bench/level/problem 树（pinned KernelBench snapshot） */
  getProblems(): Promise<ProblemsResponse> {
    return request<ProblemsResponse>("/api/problems");
  },

  /** GET /api/runs — 运行历史摘要（新在前） */
  listRuns(): Promise<RunsResponse> {
    return request<RunsResponse>("/api/runs");
  },

  /** GET /api/runs/<id> — 单个 run 的持久化快照；未知 id 返回 404 */
  getRun(runId: string): Promise<RunSnapshot> {
    return request<RunSnapshot>(`/api/runs/${encodeURIComponent(runId)}`);
  },

  /**
   * POST /api/runs — 启动一次优化 run。
   * 409 = GPU 已被活跃 run 占用；400 = 配置/请求非法。
   * api_key 只随本请求进入服务端内存。
   */
  startRun(payload: StartRunRequest): Promise<StartRunResponse> {
    return request<StartRunResponse>("/api/runs", { method: "POST", body: payload });
  },

  /**
   * POST /api/models — 用给定 base_url（+可选 key）列出可用模型。
   * 400 = 永久错误；502 = provider 不可达。
   */
  listModels(payload: ListModelsRequest): Promise<ListModelsResponse> {
    return request<ListModelsResponse>("/api/models", { method: "POST", body: payload });
  },
} as const;
