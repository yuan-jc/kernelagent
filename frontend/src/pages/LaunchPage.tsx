import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, toApiError } from "../api";
import type { ListModelsResponse, RunMode } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { MOCKS_ENABLED } from "../mocks";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  ErrorPanel,
  NotRunBadge,
} from "../components/ui";
import { prefs } from "../lib/prefs";

/**
 * 新建运行（spec §3.5）：POST /api/runs 全字段表单。
 *
 * 安全约束（不可妥协）：
 * - api_key 只存在于本组件的内存 useState，随单次请求发送；
 *   提交成功后立即清空；绝不写入 localStorage / sessionStorage / URL / 日志。
 * - 预览卡只显示 key 的占位说明，不显示内容。
 */

const MODE_OPTIONS: Array<{ value: RunMode; label: string; desc: string; demo: boolean }> = [
  {
    value: "live",
    label: "live",
    desc: "真实模型生成（需 API key 或服务端环境变量回退）；唯一可计为 LIVE_MODEL 的模式",
    demo: false,
  },
  {
    value: "demo-correct",
    label: "demo-correct",
    desc: "演示·正确候选：真 GPU 验证完整流程，不依赖 provider",
    demo: true,
  },
  {
    value: "demo-wrong",
    label: "demo-wrong",
    desc: "演示·错误候选：验证修复/拒绝路径",
    demo: true,
  },
];

const inputClass =
  "h-9 w-full rounded-md border border-border bg-surface-3 px-3 font-mono text-sm text-fg placeholder:text-text-3 focus:border-accent focus:outline-none";

function Field({
  label,
  hint,
  children,
  htmlFor,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  htmlFor?: string;
}) {
  return (
    <div>
      <label htmlFor={htmlFor} className="mb-1 block text-xs font-medium text-muted">
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-[11px] leading-relaxed text-text-3">{hint}</p>}
    </div>
  );
}

export function LaunchPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const defaults = prefs.getLaunchDefaults();
  const providerPresets = prefs.getProviderPresets();

  const [mode, setMode] = useState<RunMode>("demo-correct");
  const [problem, setProblem] = useState(params.get("problem") ?? "");
  const [model, setModel] = useState("glm-4.5");
  const [baseUrl, setBaseUrl] = useState(providerPresets[0]?.baseUrl ?? "");
  const [apiKey, setApiKey] = useState("");
  const [backend, setBackend] = useState("triton");
  const [disableThinking, setDisableThinking] = useState(false);
  const [maxCandidates, setMaxCandidates] = useState(defaults.maxCandidates);
  const [maxRepairRounds, setMaxRepairRounds] = useState(defaults.maxRepairRounds);
  const [gpuBudgetSeconds, setGpuBudgetSeconds] = useState(defaults.gpuBudgetSeconds);
  const [tokenBudget, setTokenBudget] = useState(defaults.tokenBudget);

  const [problemsData, setProblemsData] = useState<Awaited<ReturnType<typeof api.getProblems>> | null>(null);
  const [problemsError, setProblemsError] = useState<string | null>(null);
  const [modelsResult, setModelsResult] = useState<ListModelsResponse | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<{ status: number; message: string } | null>(null);
  const [submitted, setSubmitted] = useState(false);

  // 题目树：进入页面时拉一次（/api/problems）
  const problemOptions = useMemo(() => {
    const levels = problemsData?.levels ?? [];
    return levels.flatMap((level) =>
      level.problems.map((p) => ({
        spec: p.spec,
        label: `${p.label}（L${level.level}）`,
      })),
    );
  }, [problemsData]);

  async function loadProblems() {
    setProblemsError(null);
    try {
      setProblemsData(await api.getProblems());
    } catch (cause) {
      setProblemsError(toApiError(cause).message);
    }
  }

  async function handleListModels() {
    if (!baseUrl.trim()) {
      setModelsError("Base URL 不能为空。");
      return;
    }
    setModelsLoading(true);
    setModelsError(null);
    setModelsResult(null);
    try {
      const response = await api.listModels({
        base_url: baseUrl.trim(),
        // key 只随本次请求发往本地 server 内存；前端不存储、不打印。
        api_key: apiKey,
      });
      setModelsResult(response);
      if (response.models.length > 0 && !response.models.some((m) => m.id === model)) {
        setModel(response.models[0]!.id);
      }
    } catch (cause) {
      setModelsError(toApiError(cause).message);
    } finally {
      setModelsLoading(false);
    }
  }

  function buildPayload() {
    return {
      mode,
      problem: problem.trim(),
      backend,
      model: model.trim(),
      base_url: baseUrl.trim(),
      // api_key 仅随请求体进入服务端内存；见文件头安全约束。
      api_key: apiKey,
      max_candidates: Number(maxCandidates),
      max_repair_rounds: Number(maxRepairRounds),
      gpu_budget_seconds: Number(gpuBudgetSeconds),
      token_budget: Number(tokenBudget),
      disable_thinking: disableThinking,
    };
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    try {
      const response = await api.startRun(buildPayload());
      setApiKey(""); // 提交成功立即清空 key（不保留在内存 state）
      setSubmitted(true);
      navigate(`/runs/${response.run_id}`);
    } catch (cause) {
      const err = toApiError(cause);
      // 409 = GPU 被活跃 run 占用；400 = 配置非法。保留原文，不清空表单。
      setSubmitError({ status: err.status, message: err.message });
    } finally {
      setSubmitting(false);
    }
  }

  const previewPayload = { ...buildPayload() };
  previewPayload.api_key = apiKey
    ? "«已填写：仅随本次请求发送到服务端内存，随后清空，不出现在日志/URL/存储»"
    : "«留空：服务端回退到控制面环境变量 API_KEY_ENV»";

  return (
    <div className="space-y-5">
      <PageHeader
        title="新建运行"
        description="运行模式、题目、模型与凭据、预算与高级字段。单 GPU 同时只能有一个运行（重复启动返回 409）；启动后自动跳转到 Run 详情。"
      />

      {MOCKS_ENABLED && (
        <p className="rounded-md border border-accent/40 bg-accent-subtle px-3 py-2 text-xs text-accent">
          当前为 MOCK 数据源：启动请求由前端内存后端模拟（api_key 不会被读取），不会真正占用 GPU。
        </p>
      )}

      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* 运行模式 */}
          <Card>
            <CardHeader>
              <CardTitle>运行模式（mode）</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {MODE_OPTIONS.map((option) => (
                <label
                  key={option.value}
                  className={`flex cursor-pointer items-start gap-2.5 rounded-md border px-3 py-2 transition-colors ${
                    mode === option.value
                      ? "border-accent bg-accent-subtle"
                      : "border-border hover:border-border-strong"
                  }`}
                >
                  <input
                    type="radio"
                    name="mode"
                    value={option.value}
                    checked={mode === option.value}
                    onChange={() => setMode(option.value)}
                    className="mt-0.5 accent-[var(--color-accent)]"
                  />
                  <span>
                    <span className="inline-flex items-center gap-2">
                      <span className="font-mono text-sm text-fg">{option.label}</span>
                      {option.demo && <Badge variant="warning">演示 · 非 LIVE_MODEL</Badge>}
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">{option.desc}</span>
                  </span>
                </label>
              ))}
            </CardContent>
          </Card>

          {/* 题目选择 */}
          <Card>
            <CardHeader>
              <CardTitle>题目选择（problem）</CardTitle>
              {problemsData === null && (
                <Button size="sm" variant="secondary" onClick={loadProblems} type="button">
                  加载题目列表（GET /api/problems）
                </Button>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              {problemsError && (
                <ErrorPanel title="题目列表加载失败" message={problemsError} onRetry={loadProblems} />
              )}
              {problemsData && (
                <>
                  <Field label="从列表选择" htmlFor="problem-select">
                    <select
                      id="problem-select"
                      className={inputClass}
                      value={problemOptions.some((p) => p.spec === problem) ? problem : ""}
                      onChange={(e) => setProblem(e.target.value)}
                    >
                      <option value="">— 选择题目 —</option>
                      {problemOptions.map((p) => (
                        <option key={p.spec} value={p.spec}>
                          {p.label} · {p.spec}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <p className="text-[11px] text-text-3">
                    共 {problemOptions.length} 题（{problemsData.bench}）；也可在下方手输 spec。
                  </p>
                </>
              )}
              <Field
                label="问题 spec（problem）"
                htmlFor="problem"
                hint='格式 "kernelbench:l<level>:<id>"，如 kernelbench:l1:40'
              >
                <input
                  id="problem"
                  className={inputClass}
                  value={problem}
                  onChange={(e) => setProblem(e.target.value)}
                  placeholder="kernelbench:l1:40"
                  required
                />
              </Field>
            </CardContent>
          </Card>

          {/* 模型与凭据 */}
          <Card>
            <CardHeader>
              <CardTitle>模型与凭据</CardTitle>
              <span className="text-[11px] text-muted">key 仅驻内存，不落盘、不进日志</span>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-3 rounded-md border border-border bg-surface-2/40 p-3">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label="Base URL" htmlFor="base-url">
                    <input
                      id="base-url"
                      className={inputClass}
                      value={baseUrl}
                      onChange={(e) => setBaseUrl(e.target.value)}
                      placeholder="https://api.deepseek.com"
                    />
                  </Field>
                  <Field
                    label="API Key（可空 → 服务端回退环境变量）"
                    htmlFor="api-key"
                  >
                    <input
                      id="api-key"
                      type="password"
                      className={inputClass}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      autoComplete="off"
                      placeholder="仅本次请求在内存中使用"
                    />
                  </Field>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    loading={modelsLoading}
                    onClick={handleListModels}
                  >
                    用此 Key 拉取模型列表
                  </Button>
                  <span className="text-[11px] text-text-3">
                    key 不写入 localStorage / URL / 日志；拉取成功后填充下方模型候选。
                  </span>
                </div>
                {modelsError && <ErrorPanel title="模型列表获取失败" message={modelsError} />}
                {modelsResult && (
                  <div className="flex flex-wrap gap-1.5">
                    {modelsResult.models.length === 0 ? (
                      <span className="text-xs text-warning">
                        服务端返回空模型列表{modelsResult.error ? `：${modelsResult.error}` : "。"}
                      </span>
                    ) : (
                      modelsResult.models.map((m) => (
                        <Badge key={m.id} variant="primary">
                          <span className="font-mono">{m.id}</span>
                        </Badge>
                      ))
                    )}
                  </div>
                )}
              </div>

              <Field label="模型（model）" htmlFor="model">
                <input
                  id="model"
                  className={inputClass}
                  list="model-options"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="glm-4.5"
                />
                <datalist id="model-options">
                  {(modelsResult?.models ?? []).map((m) => (
                    <option key={m.id} value={m.id} />
                  ))}
                </datalist>
              </Field>
              <Field
                label="disable_thinking"
                htmlFor="disable-thinking"
                hint="关闭 provider 的思考模式（随请求发送 disable_thinking 字段）"
              >
                <label className="flex items-center gap-2 text-sm text-fg">
                  <input
                    id="disable-thinking"
                    type="checkbox"
                    checked={disableThinking}
                    onChange={(e) => setDisableThinking(e.target.checked)}
                    className="h-4 w-4 accent-[var(--color-accent)]"
                  />
                  {disableThinking ? "已关闭思考" : "保持 provider 默认"}
                </label>
              </Field>
            </CardContent>
          </Card>

          {/* 预算 */}
          <Card>
            <CardHeader>
              <CardTitle>预算</CardTitle>
              <span className="text-[11px] text-muted">搜索/调参/修复均计入预算</span>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              <Field label="候选数上限（max_candidates）" htmlFor="max-candidates">
                <input
                  id="max-candidates"
                  type="number"
                  min={1}
                  className={inputClass}
                  value={maxCandidates}
                  onChange={(e) => setMaxCandidates(Number(e.target.value))}
                />
              </Field>
              <Field label="修复轮数上限（max_repair_rounds）" htmlFor="max-repair">
                <input
                  id="max-repair"
                  type="number"
                  min={0}
                  className={inputClass}
                  value={maxRepairRounds}
                  onChange={(e) => setMaxRepairRounds(Number(e.target.value))}
                />
              </Field>
              <Field label="GPU 秒预算（gpu_budget_seconds）" htmlFor="gpu-budget">
                <input
                  id="gpu-budget"
                  type="number"
                  min={1}
                  className={inputClass}
                  value={gpuBudgetSeconds}
                  onChange={(e) => setGpuBudgetSeconds(Number(e.target.value))}
                />
              </Field>
              <Field label="Token 预算（token_budget）" htmlFor="token-budget">
                <input
                  id="token-budget"
                  type="number"
                  min={1}
                  className={inputClass}
                  value={tokenBudget}
                  onChange={(e) => setTokenBudget(Number(e.target.value))}
                />
              </Field>
            </CardContent>
          </Card>

          {/* 高级 */}
          <Card>
            <CardHeader>
              <CardTitle>高级</CardTitle>
            </CardHeader>
            <CardContent>
              <Field
                label="候选后端（backend）"
                htmlFor="backend"
                hint="当前版本服务端仅接入 triton；其他值由服务端校验并可能拒绝"
              >
                <select
                  id="backend"
                  className={inputClass}
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                >
                  <option value="triton">triton（当前唯一接入）</option>
                </select>
              </Field>
            </CardContent>
          </Card>

          {submitError && (
            <ErrorPanel
              title={
                submitError.status === 409
                  ? "GPU 被占用（409）：单 GPU 单运行"
                  : `启动被拒绝（HTTP ${submitError.status}）`
              }
              message={submitError.message}
            />
          )}
          {submitted && !submitError && (
            <p className="rounded-md border border-success/30 bg-success-soft px-3 py-2 text-xs text-ok">
              已提交，正在跳转到 Run 详情…
            </p>
          )}

          <div className="flex items-center gap-3">
            <Button type="submit" variant="primary" loading={submitting} disabled={!problem.trim()}>
              开始优化（POST /api/runs）
            </Button>
            <span className="text-[11px] text-text-3">
              提交成功后跳转 Run 详情并立即清空 key 输入框。
            </span>
          </div>
        </form>

        {/* 粘性预览卡 */}
        <div className="xl:sticky xl:top-20 xl:self-start">
          <Card>
            <CardHeader>
              <CardTitle>请求预览</CardTitle>
              <Badge variant="neutral">POST /api/runs</Badge>
            </CardHeader>
            <CardContent className="space-y-2">
              <pre className="max-h-80 overflow-auto rounded-md border border-border bg-log-bg p-2.5 font-mono text-[11px] leading-relaxed text-muted">
                {JSON.stringify(previewPayload, null, 2)}
              </pre>
              <ul className="space-y-1 text-[11px] leading-relaxed text-muted">
                <li>· api_key 仅驻内存：不写 localStorage / URL / 日志，成功后清空。</li>
                <li>· 409 = GPU 已被活跃 run 占用；400 = 配置非法（均显示服务端原文）。</li>
                <li>· demo 模式不得记为 LIVE_MODEL（页面上有常驻标注）。</li>
                <li className="flex items-center gap-1.5">
                  · 无后端取消机制，启动后没有"停止"按钮
                  <NotRunBadge what="取消能力（后端未提供）" />
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
