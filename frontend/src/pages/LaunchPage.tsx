import { useState } from "react";
import type { FormEvent } from "react";
import { api, toApiError } from "../api";
import type { ListModelsResponse } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorPanel,
  Table,
  TableWrapper,
  TBody,
  TD,
  TH,
  THead,
  TR,
} from "../components/ui";

/**
 * server.py _JOB_FIELDS_DEFAULTS 的只读镜像（仅用于展示说明，
 * 真正的默认值以服务端为准）。
 */
const SERVER_DEFAULTS = {
  max_candidates: 5,
  max_repair_rounds: 2,
  gpu_budget_seconds: 1800,
  token_budget: 200000,
} as const;

const MODE_DOCS: Array<{
  mode: string;
  desc: string;
  variant: "primary" | "warning" | "neutral";
}> = [
  {
    mode: "live",
    desc: "真实模型生成（需 API key 或服务端环境变量）；唯一可计为 LIVE_MODEL 的模式",
    variant: "primary",
  },
  {
    mode: "demo-correct",
    desc: "演示·正确候选：真 GPU 上验证流程，不依赖 provider，不得记为 LIVE_MODEL",
    variant: "warning",
  },
  {
    mode: "demo-wrong",
    desc: "演示·错误候选：验证修复/拒绝路径，不得记为 LIVE_MODEL",
    variant: "neutral",
  },
];

export function LaunchPage() {
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ListModelsResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);

  async function handleListModels(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.listModels({
        base_url: baseUrl.trim(),
        // key 只随本次请求发往本地 server，进入其内存；前端不存储。
        api_key: apiKey,
      });
      setResult(response);
    } catch (cause) {
      setError(toApiError(cause));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="新建运行"
        description="运行模式、题目、模型与凭据、预算配置。API key 只随单次请求进入服务端内存，绝不落盘（job.json 不含 key）；单 GPU 同时只能有一个运行，重复启动返回 409。"
      />

      <Card>
        <CardHeader>
          <CardTitle>启动表单</CardTitle>
          <Badge variant="neutral">POST /api/runs · 下一工作包实现</Badge>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="启动表单建设中"
            description="将按设计规范 §3.5 提供：运行模式、bench/level/problem 级联选择、模型与凭据（提交成功后立即清空 key 输入框）、预算与高级字段（backend、disable_thinking），并展示 400/409 错误原文。"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API / 模型连接</CardTitle>
          <Badge variant="neutral">POST /api/models</Badge>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleListModels} className="space-y-4">
            <div>
              <label
                htmlFor="base-url"
                className="mb-1 block text-xs font-medium text-muted"
              >
                Base URL
              </label>
              <input
                id="base-url"
                type="text"
                required
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder="https://api.example.com/v1"
                className="h-9 w-full rounded-md border border-border bg-surface-3 px-3 font-mono text-sm text-fg placeholder:text-text-3 focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label
                htmlFor="api-key"
                className="mb-1 block text-xs font-medium text-muted"
              >
                API Key（可选；留空时服务端回退到控制面环境变量）
              </label>
              <input
                id="api-key"
                type="password"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                autoComplete="off"
                placeholder="仅本次请求在内存中使用"
                className="h-9 w-full rounded-md border border-border bg-surface-3 px-3 font-mono text-sm text-fg placeholder:text-text-3 focus:border-accent focus:outline-none"
              />
            </div>
            <div className="flex items-center gap-3">
              <Button type="submit" variant="primary" loading={loading}>
                用此 Key 拉取模型列表
              </Button>
              <span className="text-[11px] text-muted">
                key 不写入 localStorage / URL / 日志。
              </span>
            </div>
          </form>

          {error && (
            <ErrorPanel className="mt-4" title="模型列表获取失败" error={error} />
          )}

          {result && (
            <div className="mt-4">
              {result.models.length === 0 ? (
                <p className="text-xs text-warning">
                  服务端返回空模型列表{result.error ? `：${result.error}` : "。"}
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {result.models.map((model) => (
                    <Badge key={model.id} variant="primary">
                      <span className="font-mono">{model.id}</span>
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>运行模式（POST /api/runs 的 mode）</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {MODE_DOCS.map((doc) => (
              <div key={doc.mode} className="flex items-start gap-2.5">
                <Badge variant={doc.variant}>
                  <span className="font-mono">{doc.mode}</span>
                </Badge>
                <p className="text-xs leading-relaxed text-muted">{doc.desc}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>服务端默认预算</CardTitle>
            <CardDescription>server.py 的默认值；每次启动可覆盖</CardDescription>
          </CardHeader>
          <CardContent>
            <TableWrapper>
              <Table>
                <THead>
                  <TR>
                    <TH>字段</TH>
                    <TH>默认值</TH>
                  </TR>
                </THead>
                <TBody>
                  <TR>
                    <TD className="font-mono text-xs tabular-nums">max_candidates</TD>
                    <TD className="font-mono text-xs tabular-nums">{SERVER_DEFAULTS.max_candidates}</TD>
                  </TR>
                  <TR>
                    <TD className="font-mono text-xs">max_repair_rounds</TD>
                    <TD className="font-mono text-xs tabular-nums">{SERVER_DEFAULTS.max_repair_rounds}</TD>
                  </TR>
                  <TR>
                    <TD className="font-mono text-xs">gpu_budget_seconds</TD>
                    <TD className="font-mono text-xs tabular-nums">{SERVER_DEFAULTS.gpu_budget_seconds}</TD>
                  </TR>
                  <TR>
                    <TD className="font-mono text-xs">token_budget</TD>
                    <TD className="font-mono text-xs tabular-nums">{SERVER_DEFAULTS.token_budget}</TD>
                  </TR>
                </TBody>
              </Table>
            </TableWrapper>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
