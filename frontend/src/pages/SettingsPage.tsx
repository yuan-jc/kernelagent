import { useState } from "react";
import { API_BASE_URL, BACKEND_PORT } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyState,
} from "../components/ui";

type Theme = "dark" | "light";

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("ka-theme", theme);
  } catch {
    // localStorage 不可用时仅本次会话生效
  }
}

/**
 * 设置页（spec §3.8）：仅本地偏好（localStorage），不与后端交互，
 * 不含任何 key；主题"暗色为默认，保留 light 钩子"。
 */
export function SettingsPage() {
  const [theme, setTheme] = useState<Theme>(currentTheme);

  return (
    <div className="space-y-5">
      <PageHeader
        title="设置"
        description="仅本地偏好（localStorage）；评测协议、容差与预算默认值属于服务端/协议层，前端不提供修改入口。Provider 预设只保存名称与 base_url，绝不保存 key。"
      />

      <Card>
        <CardHeader>
          <CardTitle>外观</CardTitle>
          <CardDescription>暗色为默认主题，写入 localStorage</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            {(["dark", "light"] as const).map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => {
                  setTheme(option);
                  applyTheme(option);
                }}
                className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                  theme === option
                    ? "border-accent bg-accent-subtle text-accent"
                    : "border-border bg-surface-2 text-muted hover:text-fg"
                }`}
              >
                {option === "dark" ? "暗色（默认）" : "亮色"}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>本地偏好</CardTitle>
            <Badge variant="neutral">下一工作包实现</Badge>
          </CardHeader>
          <CardContent>
            <EmptyState
              className="border-border"
              title="偏好项建设中"
              description="将提供：默认预算值草稿、Provider 预设（名称 + base_url，无 key）、轮询间隔（1.5s/3s/5s）、日志 auto-scroll 默认值。"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>后端连接（只读）</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs leading-relaxed text-muted">
            <p>
              API base URL 集中管理于{" "}
              <code className="font-mono text-fg">src/api/client.ts</code>：
              当前为{" "}
              <code className="font-mono text-fg">
                {API_BASE_URL === "" ? "同源（空字符串）" : API_BASE_URL}
              </code>
              。
            </p>
            <p>
              开发模式（<code className="font-mono text-fg">npm run dev</code>）
              下 Vite 把 <code className="font-mono text-fg">/api</code>{" "}
              代理到 server.py 默认端口{" "}
              <code className="font-mono text-fg">
                127.0.0.1:{BACKEND_PORT}
              </code>
              （可用环境变量{" "}
              <code className="font-mono text-fg">KA_BACKEND_PORT</code>{" "}
              覆盖）。
            </p>
            <p>
              如需把前端指向其他后端实例，设置环境变量{" "}
              <code className="font-mono text-fg">VITE_API_BASE_URL</code>{" "}
              后重新构建；不要把任何密钥写入前端代码或构建配置。
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
