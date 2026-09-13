import { useState } from "react";
import { API_BASE_URL, BACKEND_PORT } from "../api";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui";
import { MOCKS_ENABLED } from "../mocks";
import { prefs, type LaunchDefaults, type PollInterval, type ProviderPreset } from "../lib/prefs";

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

const inputClass =
  "h-8 w-full rounded-md border border-border bg-surface-3 px-2.5 font-mono text-xs text-fg placeholder:text-text-3 focus:border-accent focus:outline-none";

/**
 * 设置页（spec §3.8）：仅本地偏好（localStorage），不与后端交互，
 * 不含任何 key。candidate_trust=cooperative 警示为常驻信息，不可移除。
 */
export function SettingsPage() {
  const [theme, setTheme] = useState<Theme>(currentTheme);
  const [pollInterval, setPollIntervalState] = useState<PollInterval>(prefs.getPollInterval());
  const [autoscroll, setAutoscrollState] = useState(prefs.getLogAutoscroll());
  const [defaults, setDefaults] = useState<LaunchDefaults>(prefs.getLaunchDefaults());
  const [presets, setPresets] = useState<ProviderPreset[]>(prefs.getProviderPresets());
  const [newPreset, setNewPreset] = useState<ProviderPreset>({ name: "", baseUrl: "" });
  const [saved, setSaved] = useState(false);
  const [mockNext, setMockNext] = useState<boolean>(MOCKS_ENABLED);

  function flashSaved() {
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="设置"
        description="仅本地偏好（localStorage）；评测协议、容差与预算默认值属于服务端/协议层，前端不提供修改入口。Provider 预设只保存名称与 base_url，绝不保存 key。"
        actions={saved ? <Badge variant="success">已保存</Badge> : undefined}
      />

      <Card className="border-warning/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Badge variant="warning">常驻警示</Badge>
            candidate_trust = cooperative
          </CardTitle>
        </CardHeader>
        <CardContent className="text-xs leading-relaxed text-muted">
          自动评估为协作式（非对抗式验证）：<span className="text-fg">champion 必须经人工审查后方可采信</span>。
          该警示在顶栏与所有 completed run 详情页常驻显示；此设置页不提供关闭入口
          （诚实性约束，不属于本地偏好）。
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 数据源 */}
        <Card>
          <CardHeader>
            <CardTitle>数据源（mock / 真实后端）</CardTitle>
            <Badge variant={MOCKS_ENABLED ? "primary" : "neutral"}>
              当前：{MOCKS_ENABLED ? "MOCK" : "真实后端"}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-2.5 text-xs leading-relaxed text-muted">
            <p>
              MOCK 数据源用于在 C3 的 P2–P5 端点落地前开发与演示：全部请求由
              <code className="mx-1 font-mono text-fg">frontend/src/mocks/</code>
              的内存后端应答（含时间线驱动的"活"run）。
            </p>
            <div className="flex items-center gap-2">
              <label className="flex cursor-pointer items-center gap-2 text-fg">
                <input
                  type="checkbox"
                  checked={mockNext}
                  onChange={(e) => {
                    setMockNext(e.target.checked);
                    try {
                      window.localStorage.setItem("ka-use-mocks", e.target.checked ? "1" : "0");
                    } catch {
                      // ignore
                    }
                  }}
                  className="h-4 w-4 accent-[var(--color-accent)]"
                />
                使用 MOCK 数据源（下次加载生效）
              </label>
              <Button size="sm" variant="secondary" onClick={() => window.location.reload()}>
                立即重载
              </Button>
            </div>
            <p className="text-[11px] text-text-3">
              也可用构建期环境变量 <code className="font-mono">VITE_USE_MOCKS=1</code>{" "}
              强制（优先级高于 localStorage）；见 README「mock 与真实数据切换」。
            </p>
          </CardContent>
        </Card>

        {/* 外观与轮询 */}
        <Card>
          <CardHeader>
            <CardTitle>外观与刷新</CardTitle>
            <CardDescription>暗色默认；主题写入 localStorage</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
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
            <div>
              <p className="mb-1 text-xs font-medium text-muted">Run 详情轮询间隔（running 时）</p>
              <div className="flex gap-2">
                {([1500, 3000, 5000] as const).map((ms) => (
                  <button
                    key={ms}
                    type="button"
                    onClick={() => {
                      setPollIntervalState(ms);
                      prefs.setPollInterval(ms);
                      flashSaved();
                    }}
                    className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                      pollInterval === ms
                        ? "border-accent bg-accent-subtle text-accent"
                        : "border-border bg-surface-2 text-muted hover:text-fg"
                    }`}
                  >
                    {(ms / 1000).toFixed(1)}s
                  </button>
                ))}
              </div>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-fg">
              <input
                type="checkbox"
                checked={autoscroll}
                onChange={(e) => {
                  setAutoscrollState(e.target.checked);
                  prefs.setLogAutoscroll(e.target.checked);
                  flashSaved();
                }}
                className="h-4 w-4 accent-[var(--color-accent)]"
              />
              日志查看器默认自动滚动
            </label>
          </CardContent>
        </Card>

        {/* 默认预算草稿 */}
        <Card>
          <CardHeader>
            <CardTitle>新建运行默认值（表单预填草稿）</CardTitle>
            <CardDescription>仅预填 /launch 表单；服务端默认值不变</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <div className="grid grid-cols-2 gap-2">
              {(
                [
                  ["maxCandidates", "候选数上限"],
                  ["maxRepairRounds", "修复轮上限"],
                  ["gpuBudgetSeconds", "GPU 秒预算"],
                  ["tokenBudget", "Token 预算"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="block">
                  <span className="mb-1 block text-[11px] text-muted">{label}</span>
                  <input
                    type="number"
                    min={1}
                    className={inputClass}
                    value={defaults[key]}
                    onChange={(e) => setDefaults({ ...defaults, [key]: Number(e.target.value) })}
                  />
                </label>
              ))}
            </div>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                prefs.setLaunchDefaults(defaults);
                flashSaved();
              }}
            >
              保存默认值
            </Button>
          </CardContent>
        </Card>

        {/* Provider 预设（无 key） */}
        <Card>
          <CardHeader>
            <CardTitle>Provider 预设（名称 + base_url）</CardTitle>
            <CardDescription>不保存任何 key</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <ul className="space-y-1.5">
              {presets.map((preset, i) => (
                <li key={`${preset.name}-${i}`} className="flex items-center justify-between gap-2 rounded-md border border-border bg-surface-2 px-2.5 py-1.5">
                  <div className="min-w-0">
                    <p className="text-xs text-fg">{preset.name}</p>
                    <p className="truncate font-mono text-[11px] text-muted">{preset.baseUrl}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        const next = presets.filter((_, j) => j !== i);
                        setPresets(next);
                        prefs.setProviderPresets(next);
                        flashSaved();
                      }}
                    >
                      删除
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
            <div className="grid grid-cols-[1fr_1.4fr] gap-2">
              <input
                className={inputClass}
                placeholder="名称"
                value={newPreset.name}
                onChange={(e) => setNewPreset({ ...newPreset, name: e.target.value })}
              />
              <input
                className={inputClass}
                placeholder="https://api.example.com"
                value={newPreset.baseUrl}
                onChange={(e) => setNewPreset({ ...newPreset, baseUrl: e.target.value })}
              />
            </div>
            <Button
              size="sm"
              variant="secondary"
              disabled={!newPreset.name.trim() || !newPreset.baseUrl.trim()}
              onClick={() => {
                const next = [...presets, { name: newPreset.name.trim(), baseUrl: newPreset.baseUrl.trim() }];
                setPresets(next);
                prefs.setProviderPresets(next);
                setNewPreset({ name: "", baseUrl: "" });
                flashSaved();
              }}
            >
              添加预设
            </Button>
          </CardContent>
        </Card>
      </div>

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
            <code className="font-mono text-fg">127.0.0.1:{BACKEND_PORT}</code>
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
  );
}
