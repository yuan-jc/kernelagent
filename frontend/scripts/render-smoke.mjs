/**
 * 渲染冒烟测试（无浏览器）：esbuild 把 src/main.tsx 打成 IIFE，
 * 在 jsdom 里执行，window.fetch 转发到真实后端；逐路由断言渲染内容。
 *
 * 用法：
 *   node scripts/render-smoke.mjs [--mock]
 *   默认打真实后端（KA_SMOKE_BASE_URL，默认 http://127.0.0.1:8501，
 *   需要先启动 `.venv/bin/python -m kernelagent.webapp`）；
 *   --mock 时设置 localStorage ka-use-mocks=1（mock 模式不依赖后端）。
 *
 * 该脚本仅用于开发期验证；不属于构建/运行时依赖。
 */

import { build } from "esbuild";
import { JSDOM } from "jsdom";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const MOCK = process.argv.includes("--mock");
const BASE = process.env.KA_SMOKE_BASE_URL ?? "http://127.0.0.1:8501";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 1) 打包应用（IIFE，css 置空）
await build({
  entryPoints: [resolve(here, "../src/main.tsx")],
  bundle: true,
  format: "iife",
  outfile: "/tmp/ka-render-smoke-bundle.js",
  loader: { ".css": "empty" },
  define: {
    "process.env.NODE_ENV": '"production"',
    "import.meta.env.VITE_API_BASE_URL": "undefined",
    "import.meta.env.VITE_USE_MOCKS": "undefined",
  },
  jsx: "automatic",
  logLevel: "silent",
});
const bundle = readFileSync("/tmp/ka-render-smoke-bundle.js", "utf8");

// 2) jsdom 宿主
const dom = new JSDOM(`<!doctype html><html><body><div id="root"></div></body></html>`, {
  url: `${BASE}/`,
  runScripts: "outside-only",
  pretendToBeVisual: true,
});
const { window } = dom;

// fetch 转发到后端（相对路径 -> 绝对路径）；react-router v7 还需要
// Request/Response/Headers/TextDecoder 等全局，jsdom 缺失，从 Node 注入。
for (const key of ["Request", "Response", "Headers", "fetch", "FormData", "TextDecoder", "TextEncoder", "AbortController", "AbortSignal", "URL"]) {
  if (typeof globalThis[key] === "function") {
    // 无条件覆盖：undici 的 brand check 只认同 realm 的实例
    // （jsdom 的 AbortSignal 传入 Node 的 Request 会抛错）。
    window[key] = globalThis[key];
  }
}
window.fetch = async (input, init) => {
  let url = typeof input === "string" ? input : input instanceof URL ? input.href : input?.url ?? String(input);
  if (url.startsWith("/")) url = `${BASE}${url}`;
  return fetch(url, init);
};
// jsdom 没有 requestIdleCallback
window.requestIdleCallback ??= (cb) => window.setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 50 }), 0);
window.cancelIdleCallback ??= (id) => window.clearTimeout(id);

if (MOCK) {
  window.localStorage.setItem("ka-use-mocks", "1");
}

let consoleErrors = [];
window.console.error = (...args) => {
  consoleErrors.push(args.map((a) => (a instanceof Error ? a.message : String(a))).join(" "));
};
window.addEventListener("error", (e) => consoleErrors.push(`window.onerror: ${e.message}`));

window.eval(bundle);
await sleep(400);

const RESULTS = [];
function check(name, ok, detail = "") {
  RESULTS.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : `  -> ${detail}`}`);
}

async function go(hash, waitMs = 1800) {
  window.location.hash = hash;
  await sleep(waitMs);
  return window.document.body.textContent ?? "";
}

// 取 run id：真实模式走 API；mock 模式下应用层使用 mock 数据（网络层
// 仍是真实后端），所以必须从渲染出的 Runs 表里提取。
let runId = null;
if (!MOCK) {
  const runsRes = await fetch(`${BASE}/api/runs`);
  const runsJson = await runsRes.json();
  runId = runsJson.runs?.[0]?.run_id ?? null;
} else {
  const t = await go("#/runs");
  runId = t.match(/\d{8}-\d{6}/)?.[0] ?? null;
  await go("#/");
}
if (!runId) {
  console.error("没有提取到任何 run id，无法继续");
  process.exit(1);
}
console.log(`(smoke) runId = ${runId}`);

let text = await go("#/");
check("Dashboard: 标题渲染", text.includes("总览") && text.includes("系统健康"), text.slice(0, 120));
check("Dashboard: 最近 Runs 表", text.includes("最近 Runs"), "");
check("Dashboard: trust 常驻警示", text.includes("candidate_trust=cooperative"), "");
check("Dashboard: 统计瓦片（缺失加速比=NOT_RUN）", text.includes("最佳加速比") && text.includes("NOT_RUN"), "");
if (!MOCK) check("Dashboard: 真实 GPU 设备", text.includes("GPU-ea248ec5"), "");
check("Dashboard: run id 可见", text.includes(runId), "");

text = await go("#/runs");
check("Runs: 筛选条", text.includes("状态：全部") && text.includes("模式：全部"), "");
check("Runs: 表格含真实 run", text.includes(runId), "");
check("Runs: 时间列有值或 —", /\d+[smhd] ago|—/u.test(text), "");

text = await go(`#/runs/${runId}`);
check("RunDetail: 状态徽章与 run id", text.includes(runId), "");
check("RunDetail: 泳道卡", text.includes("候选流水"), "");
check("RunDetail: 预算面板", text.includes("GPU 秒") && text.includes("Token"), "");
check("RunDetail: 六阶段 stepper", text.includes("晋升确认") && text.includes("容器评测"), "");
check("RunDetail: 终态/审查说明", text.includes("需人工审查") || text.includes("合法结果"), "");

// tabs
const clickTab = async (label) => {
  const btns = [...window.document.querySelectorAll("button[role=tab]")];
  const target = btns.find((b) => b.textContent?.includes(label));
  if (!target) throw new Error(`tab ${label} not found`);
  target.click();
  await sleep(1800);
  return window.document.body.textContent ?? "";
};

text = await clickTab("候选对比");
check("RunDetail/候选对比: 表头与 CI 或 NOT_RUN", text.includes("速度比 95% CI") && (text.includes("×") || text.includes("NOT_RUN")), "");
check("RunDetail/候选对比: baseline 行或占位说明", text.includes("baseline-eager") || text.includes("暂无候选记录"), "");

text = await clickTab("日志与产物");
check("RunDetail/日志: journal 事件流渲染", text.includes("budget_reserved") || text.includes("不可用") || text.includes("暂无日志"), JSON.stringify(text.slice(0, 200)));
check("RunDetail/日志: 自动滚动开关", text.includes("自动滚动"), "");
check("RunDetail/日志: 数据来源徽章", text.includes("P3 增量") || text.includes("全量兜底") || text.includes("不可用"), "");

text = await clickTab("证据");
check("RunDetail/证据: 快照证据字段", text.includes("candidate_trust") && text.includes("adversarially_secure"), "");
check(
  "RunDetail/证据: report 身份链或诚实降级",
  /commit|report\.json|report 端点/i.test(text),
  text.slice(0, 160),
);

text = await go(`#/runs/${runId}/profile`);
check("Profile: 页面标题", text.includes("Profile 报告"), "");
check("Profile: NOT_RUN 徽章或批次图", text.includes("NOT_RUN") || text.includes("batch 1.."), "");
check("Profile: 容器日志区", text.includes("容器日志"), "");

text = await go("#/launch");
check("Launch: 模式单选", text.includes("demo-correct") && text.includes("演示 · 非 LIVE_MODEL"), "");
check("Launch: 表单字段", text.includes("Base URL") && text.includes("API Key") && text.includes("token_budget"), "");
check("Launch: 请求预览（key 打码说明）", text.includes("请求预览") && text.includes("api_key 仅驻内存"), "");

text = await go("#/benchmarks");
check("Benchmarks: KernelBench 已接入", text.includes("KernelBench") && text.includes("已接入"), "");
check("Benchmarks: Planned 卡（MKB/rk/user-bench）", text.includes("MKB") && text.includes("Planned") && text.includes("user-bench"), "");
check("Benchmarks: 去运行链接", text.includes("去运行"), "");

text = await go("#/settings");
check("Settings: 数据源切换", text.includes("数据源") && text.includes("MOCK"), "");
check("Settings: cooperative 常驻卡", text.includes("candidate_trust = cooperative"), "");
check("Settings: Provider 预设", text.includes("Provider 预设"), "");

// key 卫生：localStorage 中不允许出现任何 key 值痕迹
const lsDump = JSON.stringify(Object.fromEntries(Object.entries(window.localStorage)));
check("安全: localStorage 不含 ka-* 以外的可疑键 / 无 api_key 字段", !lsDump.includes("api_key") && !lsDump.includes("sk-"), "");

// Launch E2E：仅 mock 模式执行（绝不向真实后端提交启动请求，避免占用真 GPU）。
// 注意：mock 世界预置了一个"进行中"的演示 run，会按单 GPU 约束返回 409；
// 这里轮询重试直到演示 run 结束、GPU 释放（也是对 409 行为的验证）。
if (MOCK) {
  const setInput = (id, value) => {
    const el = [...window.document.querySelectorAll("input")].find((i) => i.id === id);
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new window.Event("input", { bubbles: true }));
  };
  let navigated = false;
  let sawBusy409 = false;
  for (let attempt = 0; attempt < 40 && !navigated; attempt++) {
    await go("#/launch", 600);
    setInput("problem", "kernelbench:l1:40");
    await sleep(120);
    [...window.document.querySelectorAll("button")].find((b) => b.textContent?.includes("开始优化")).click();
    await sleep(2000);
    if (window.location.hash.includes("/runs/")) {
      navigated = true;
    } else if ((window.document.body.textContent ?? "").includes("still using the GPU")) {
      sawBusy409 = true;
    }
  }
  check("Launch E2E: 忙碌时 409、GPU 释放后提交成功", navigated && sawBusy409, `navigated=${navigated} saw409=${sawBusy409}`);
  check("Launch E2E: 跳转到新 Run 详情", /#\/runs\/\d{8}-\d{6}/.test(window.location.hash), window.location.hash);
  const t2 = window.document.body.textContent ?? "";
  check("Launch E2E: 新 run 泳道渲染", t2.includes("候选流水"), "");
  // 单 GPU 单运行：新 run 进行中再次启动应 409
  await go("#/launch", 800);
  setInput("problem", "kernelbench:l1:40");
  await sleep(150);
  [...window.document.querySelectorAll("button")].find((b) => b.textContent?.includes("开始优化")).click();
  await sleep(1800);
  const t3 = window.document.body.textContent ?? "";
  check("Launch E2E: 二次启动展示 409 错误", t3.includes("GPU 被占用") && t3.includes("still using the GPU"), "");
}

const realErrors = consoleErrors.filter(
  (e) => !e.includes("Warning:") && !e.includes("Not implemented") && !e.includes("Could not parse CSS"),
);
check("无 console error", realErrors.length === 0, realErrors.slice(0, 3).join(" | "));

const failed = RESULTS.filter((r) => !r.ok);
console.log(`\n${RESULTS.length - failed.length}/${RESULTS.length} checks passed`);
process.exit(failed.length === 0 ? 0 : 1);
