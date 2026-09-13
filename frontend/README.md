# kernelagent 前端（React + Vite + TypeScript + Tailwind v4）

kernelagent 本地控制台前端。后端是标准库 HTTP server：
`src/kernelagent/webapp/server.py`（默认 `127.0.0.1:8501`）。

技术栈：Vite 8 · React 19 · TypeScript（strict）· Tailwind CSS v4 · react-router v7（hash 路由）。

**遵循的设计规范**：`research/frontend/frontend-design-spec.md`（A1 调研产出）。
本阶段完成其 §8.4 的第 2–6 步：全部 7 个页面从占位做成完整实现，
含阶段时间线/泳道、日志查看器、错误三层呈现、mock 数据源与渲染冒烟测试。
与规范的已知偏差见文末。

## 页面与数据源（全部已接通）

| 路由 | 页面 | 状态 |
| --- | --- | --- |
| `/` | Dashboard 总览 | 系统健康卡、活跃 run 卡（1.5s 轮询：阶段 stepper + 双预算条）、统计瓦片（缺失加速比显示 NOT_RUN）、最近 8 条 runs |
| `/runs` | Runs 列表 | 状态/模式/关键字筛选（localStorage 持久化）+ 状态徽章 + 客户端分页（15/页）；时间列优先 `started_at`，缺失回退 run_id 编码时间，再缺失 "—" |
| `/runs/:runId` | Run 详情 | 顶部状态 + 终态解释卡（no_improvement/budget_exhausted 合法结果文案）+ 基础设施错误横幅（error_tail 尾 300 字直接可见、可展开/复制）；Tab：概览（预算/配置/泳道/单候选六阶段 stepper/Champion 卡）、候选对比（champion vs baseline，CI 缺失 NOT_RUN，detail 400 字截断可展开）、日志与产物（P3 journal 事件流 + 智能跟随/auto-scroll 开关；P5 workspace 文件只读查看）、证据（P2 身份链 + 快照证据）；1.5s 轮询、终态停止、隐藏暂停、连续失败退避 |
| `/runs/:runId/profile` | Profile 报告 | P4 records 驱动的批次延迟散点图（手写 SVG）+ 样本统计（n/min/mean/p95/max）+ async_leak 徽章 + 容器 stdout/stderr 日志查看器；无数据整段 NOT_RUN 空态，不画空图 |
| `/launch` | 新建运行 | POST /api/runs 全字段：mode（demo 常驻"演示·非 LIVE_MODEL"标注）、problem 级联（/api/problems + ?problem= 预填）、model/base_url/api_key（拉模型列表填充 datalist）、四项预算、backend、disable_thinking；粘性请求预览卡（key 打码）；409/400 展示服务端原文；成功后跳转 Run 详情并立即清空 key |
| `/benchmarks` | Benchmark 管理 | KernelBench 题目树（level 折叠、过滤、"去运行"预填跳转）；MKB / rk / user-bench 以 "Planned" 卡诚实展示（端点未实现，不提供假数据） |
| `/settings` | 设置 | 数据源切换（mock/真实）、轮询间隔（1.5/3/5s）、日志 auto-scroll 默认、主题（暗色默认）、新建运行默认值草稿、Provider 预设（名称+base_url，无 key）、candidate_trust 常驻警示说明、后端地址说明 |

全局：顶栏常驻 `candidate_trust=cooperative · champion 需人工审查` 警示徽章
（不可移除/弱化）；mock 模式下追加 MOCK 徽章；窄屏（<md）折叠为水平导航，
1280+ 为主力布局。

## mock 与真实数据切换

**开关优先级**（实现于 `src/mocks/switch.ts`，唯一出处）：

1. 构建期环境变量 `VITE_USE_MOCKS=1`（或 `=0` 强制关闭）——dev 与 build 都生效；
2. 运行时 `localStorage["ka-use-mocks"] = "1"/"0"`（设置页「数据源」可切换，立即重载生效）；
3. 默认关闭 = 真实后端。

```bash
# mock 模式开发（不需要后端）
VITE_USE_MOCKS=1 npm run dev

# mock 模式构建（单文件产物自包含 mock，适合纯前端演示）
VITE_USE_MOCKS=1 npm run build
```

**mock 覆盖的端点**（`src/mocks/mockApi.ts`；打开开关后 client.ts 全部路由进内存后端）：

- 现有 6 端点：health / problems / runs / run detail / POST runs（含 409 单 GPU 约束与 400 校验）/ models；
- C3 新端点：P2 report、P3 journal（支持 `?after=` 增量）、P4 records 清单与单条、P5 workspace 清单与文件。

**mock 数据构成**：

- 4 个历史终态 run：completed（有 champion+CI）、no_improvement、budget_exhausted、infra_error（error.txt 尾）；
- 1 个"时间线驱动"的活 run：模块加载即启动，约 110s 走完
  generate→policy→evaluate→correctness_pro→timing→confirm，候选 000 在
  correctness 失败、候选 001 晋升；期间 budget 结算、journal 事件、records、
  workspace 产物按事件逐步出现 —— 与真实 run 的数据形态一致
  （journal 无时间戳、运行中 report.candidates 为空等）；
- Launch 页提交会创建新的时间线 run（demo-wrong 结局为 no_improvement），
  mock 的 api_key 字段**有意不读取**（mock 不接触任何凭据）。

**等 C3 端点的部分已不需要**：本阶段联调时 C3 的 P1–P5 已在真实后端落地
（journal/report/records/workspace 全部实测 200），mock 仅作为离线开发与
演示用途保留。

## 开发

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173（/api 代理到 127.0.0.1:8501）
```

- 换后端端口：`KA_BACKEND_PORT=xxxx npm run dev`
- npm 网络慢：`HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 npm install`
- API base URL 集中在 `src/api/client.ts`，可用 `VITE_API_BASE_URL` 覆盖（构建期注入）。
  **不要把任何密钥写进 env 文件或代码。**

## 构建与产物托管

```bash
npm run build        # tsc -b && vite build -> dist/index.html（单文件）
npm run preview      # 本地预览构建产物
```

`vite.config.ts` 启用 `vite-plugin-singlefile`：JS/CSS 全部内联，产物是
自包含的 `dist/index.html`。前端使用 hash 路由（`#/runs/xxx`），拷贝一个
文件即可被现有 server.py 托管：

```bash
cp frontend/dist/index.html src/kernelagent/webapp/static/index.html
```

（规范提案 P1 的静态资源 + 路由回退落地后，可去掉 singlefile、把
`src/App.tsx` 的 `createHashRouter` 换成 `createBrowserRouter`。）

## 渲染冒烟测试（无浏览器）

`scripts/render-smoke.mjs` 用 esbuild 把应用打成 IIFE，在 jsdom 里真实执行
（fetch 转发到后端），逐路由断言渲染内容与安全约束：

```bash
node scripts/render-smoke.mjs          # 打真实后端（需先启动 .venv/bin/python -m kernelagent.webapp）
node scripts/render-smoke.mjs --mock   # 纯 mock 模式（不需要后端；含 Launch 提交 E2E + 409 流程）
```

- 真实后端：**35/35 通过**（含 journal 增量徽章、records 图表、workspace 文件、证据链 commit）；
- mock 模式：**38/38 通过**（另含提交→跳转→泳道→二次启动 409 的 E2E，
  以及「localStorage 中不出现 api_key」的安全断言）；
  mock 的启动 E2E 会先遇到预置演示 run 的 409，轮询等 GPU 释放后成功——
  这本身是对单 GPU 约束的验证。
- 两个模式都断言「无 console error」。

## API 覆盖（src/api/client.ts）

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/api/health` | GET | 存活、GPU 设备、活跃 run（顶栏 10s 轮询） |
| `/api/problems` | GET | bench/level/problem 题目树（Benchmarks 页 / Launch 级联） |
| `/api/runs` | GET | 运行历史摘要（新在前；含 C3 新增 `started_at`） |
| `/api/runs/<id>` | GET | 快照：状态/预算/候选/in-flight/progress/error 尾/job（不含 key） |
| `/api/runs/<id>/report` | GET | P2：report.json 原文（证据 tab 身份链；缺失时诚实降级） |
| `/api/runs/<id>/journal?after=N` | GET | P3：journal 增量（hooks/useJournal 增量优先，未实现时全量兜底） |
| `/api/runs/<id>/records` | GET | P4：records 只读清单（Profile 页据此枚举） |
| `/api/runs/<id>/records/<action>` | GET | P4：单条记录（batches_ms / async_leak / detail） |
| `/api/runs/<id>/workspace`、`/workspace/file?path=` | GET | P5：工作区只读清单/文件（后端负责 path 白名单） |
| `/api/runs` | POST | 启动 run（全字段；409=GPU 占用、400=配置非法均展示原文） |
| `/api/models` | POST | 列模型；key 只进服务端内存 |

## 轮询策略（spec §5 的实现）

- Run 详情 / Dashboard 活跃卡：默认 1.5s（设置页可改 3s/5s）；
- Runs 列表 / Dashboard 概览：15s（有活跃 run 时 5s）；
- 终态（completed / no_improvement / budget_exhausted / infra_error /
  journal_corrupt / unknown，见 `lib/states.ts`）立即停止轮询；
- `visibilitychange` 隐藏暂停、回前台立即刷一次；
- 连续 3 次失败退避到 10s 并显示"连接失败，退避重试中"（不静默假装在刷新）。

## 安全边界（与仓库约束一致）

- **API key 只存在于 Launch 页的内存 useState**，随单次请求 POST；提交成功
  立即清空；预览卡只显示打码占位；绝不写入 localStorage / sessionStorage /
  URL / 日志（渲染冒烟含对应断言）。mock 模式下同样不读取 key。
- `candidate_trust=cooperative` 与 `adversarially_secure` 原样可见：顶栏常驻
  徽章 + Champion 卡 + 设置页说明；completed 不渲染为"已验证可信"。
- demo 模式在 Launch 单选与 Runs/Dashboard 表格中显著标注"演示，非 LIVE_MODEL"。
- 缺失指标显示 NOT_RUN/"—"，不画 0、不编造时间（单阶段耗时在 journal 补 ts
  前一律 "—"；`/api/runs` 无 started_at 时回退 run_id 编码时间并注明来源）。
- 没有"停止运行"按钮：后端无取消机制，不提供 no-op 操作。
- 正式 timing/profiling、晋升判定、状态推导都在服务端/evaluator；前端只展示
  run 目录的持久化事实。

## 设计 token 与组件

- `src/index.css` 的 `@theme` 是颜色唯一出处（spec §2.2：分层暗色表面、
  Indigo accent、状态色 run/ok/warn/bad/muted-s + 14%/25% 徽章公式；
  亮色为预留钩子）。
- 枚举映射唯一出处：`src/components/statusMap.ts`（run 状态、候选状态、六阶段）。
- 流程可视化组件：`StageStepper`（六阶段五态）、`StageSwimlanes`（泳道，
  状态推导纯函数在 `src/lib/lanes.ts`：report 终态 > journal 事件 > in_flight/progress）、
  `BudgetBar`（settled/reserved 双层）、`LogViewer`（智能跟随 + auto-scroll）、
  `NotRunBadge`、`CandidateTable`、`ChampionCard`、`CodeView`。
- 本地偏好唯一入口：`src/lib/prefs.ts`（无任何 key 字段）。

## 与设计规范的已知偏差

1. **未引入 TanStack Query / zustand**：`lib/useApi.ts`（已含终态停止、
   隐藏暂停、失败退避）+ `lib/prefs.ts` 覆盖同等需求，接口形态接近 useQuery；
2. **HashRouter**：server.py 静态回退（P1 的 SPA 部分）与当前 singlefile 产物
   配合，hash 路由零后端改动；切 BrowserRouter 只需改 `src/App.tsx`；
3. **目录命名**：规范建议 `src/app/layout`、`api/queries.ts`；现用
   `components/layout|ui`、`hooks/`、`lib/`，职责等价；
4. **代码高亮/图表**：未引入 highlight.js / Recharts —— Profile 图为手写
   SVG，源码查看为带行号的等宽视图；数据量大后可整体替换；
5. **渲染冒烟依赖**：`jsdom` + `esbuild` 为 devDependencies（仅
   `scripts/render-smoke.mjs` 使用），不属于运行时产物。
