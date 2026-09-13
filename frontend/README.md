# kernelagent 前端（React + Vite + TypeScript + Tailwind v4）

kernelagent 本地控制台前端（B1 脚手架阶段）。后端是标准库 HTTP server：
`src/kernelagent/webapp/server.py`（默认 `127.0.0.1:8501`）。

技术栈：Vite 8 · React 19 · TypeScript（strict）· Tailwind CSS v4 · react-router v7。

**遵循的设计规范**：`research/frontend/frontend-design-spec.md`（A1 调研产出）。
本阶段完成其 §8.4 实现顺序的第 1 步（脚手架 + 令牌 + AppShell + statusMap/Badge）
并部分完成第 2 步（api/types + 全部现有端点接通）；与规范的已知偏差见文末。

## 目录结构（关键部分）

```
frontend/
├── index.html                  # 入口 HTML（暗色默认，首帧前应用主题）
├── vite.config.ts              # /api 代理 + 单文件产物（vite-plugin-singlefile）
└── src/
    ├── api/                    # 类型化 API client（唯一前后端通信入口）
    │   ├── types.ts            # 全部请求/响应 TS 类型（与 server.py 契约对齐）
    │   ├── client.ts           # fetch 封装 + ApiError + base URL 管理
    │   └── index.ts
    ├── components/
    │   ├── layout/             # AppLayout / Sidebar / Topbar / PageHeader
    │   ├── ui/                 # Card/Badge/Button/Table/Spinner/EmptyState/ErrorPanel/NotRunBadge
    │   ├── statusMap.ts        # 全部枚举→{标签,颜色} 唯一出处（spec §4.2）
    │   ├── RunStateBadge.tsx   # run 状态徽章（statusMap 驱动）
    │   └── icons.tsx           # 内联线框图标（零依赖）
    ├── lib/
    │   ├── useApi.ts           # 轻量数据 hook（加载/错误/刷新/轮询/终态停止）
    │   ├── states.ts           # TERMINAL_RUN_STATES 终态判定（spec §5）
    │   └── format.ts           # 时长/数量/哈希格式化
    ├── pages/                  # Dashboard/Runs/RunDetail/RunProfile/Launch/Benchmarks/Settings/NotFound
    ├── App.tsx                 # hash 路由表（spec §8.3）
    └── index.css               # Tailwind v4 @theme 设计 token（spec §2.2，暗色默认）
```

## 路由（spec §8.3）

| 路由 | 页面 | 状态 |
| --- | --- | --- |
| `/` | Dashboard 总览 | 已接 `/api/health` `/api/runs` `/api/problems` |
| `/runs` | Runs 列表 | 已接 `/api/runs`（15s 轮询） |
| `/runs/:runId` | Run 详情 | 已接快照；1.5s 轮询、终态停止；时间线/产物 tab 为占位 |
| `/runs/:runId/profile` | Profile 报告 | 占位（依赖规范提案 P4/P5） |
| `/launch` | 新建运行 + API/模型配置 | 已接 `/api/models`（拉模型列表）；启动表单为下一工作包 |
| `/benchmarks` | Benchmark 管理 | 已接 `/api/problems` 题目树 + 过滤 |
| `/settings` | 设置 | 主题切换（暗色默认）；本地偏好为占位 |

## 开发

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

开发期 Vite dev server 把 `/api/*` 代理到后端（见 `vite.config.ts`）：

- 目标：`http://127.0.0.1:8501`（server.py 的 `serve()` 默认端口）
- 换端口：`KA_BACKEND_PORT=xxxx npm run dev`（代理目标随之变化）
- 若 npm 网络慢：`HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 npm install`

API base URL 集中在 `src/api/client.ts`：默认同源（空字符串），可用环境变量
`VITE_API_BASE_URL` 覆盖（构建期注入）。**不要把任何密钥写进 env 文件或代码。**

## 构建

```bash
npm run build        # tsc -b && vite build -> dist/index.html（单文件）
npm run preview      # 本地预览构建产物
```

`vite.config.ts` 启用了 `vite-plugin-singlefile`：JS/CSS 全部内联，
产物是自包含的 `dist/index.html`（约 380KB）。原因见下节。

## 产物如何被 server.py 托管（当前边界，未改 server.py）

`server.py` 的路由只有三类：

1. `GET /`、`GET /index.html` —— 返回 `src/kernelagent/webapp/static/index.html`
   这**一个文件**；
2. `/api/*` —— JSON API；
3. 其余路径 —— JSON 404。**没有任何静态资源文件服务**。

因此直接把 `dist/` 目录拷过去无法工作（`/assets/*.js` 会 404）。两种方案：

- **方案 A（当前默认，无需改后端）**：单文件构建已把 JS/CSS 内联，
  拷一个文件即可：

  ```bash
  cp frontend/dist/index.html src/kernelagent/webapp/static/index.html
  ```

  前端使用 hash 路由（`#/runs/xxx`），所有深链接都落在 `/`，无需服务端
  SPA fallback。
- **方案 B（规范提案 P1，约 20 行后端改动）**：为 server.py 增加
  `/assets/*` 静态服务 + 非 API 路径回退 index.html，然后去掉
  singlefile 插件、把 `createHashRouter` 换成 `createBrowserRouter`
  （只改 `src/App.tsx` 一处）。落地 P1 后建议切换。

注意：当前 `server.py` 只认 `static/index.html` 一个静态路径；本阶段
不修改 `server.py` 与其 `static/` 目录。

## API 覆盖（src/api/client.ts，全部 6 个现有端点）

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/api/health` | GET | 存活、GPU 设备、活跃 run（顶栏 10s 轮询） |
| `/api/problems` | GET | bench/level/problem 题目树（Benchmarks 页） |
| `/api/runs` | GET | 运行历史摘要（新在前；列表/总览页） |
| `/api/runs/<id>` | GET | 单 run 快照：状态/预算/候选/错误尾/in-flight/job |
| `/api/runs` | POST | 启动 run（client 已实现含 backend/disable_thinking；UI 表单为下一工作包） |
| `/api/models` | POST | 按 base_url(+可选 key) 列模型，返回 `{id, owned_by}[]`；key 只进服务端内存 |

非 2xx 统一抛 `ApiError`（`status` + 服务端 `{"error": ...}` 消息）。
400 = 配置非法；**409 = GPU 被活跃 run 占用**；502 = provider 不可达。

## 设计 token（spec §2.2）

`src/index.css` 的 `@theme` 是颜色唯一出处（暗色默认；亮色经
`:root[data-theme="light"]` 覆盖同名变量，是预留钩子）：

- 规范名：表面 `bg/surface/surface-2/surface-3`、边框 `border/border-strong`、
  文字 `text/text-2/text-3`、主色 `accent/accent-hover/accent-subtle`（Indigo）、
  状态 `run(sky)/ok(emerald)/warn(amber)/bad(red)/muted-s(slate)`、
  特例 `champion/log-bg`。
- 脚手架别名（同值）：`fg/muted/surface-raised/primary*/success/warning/error/running`
  与各 `*-soft`（= 状态色 14% alpha，徽章公式：底 14% / 字亮色 / 边 25%）。
- 枚举映射唯一出处：`src/components/statusMap.ts`（run 状态、候选状态、六阶段）。
- 缺失指标用 `NotRunBadge`/"—"，绝不显示 0（项目不变量）。

## 轮询策略（spec §5，当前实现）

- Run 详情：1.5s，终态（completed/no_improvement/budget_exhausted/
  infra_error/journal_corrupt/unknown，见 `lib/states.ts`）立即停止；
- 顶栏健康：10s；列表/总览：15s（"有活跃 run 时 5s"与 visibilitychange
  暂停为后续项，随 TanStack Query 一并接入）。

## 安全边界（与仓库约束一致）

- API key 只存在于"新建运行"页的内存表单 state，随单次请求 POST；
  不写入 localStorage、URL、日志或构建产物；后续启动表单实现时须保持
  "提交成功后立即清空 key 输入框"。
- 正式 timing/profiling、晋升判定、状态推导都在服务端/evaluator；
  前端只展示 run 目录的持久化事实，缺失指标显示 NOT_RUN，不按零处理。
- `candidate_trust=cooperative` 在顶栏常驻警示（champion 需人工审查），
  completed 详情页含同样的诚实性说明卡。

## 与设计规范的已知偏差（后续工作包收敛）

1. **TanStack Query v5 + zustand 未引入**：现用自研 `lib/useApi.ts`
   （已实现终态停止轮询），接口形态接近 useQuery，替换成本低；
2. **HashRouter**：等规范提案 P1（server.py 静态回退）落地后切
   BrowserRouter（改 `App.tsx` 一处 + 去 singlefile）；
3. **目录命名**：规范建议 `src/app/layout`、`api/queries.ts`、扁平
   components；现用 `components/layout|ui`、`lib/useApi.ts`，职责等价；
4. **占位件**：Launch 启动表单、StageStepper/泳道时间线、日志查看器、
   Evidence tab、设置页本地偏好，均按规范标注"下一工作包"。
