# KernelAgent Web 控制台前端重写设计规范

状态：DRAFT（A1 调研产出，供实现 agent 使用）
日期：2026-09-14 凌晨
范围：仅前端（React + Vite + Tailwind）设计规范；后端改动以"提案"形式列出，需另行实现。
硬约束：本文档不包含任何 API key 或 .env 内容；所有后端提案不得改变"key 只驻内存、不落盘、不进日志"的安全语义。

---

## 0. 一句话结论

用 React 18 + Vite + TypeScript + Tailwind 重写为多页 SPA（暗色、数据密集、等宽数字），
信息架构为 7 个页面（Dashboard / Runs / Run 详情 / 新建运行 / Benchmark / Profile / 设置），
现有 6 个 API 端点是必须兼容的契约；新前端落地**必须**让后端补两类能力：
（1）SPA 静态资源服务与路由回退；（2）run 目录细粒度只读端点（journal / records / report / workspace 文件）。
实时更新以**轮询**为默认方案（与现状一致），SSE 作为二期可选升级。

---

## 1. UCAgent 前端调研总结

### 1.1 项目定位

- 仓库：https://github.com/XS-MLVP/UCAgent （"UnityChip Verification AI-Agent"，中科院计算所"万众一芯"/XS-MLVP 开源平台，香山生态的 AI 芯片单元验证 agent）
- 文档站：https://ucagent.open-verify.cc （MkDocs Material；TUI 章节 `/content/02_usage/04_tui/`，Web Master 指南在"功能介绍"下）
- 形态：同一个 Python 后端提供三种交互面 —— CLI 命令、**TUI**（Textual 框架）、**Web UI**（"Web Master" 模式，`ucagent --as-master-persist --as-master`，默认 http://localhost:8800，支持 `地址:端口:密码` HTTP Basic Auth）

### 1.2 技术栈（重要教训）

- Web 前端是 **Jinja 服务端模板 + 内联 JS 的单文件大页面**：`ucagent/server/templates/` 下 6 个页面
  `launch.html`(168KB) / `agent.html`(141KB) / `master.html`(75KB) / `task.html`(32KB) / `complete.html` / `terminal.html`。
- 静态资源只有三个第三方库：`highlight.js`（代码高亮，atom-one-dark 主题）、`marked.min.js`（Markdown 渲染）、`xterm`（Web 终端）、`surfer`（开源波形查看器，用于看验证波形）。
- **教训**：单文件页面在功能膨胀后不可维护（140KB HTML）—— 这正是 kernelagent 现有 static/index.html（364 行内联 JS/CSS）即将面对的路径，重写为组件化 SPA 的决策正确。

### 1.3 TUI（Textual）交互设计——最有价值的部分

界面分区：左侧 **Mission 面板** + 右上 **Status 面板** + **Messages 消息流** + 底部 **Console**（Output 分页 + Input 命令行），每秒自动刷新且不打断输入。

- **阶段状态色语义**（全站统一）：白=待执行，红=正在执行，绿=通过，黄=跳过；另有蓝/绿/红表示 LLM 检查开关状态。每个阶段条目带：索引、标题、**失败计数、累计耗时**。
- **Changed Files 列表**：文件名 + 修改时间 + 相对时间（"3m ago"），越新越绿 —— 把"最近发生了什么"变成一眼可见。
- **Tools Call 状态**：工具名(调用次数)，忙碌中黄色高亮 —— "agent 正在干什么"的可视化。
- **智能跟随滚动**：消息流默认跟随最新；用户手动上滚后停住保持位置；滚回底部自动恢复跟随；Esc 一键退出手动模式。日志类 UI 的最佳交互范式。
- **状态三元组**：`ProviderTokens(input/output/total) · Context(当前值/阈值) · Compression(压缩原因与前后规模)` —— 预算/上下文占用常驻可见。
- 布局可调（Ctrl+方向键 / Ctrl+H/J/K/L）、主题选择器（Ctrl+T）、F1 快捷键帮助、Tab 命令补全、后台命令 `&` + 完成提示。
- 输入框忙碌时轮转显示 `(wait.) (wait..) (wait...)`。

### 1.4 Web UI（Web Master）页面结构

| 页面 | 职责 | 关键交互 |
|---|---|---|
| **Launch** | 工作区创建、文件导入/上传、模块解析、编译、**启动命令预览 + 启动** | 表单向导式：准备→解析→编译→预览命令→启动 |
| **Task** | 托管任务列表：按 Status/DUT/Module/关键字筛选、分页（10/20/50/100）、状态聚焦（failed/stopped） | 右侧**粘性详情卡**：stdout+stderr 合并日志、ANSI→HTML 渲染、Auto scroll 开关（localStorage 持久化）；URL hash 深链 `#task-<id>` 自动翻页定位；停止/删除 |
| **Agent** | 阶段级控制与复盘：每阶段 HM/Skip/LFail/LPass 开关、多选批量设置；**阶段产物文件清单 + 内容预览 + Diff 视图** | 推荐流程：批量设策略 → 检查推进 → 失败阶段进 Diff 复盘 |
| **Dashboard(master)** | Agent 汇总：过滤、排序（last_seen/progress）、分页、批量删除、离线自动清理 | |
| **Terminal** | xterm Web 终端，多会话（不同 URL），与本地终端并行 | |

task.html 具体实现细节（值得直接抄的工程决策）：

- 轮询 `setInterval(loadTasks, 5000)`，5 秒间隔，详情卡随列表一起刷新。
- 状态徽章配色（暗色背景上低饱和 + 半透明底）：
  ```css
  .ok  { background:rgba(34,197,94,.14);  color:#86efac; border-color:rgba(34,197,94,.25) }   /* running */
  .err { background:rgba(239,68,68,.14);  color:#fca5a5; border-color:rgba(239,68,68,.25) }   /* failed  */
  .warn{ background:rgba(245,158,11,.14); color:#fcd34d; border-color:rgba(245,158,11,.25) }  /* 其他    */
  ```
- 设计令牌：`--bg:#0f1117 --surface:#1a1d26 --surface2:#22263a --border:#2e3248 --text:#e2e8f0 --text-dim:#8892a4 --accent:#6366f1 --accent-hover:#818cf8`，日志底 `#0b0d12`，选中行 `outline:2px solid rgba(99,102,241,.75)`。
- 日志面板：等宽字体、ANSI 转义码映射到固定色值（31→#f87171、32→#4ade80、34→#60a5fa…）、Auto scroll 偏好持久化。
- **破坏性操作门控**：顶部 "Show delete" 复选框，未开启时所有删除按钮 alert 拒绝；删除前确认框。
- 筛选状态持久化到 localStorage；手动 Refresh 按钮兜底。

### 1.5 可借鉴点清单（按优先级）

1. **阶段状态色语义统一 + 每阶段失败计数/耗时**：pending / running / pass / fail / skipped 五态贯穿 TUI 与 Web，映射到我们的 `generate → policy → evaluate → correctness_pro → timing → confirm`。
2. **Task 页范式**：左列表（筛选/分页/状态徽章/localStorage 持久化）+ 右粘性详情卡（等宽日志、ANSI 渲染、auto-scroll 开关、hash 深链）。直接对应我们的 Runs 列表 + Run 详情。
3. **产物复盘三件套**（agent.html）：产物文件清单 → 内容预览 → **Diff 视图**。对应我们的候选源码查看、候选 vs baseline、champion 对比、workspace 容器 stdout/stderr。
4. **智能跟随滚动 + Auto scroll 开关**：实时日志的标配交互。
5. **低饱和徽章配色公式**：`bg=状态色 14% alpha、text=状态色 300 级、border=状态色 25% alpha`，暗色下既醒目又不刺眼。
6. **破坏性操作门控**（Show delete 复选框 + 确认）：对应我们的"停止运行/清理"类操作（若后端支持）。
7. **预算常驻可见**（Status 三元组）：对应 GPU 秒 / Token 双预算条 + reserved vs settled 双态。
8. **专业查看器嵌入**（surfer 波形 / xterm 终端 / highlight.js）：提示我们 Profile 页可嵌轻量查看器而非自绘。
9. **TUI 与 Web 互补定位**：TUI 管本地细粒度调试，Web 管观察与复盘 —— 我们的控制台定位"观察 + 复盘 + 启动"，不要试图在 Web 里复刻全部 CLI 能力。
10. **反面教材**：单文件 140KB HTML 页面 —— 组件化的理由。

---

## 2. 美观调研结论与视觉语言

### 2.1 原则（来源见 §9）

- 暗色主题**不用纯黑**（#000）：用深灰/深蓝灰分层表面表达层级（elevation by surface color）。
- 正文文字**不用纯白**：用 off-white（#e5e7eb 级），目标 WCAG ≥ 4.5:1。
- **彩色只留给语义**：状态色（成功/失败/警告/进行中）与主色 accent 之外一律灰阶；开发者仪表盘的日志、指标、状态必须"跳出来"，所以背景必须克制。
- 状态色用**去饱和/半透明**变体做底、亮色做字（UCAgent 徽章公式）。
- 数据密集区（表格、日志、时间线）字号 13px、行高紧凑；数字一律等宽字体 + `tabular-nums`。

### 2.2 设计令牌（具体色值，Tailwind v4 @theme 变量）

底层沿用 Tailwind 调色板保证一致性；状态色对齐 UCAgent（Tailwind 系色值），accent 选 Indigo。

```css
/* 表面与边框（分层暗色，不用纯黑） */
--color-bg:        #0b0f17;   /* 应用底 */
--color-surface:   #111827;   /* 面板 = gray-900 */
--color-surface-2: #1a2230;   /* 次级面板 / 悬停行 */
--color-surface-3: #222b3d;   /* 输入框 / chip 底 */
--color-border:    #253046;   /* 常规边框 */
--color-border-strong: #33405c;

/* 文字 */
--color-text:      #e6eaf2;   /* 主文字（off-white） */
--color-text-2:    #9aa5b8;   /* 次级 */
--color-text-3:    #64748b;   /* 弱化/占位 = slate-500 */

/* 主色（Accent）：Indigo */
--color-accent:        #6366f1;  /* indigo-500：主按钮、选中态、链接 */
--color-accent-hover:  #818cf8;  /* indigo-400 */
--color-accent-subtle: rgba(99,102,241,.14);

/* 状态色（语义，全站唯一出处） */
--color-run:      #38bdf8;  /* sky-400   running/进行中 */
--color-ok:       #34d399;  /* emerald-400  completed/promoted/pass */
--color-warn:     #fbbf24;  /* amber-400    budget_exhausted/接近预算/skipped */
--color-bad:      #f87171;  /* red-400      infra_error/journal_corrupt/failed */
--color-muted-s:  #94a3b8;  /* slate-400    no_improvement/retained/unknown/NOT_RUN */
/* 徽章底色公式：bg = 状态色 @14% alpha，text = 状态色(或再亮一档)，border = 状态色 @25% alpha */

/* 特例 */
--color-champion: #fbbf24;  /* champion 用金色点缀（🏆 边框/图标），底仍走 ok 徽章 */
--color-log-bg:   #0b0d12;  /* 日志/代码块底（比 surface 更深） */
```

Run 状态 → 颜色映射（与后端枚举一一对应，不得新增语义）：

| state | 色 | 中文 |
|---|---|---|
| running | run(#38bdf8) | 运行中 |
| completed | ok(#34d399) | 完成（已晋升） |
| no_improvement | muted(#94a3b8) | 无改进 |
| budget_exhausted | warn(#fbbf24) | 预算耗尽 |
| infra_error | bad(#f87171) | 基础设施错误 |
| journal_corrupt | bad(#f87171) | journal 损坏 |
| unknown | muted | 未知 |

候选状态 → 颜色：promoted→ok、retained→muted、failed→bad、rejected→warn、其他→muted（沿用现 UI 语义）。

### 2.3 字体

- UI：`Inter, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif`（中文回退保留）。
- 代码/日志/run_id/哈希/数字：`"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace`；所有数值单元格 `tabular-nums`。
- 字号阶梯：页面标题 18/600，面板标题 13/600 大写字距（沿用现 UI 的 uppercase+tracking 风格），正文 14，表格与密集数据 13，徽章 11-12，日志 12.5/1.6。

### 2.4 间距 / 形状 / 密度

- 4px 基线网格；页面留白 24px，卡片间距 16px，卡内边距 16-20px。
- 圆角：面板 rounded-xl(12px)，卡片/输入 rounded-lg(8px)，徽章全圆 pill。
- 表格行高 40px；时间线条目高 28-32px；控件高：输入 34px、按钮 36px、小按钮 28px。
- 边框 1px 为主，不用大阴影（暗色下用表面分层代替投影）。

### 2.5 图表

- 加速比柱状/区间图、预算消耗曲线：直接用轻量方案（Recharts 或手写 SVG），轴/网格线用 `--color-border`，数据系列用 accent/ok，CI 区间用 14% alpha 填充。
- 缺失数据画"—"并给 NOT_RUN 徽章，**绝不画 0**（项目不变量：缺失指标不是零）。

---

## 3. 信息架构

### 3.1 页面清单与路由表

| 路由 | 页面 | 职责 | 现有数据源 |
|---|---|---|---|
| `/` | Dashboard 总览 | GPU 状态、活跃 run 卡、最近 runs、汇总指标 | `/api/health` `/api/runs` |
| `/runs` | Runs 列表 | 全部历史 run 表格 + 筛选 | `/api/runs` |
| `/runs/:runId` | Run 详情 | 阶段时间线、预算、候选对比、champion、错误、日志/产物、证据 | `/api/runs/:id` + 新增提案 P2-P5 |
| `/launch` | 新建运行 | 模式/题目/模型/API 配置/预算表单（含模型拉取） | `/api/problems` `/api/models` `POST /api/runs` |
| `/benchmarks` | Benchmark 管理 | KernelBench level/题目浏览、适配器接入状态 | `/api/problems` |
| `/runs/:runId/profile` | Profile 报告 | timing 样本分布、批次耗时、容器日志（依赖提案 P4/P5） | 提案 |
| `/settings` | 设置 | Provider 预设、默认预算、主题、轮询间隔（仅本地 localStorage，不含 key） | 纯前端 |

侧边导航固定 220px（可折叠为图标栏），顶部条显示：GPU 徽章、活跃 run 徽章、`candidate_trust = cooperative（champion 需人工审查）` 常驻警示徽章（保留现 UI 的诚实性声明）。

### 3.2 Dashboard（`/`）

组件：`GpuStatusCard`、`ActiveRunCard`（状态徽章 + 阶段时间线迷你版 + 双预算条）、`RecentRunsTable`（最近 8 条）、`StatTile`×4（总 runs / 完成率 / 最佳加速比 / 有 champion 的 run 数——仅统计，缺失显示 "—"）。

```
+--------------------------------------------------------------------------+
| KernelAgent   [Dashboard][Runs][New Run][Benchmarks][Settings]   GPU:...  |
+------------------+-------------------------------------------------------+
|  (220px 侧栏)    |  +----------------+  +----------------+               |
|                  |  | GPU 状态        |  | 活跃运行        |               |
|  总览            |  | cuda:0 · idle  |  | 102458 running |               |
|  Runs            |  +----------------+  | ▮▮▮▮▯▯ 阶段条   |               |
|  新建运行         |  +--[统计瓦片]x4----+  | GPU ██░ 37%    |               |
|  Benchmarks      |                     | Token ███░ 61% |               |
|  设置            |                     +----------------+               |
|                  |  +-----------------------------------------------+     |
|  [trust 徽章]    |  | 最近运行                                        |     |
|                  |  | run_id   模式   题目        状态      champion  |     |
|                  |  | 102604  live   l1:40    ●运行中      —          |     |
|                  |  | 102458  live   l1:40    ✓完成(晋升)  sha256:…   |     |
|                  |  +-----------------------------------------------+     |
+------------------+-------------------------------------------------------+
```

### 3.3 Runs 列表（`/runs`）

组件：`RunsFilterBar`（状态下拉、模式下拉、关键字、Refresh）、`RunsTable`（run_id 等宽、模式、题目、状态徽章、champion 有无、开始时间相对值）、行点击进详情；筛选持久化 localStorage（借鉴 UCAgent）。分页客户端实现（现接口无分页参数；数据量小）。

```
+----------------------------------------------------------------------+
| Runs            [状态: 全部▾] [模式: 全部▾] [搜索________] [刷新]      |
+----------------------------------------------------------------------+
| run_id*        | 模式  | 题目     | 状态          | champion | 开始    |
| 20260913-102604| live  | l1:40   | ● running     | —        | 3m ago |
| 20260913-102458| live  | l1:40   | ✓ completed   | 🏆 有     | 12m ago|
| 20260913-101741| live  | l1:40   | ✕ infra_error | —        | 1h ago |
+----------------------------------------------------------------------+
  * 等宽字体，行 hover 高亮，点击进入 /runs/:runId
```

### 3.4 Run 详情（`/runs/:runId`）—— 核心页面

组件：`RunHeader`（run_id、状态徽章、模式、题目、复制 sha）、`BudgetPanel`（GPU 秒/Token 双条 + reserved/settled 细分）、`StageTimeline`（§4）、`CandidateTable`（对比视图）、`ChampionCard`、`ErrorPanel`、Tabs：`概览 | 候选对比 | 日志与产物 | 证据`。

```
+--------------------------------------------------------------------------+
| ← 返回   20260913-102458  [✓完成(晋升)]  live · kernelbench:l1:40 · triton |
+--------------------------------------------------------------------------+
| GPU 1200/1800s (67%) ██████░░░     Token 82k/200k (41%) █████░░░░         |
+--------------------------------------------------------------------------+
| 候选流水（每个候选一行泳道，详见 §4.2）                                      |
| baseline-eager  █▦▦▦▦▦  timing  measured   —                              |
| candidate-000   ▦▁▁▁▁▁  generation ✕ failed  "provider HTTP 401"          |
| candidate-001   ▦▦▦▁▁▁  timing    ● running                               |
+--------------------------------------------------------------------------+
| [概览][候选对比][日志与产物][证据]                                           |
|  候选对比:                                                                 |
|  | 候选         | 状态    | 阶段    | 加速比 CI        | 详情(截断)  |      |
|  | baseline     | measured| timing  | —(基准)          | batches 12 |      |
|  | candidate-001| promoted| confirm | 1.18× – 1.34×   | …          |      |
|  Champion: 🏆 sha256:9f2c… · path:…  [查看源码][与 baseline Diff]          |
+--------------------------------------------------------------------------+
```

### 3.5 新建运行（`/launch`）

保留现表单全部字段并补齐后端已支持但旧 UI 未暴露的参数（`backend`、`disable_thinking`）。分组布局：`运行模式 → 题目选择（bench/level/problem 级联）→ 模型与凭据 → 预算 → 高级`。"用此 Key 拉取模型列表"保留（POST /api/models）；提交成功后清空 key 输入框（沿用现行为）；空 key 时提示将回退到服务端环境变量（API_KEY_ENV），失败时展示 400/409 错误原文。

```
+--------------------------------------+------------------------------------+
| 运行模式                              | 提示栏                              |
| ( ) live   真实模型(LIVE,需Key)       | · key 仅驻内存,不落盘不进日志        |
| ( ) demo-correct 演示·正确候选        | · 二次启动会 409:单 GPU 单运行       |
| ( ) demo-wrong   演示·错误候选        | · demo 模式不得记为 LIVE_MODEL      |
| Benchmark/Level/Problem 级联选择      |                                    |
| 模型 [glm-4.5    ] [拉取模型列表]      |   (粘性预览卡)                      |
| Base URL [https://api.deepseek.com]   |   预览将要 POST 的 JSON(隐去 key)   |
| API Key  [********] (必填提示/回退提示) |   mode/problem/model/base_url/     |
| 候选数[5] 修复轮[2] GPU秒[1800]        |   max_candidates/…/token_budget    |
| Token[200000] 后端[triton▾] 思考[关☐]  |   [开始优化]                        |
+--------------------------------------+------------------------------------+
```

### 3.6 Benchmark 管理（`/benchmarks`）

组件：`BenchStatusCards`（KernelBench=已接入 / TritonBench、FlashInfer-Bench=适配器未接入，数据来自 `/api/problems` 的 `note` 字段，诚实展示不可用状态）、`LevelAccordion`（Level 1-4 折叠列出题目、题数）、题目行显示 spec（`kernelbench:l1:40`）与"去运行"快捷链接（跳 /launch 预填）。

### 3.7 Profile 报告（`/runs/:runId/profile`）

依赖提案 P4/P5。组件：`BatchLatencyChart`（timing record 的 `batches_ms[]` 箱线/散点）、`SampleStats`（样本数、均值、CI）、`ContainerLogViewer`（workspace/container-*/stdout.log+stderr.log，ANSI 渲染 + auto-scroll）、`AsyncLeakBadge`（record 的 `async_leak`）。无数据时整页 NOT_RUN 状态而非空图。

### 3.8 设置（`/settings`）

仅本地偏好：默认预算值、Provider 预设（名称+base_url，**不含 key**）、轮询间隔（1.5s/3s/5s）、日志 auto-scroll 默认值、主题（先只做暗色，留 light 钩子）。全部 localStorage，不与后端交互。

---

## 4. Agent 流程状态可视化设计

### 4.1 阶段模型

固定六阶段（与后端 progress stage 枚举一致，不得增删语义）：

```
generate(生成) → policy(策略门) → evaluate(容器评测) → correctness_pro(增强正确性) → timing(正式计时) → confirm(晋升确认)
```

另注意：record 里失败发生在阶段之前时会标 `stage: "generation"`（如 provider 失败），需归并显示到 generate 节点并标红。

### 4.1.1 单候选阶段时间线（横向 stepper，Run 详情概览与 ActiveRunCard 复用）

```
  生成        策略门      容器评测     增强正确性    正式计时     晋升确认
 ┌──────┐   ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
 │  ✓   │───│  ✓   │───│  ●   │────│  ○   │────│  ○   │────│  ○   │
 │ 1.2s │   │ skip │   │ 34s  │    │ —    │    │      │    │      │
 └──────┘   └──────┘   └──────┘    └──────┘    └──────┘    └──────┘
```

节点五态（对齐 UCAgent 语义）：
- `done`：填充 ok 色圆点 + 对勾；节点下显示耗时（有数据时）
- `running`：run 色 + 呼吸动画（`animate-pulse`）+ 已耗时实时累加
- `pending`：空心圆 + border 色
- `failed`：bad 色圆点 + ✕；节点红色描边，点击展开错误
- `skipped`：warn 色 + "skip"（如 policy 门直接拒绝）

数据来源：进行中用快照的 `in_flight[]` + `progress{action_id: stage}`（每 action 只给当前 stage）；已完成的候选在终态后从 report 的 `candidates[]` 读取 `stage/status`。**中间态时间戳目前仅 progress.json 有 `ts`，单阶段精确耗时为二期能力（需 P3 journal 端点 + 后端补 ts），一期显示 "—" 不得编造数字。**

### 4.1.2 多候选泳道图（Run 详情概览）

每候选一行，行首为候选名（等宽）+ 状态徽章，行主体为六段水平条：已完成段填 ok 色、进行中段 run 色 + 流光动画、失败段 bad 色、未到达段为暗底。行右侧放关键指标（加速比 CI 或失败原因 chip）。当前活跃行置顶并高亮左边框。

```
baseline-eager   [measured ]  ██████████████████████  —
candidate-000    [✕ failed ]  ██▍                     generation: provider HTTP 401
candidate-001    [● running]  ██████████████░░░░░░░░  timing…
candidate-002    [○ pending]  ░░░░░░░░░░░░░░░░░░░░░░
                              gen  pol  eval  pro  timing  confirm
```

### 4.2 状态徽章系统

统一 `Badge` 组件：`pill + 1px 边框 + 11px 字号`，底/字/边按 §2.2 公式。带圆点前缀（`● running` 时圆点呼吸）。所有枚举集中到一个 `statusMap.ts`（state/candidate status/stage → 中文标签 + 色组），禁止组件内散落映射。

### 4.3 错误呈现（三层）

1. **Run 级横幅**（`infra_error`/`journal_corrupt`）：页面顶部红色横幅 + `error_tail`（后端已截断 1500 字符）等宽展示，尾部 300 字直接可见，"展开全部/复制"按钮；横幅必须注明"基础设施错误≠候选失败"（项目不变量）。
2. **候选级失败**：候选行内 warn/bad chip + `detail` 截断 400 字符，点击展开完整 detail（现 UI 已截断 400，新 UI 做成可展开）。
3. **终态解释**：`no_improvement`、`budget_exhausted` 用中性/琥珀说明卡说明这是**合法结果**（预算耗尽不是错误）；`completed` 卡显示 champion + CI + trust 状态 + "需人工审查"提醒（cooperative 语义不可弱化）。

---

## 5. 实时状态更新方案

**结论：默认轮询，SSE 二期可选。**

依据（§9 参考）：内部仪表盘秒级延迟可接受时轮询最简单可靠——每请求独立、无重连问题、无代理缓冲坑；SSE 适合亚秒级推送，浏览器 EventSource 自带断线重连 + Last-Event-ID，但需后端 `text/event-stream` 支持且偶发代理缓冲问题。现有 server 是 stdlib ThreadingHTTPServer，手写 SSE 可行但要处理客户端断连检测；收益（1.5s→0s）对本地单用户小。

| 场景 | 间隔 | 说明 |
|---|---|---|
| Run 详情（state=running） | 1.5s | 沿用现值；进入终态立即停止（后端 TERMINAL_STATES：completed/no_improvement/budget_exhausted/infra_error，UI 另加 journal_corrupt/unknown） |
| Runs 列表 / Dashboard | 5s | 有活跃 run 时 5s，否则 15s |
| 存在活跃 run 的 Dashboard ActiveRunCard | 1.5s | |
| 手动刷新按钮 | - | 永远提供（UCAgent 兜底模式） |

工程规则：
- 用 TanStack Query 的 `refetchInterval` 实现，自动获得错误重试与缓存去重；连续 3 次失败退避到 10s 并显示"连接失败，重试中"横幅（不静默假装在刷新）。
- `document.visibilityState === 'hidden'` 时暂停轮询，回前台立即刷一次。
- 终态停止轮询的同时刷新一次 runs 列表（沿用现 UI 行为）。
- 二期 SSE 提案见 P6；前端抽象成 `useRunSnapshot(runId)` 单一 hook，轮询/SSE 切换不影响组件层。

---

## 6. 现有 API 端点清单（必须兼容的契约）

来源：`src/kernelagent/webapp/server.py`（385 行）与 `src/kernelagent/webapp/jobs.py`。服务器为 stdlib `ThreadingHTTPServer`，默认 `127.0.0.1:8501`，无 CORS 头（同源使用；Vite 开发期需 proxy）。语义基线：**UI 展示的一切持久事实来自 run 目录（journal.jsonl / records/ / report.json / job.json），server 不发明状态**。

### 6.1 GET /

返回 `static/index.html` 单文件；**除 `/` 与 `/index.html` 外没有任何静态资源路由**（重写 SPA 的最大兼容缺口，见 P1）。

### 6.2 GET /api/health

```json
{ "ok": true, "gpu_device": "nvidia.com/gpu=GPU-ea24…", "active_run_id": "20260913-102604" | null }
```

### 6.3 GET /api/problems

```json
{
  "bench": "kernelbench",
  "note": "tritonbench / flashinfer-bench adapters exist but are not wired into optimize",
  "levels": [
    { "level": 1,
      "problems": [ { "id": 40, "spec": "kernelbench:l1:40", "label": "40_LayerNorm" } ] }
  ]
}
```

### 6.4 GET /api/runs

```json
{ "runs": [
  { "run_id": "20260913-102458", "state": "completed", "mode": "live",
    "problem": "kernelbench:l1:40", "champion": "<sha256>|null" } ] }
```
按目录名倒序（新在前）。state 枚举见 §6.5。champion 字段为 `report.champion.candidate_sha256`。

### 6.5 GET /api/runs/{run_id}（run_snapshot）

```json
{
  "run_id": "20260913-102458",
  "state": "running | completed | no_improvement | budget_exhausted | infra_error | journal_corrupt | unknown",
  "budget": {
    "gpu_seconds_limit": 1800.0, "tokens_limit": 200000,
    "settled_gpu_seconds": 1200.0, "settled_tokens": 0,
    "reserved_gpu_seconds": 0.0, "reserved_tokens": 0
  },
  "in_flight": ["candidate-001"],
  "progress": { "candidate-001": "timing" },
  "candidates": [
    { "candidate": "candidate-000", "status": "failed|rejected|retained|promoted|…",
      "stage": "generation", "detail": "provider HTTP 401",
      "ratio_ci_95": [1.18, 1.34] }          // 仅 timing 后的候选有
  ],
  "champion": { "candidate_sha256": "…|null", "path": "…|null" },
  "candidate_trust": "cooperative",
  "adversarially_secure": false,
  "error_tail": "…error.txt 末 1500 字符 | null",
  "job": {
    "run_id": "…", "mode": "live|demo-correct|demo-wrong",
    "config": { "problem": "kernelbench:l1:40", "backend": "triton",
      "model_id": "glm-4.5", "base_url": "https://api.deepseek.com",
      "max_candidates": 5, "max_repair_rounds": 2,
      "gpu_budget_seconds": 1800.0, "token_budget": 200000,
      "gpu_device": "nvidia.com/gpu=…" },
    "started_at": 1789266290.0
  }
}
```

- **不含 api_key**（job.json 是"UI 的持久配置记录，永不包含 key"）。
- `candidates` 仅来自 report.json → **运行中通常为空数组**；运行中的实时进度只能来自 `in_flight`+`progress`（一期 UI 的已知局限，P3 可解）。
- `error.txt` 存在且 state ∈ {unknown, running} → state 升级为 `infra_error`。
- journal 解析失败 → `journal_corrupt`（快照直接返回，无其余字段更新）。
- 404：`{ "error": "unknown run '…'" }`。

### 6.6 POST /api/runs（启动）

请求体（后端实际接受的完整字段；**旧 UI 未发送 `backend`、`disable_thinking`，新 UI 应暴露**）：

```json
{ "mode": "live|demo-correct|demo-wrong",
  "problem": "kernelbench:l1:40",
  "model": "glm-4.5",
  "base_url": "https://api.deepseek.com",
  "api_key": "（仅驻内存；可空 → 回退服务端环境变量 API_KEY_ENV）",
  "backend": "triton",
  "disable_thinking": false,
  "max_candidates": 5, "max_repair_rounds": 2,
  "gpu_budget_seconds": 1800.0, "token_budget": 200000 }
```

响应：200 `{ "run_id": "20260913-103000", "state": "running" }`；
400 `{"error": …}`（mode 非法 / problem 无法解析 / 配置错误）；**409** `{"error": "run <id> is still using the GPU"}`（单 GPU 单运行）。

### 6.7 POST /api/models（拉取模型列表）

请求 `{ "base_url": "https://…", "api_key": "sk-… 或空串" }`（空则回退 env；key 不落盘）。
响应 200 `{ "models": [ { "id": "glm-4.5", … } ] }`；
400 `{ "error": …, "models": [] }`（PermanentModelError，如鉴权失败）；502 `{ "error": "provider unreachable: …", "models": [] }`。

### 6.8 前端必须遵守的安全/语义约束

1. api_key 只存在于内存表单 state，**禁止**写入 localStorage/URL/日志/错误上报；提交成功后立即清空输入框。
2. `candidate_trust=cooperative` 与 `adversarially_secure=false` 必须原样可见，不得美化成"可信"。
3. demo 模式运行必须显著标注"演示，非 LIVE_MODEL"。
4. 缺失指标显示 NOT_RUN/"—"，不显示 0。
5. 终态判定以后端枚举为准，前端不得自行推断"成功"。

---

## 7. 后端新增 API 提案（路径/方法/响应示例）

优先级：P0/P1 是 SPA 落地前提；P2-P5 支撑 Run 详情/Profile 页；P6 二期可选。

### P1（必须）SPA 静态资源与路由回退

- `GET /assets/<path>`：以 `webapp/static/` 为 root 服务 Vite 构建产物（js/css/字体），带 Content-Type 与简单缓存头（hashed 文件名可 `max-age=31536000`）。
- `GET /{route}` 与 `GET /runs/{id}` 等非 `/api/*` 路径：回退返回 index.html（支持 history 路由）。若不想改 server，前端改用 HashRouter 可绕过（代价：URL 不美、无法服务端区分路由），**推荐直接加回退，约 20 行**。

### P2（高）原始报告

- `GET /api/runs/{id}/report` → report.json 原文（含 `commit`、`protocol`、`problem_sha256`、`problem_path`、`gpu_device`、`journal_entries`、`config`）。用于 Run 详情"证据"tab 的身份链展示（代码/环境/协议身份是项目不变量）。

### P3（高）journal 增量读取（运行中候选进度 + 预算台账）

- `GET /api/runs/{id}/journal?after=<line_no>` → 服务端按行号增量返回（journal.jsonl 每行一条）：

```json
{ "run_id": "…", "next_after": 17,
  "entries": [
    { "seq": 3, "kind": "budget_reserved", "action_id": "candidate-001",
      "gpu_seconds": 300.0, "tokens": 0,
      "entry_hash": "671c…", "prev_hash": "5c14…" },
    { "seq": 4, "kind": "action_started", "action_id": "candidate-001",
      "input_hash": "candidate:8d97…", "budget_reservation": "671c…" }
  ] }
```

- 支撑：泳道图精确事件流、reserved/settled 台账可视化、hash 链展示。**注意：journal 条目无时间戳字段**，后端需在返回时补充文件行偏移对应的时间或让 optimization 写入 ts（标为后端待办，前端不强造）。

### P4（高）单条 action 记录（候选对比 / Profile 数据源）

- `GET /api/runs/{id}/records/{action_id}` → `records/{action_id}.json` 原文。timing 记录含 `batches_ms[]`（Profile 图表数据）、`async_leak`、`status:"measured"`。
- `GET /api/runs/{id}/records/{action_id}/progress` → progress.json（`stage` + `ts`），补齐泳道时间戳。

### P5（中）工作区产物只读访问（日志与源码 Diff）

- `GET /api/runs/{id}/workspace` → 顶层条目列表：

```json
{ "entries": [
  { "path": "container-416f3e33…", "type": "dir",
    "files": ["container-record.json", "stdout.log", "stderr.log"] },
  { "path": "timing-inputs-eager-timing", "type": "dir",
    "files": ["candidate.py", "timing_driver.py", "case.json"] } ] }
```

- `GET /api/runs/{id}/workspace/file?path=timing-inputs-eager-timing/candidate.py` → `{ "path": "…", "size": 1234, "content": "…" }`（文本上限如 256KB，二进制返回 415）。
- **安全要求**：后端必须做 path 白名单/规整（resolve 后必须仍在 run workspace 内、拒绝 `..` 与符号链接逃逸），只读。这是唯一触碰任意文件的端点，实现 agent 不得省略校验。

### P6（二期可选）SSE 快照流

- `GET /api/runs/{id}/events` → `text/event-stream`，每 1.5s（或快照变化时）推送 `event: snapshot\ndata: <run_snapshot JSON>\n\n`；心跳注释行防代理超时。前端 `useRunSnapshot` 内部可切换 EventSource；轮询路径保留为回退。

### 明确不提案

- `POST /api/runs/{id}/cancel`：后端当前无安全中止机制（线程 daemon 运行、无取消端口），**在 orchestrator 提供取消能力前不做**，避免 no-op 按钮冒充功能。UI 一期不出现"停止"按钮。
- 任何写操作端点（改配置/清目录）。

---

## 8. React + Vite + Tailwind 项目结构建议

### 8.1 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 构建 | Vite 5 + `@vitejs/plugin-react` | 标准起步 |
| 语言 | TypeScript（strict） | API 契约类型化（§6 的 zod 或手写 type） |
| 样式 | Tailwind（v4 `@theme` 令牌 或 v3 config extension） | 设计令牌进 CSS 变量 |
| 路由 | react-router v6，**BrowserRouter**（配 P1 回退）；若 P1 不做则 HashRouter | |
| 服务端状态 | **TanStack Query v5**（轮询=refetchInterval、缓存、重试都是内置能力） | 比 zustand+手写 useEffect 省一半代码 |
| 客户端状态 | **zustand**（仅：launch 表单草稿、runs 筛选、UI 偏好） | 轻量；避免为小状态上 Redux |
| 图表 | 首选手写 SVG（阶段条/预算条）；统计图用 Recharts | 阶段条是定制组件，图表仅 Profile 用 |
| 代码高亮 | highlight.js（atom-one-dark，与 UCAgent 同款）+ react 适配 | Diff/源码查看 |
| Diff | `diff` + 手工两栏渲染（避免重型 monaco；若 Profile 页需求强再上 @monaco-editor/react lazy） | |
| 测试 | Vitest + Testing Library（状态映射/徽章/时间线纯函数优先测） | |

### 8.2 目录树

```
webapp-ui/
├── index.html
├── vite.config.ts            # server.proxy: '/api' -> http://127.0.0.1:8501
├── tailwind.config.ts        # 或 v4 @theme（§2.2 令牌唯一出处）
├── tsconfig.json
└── src/
    ├── main.tsx              # QueryClientProvider + RouterProvider
    ├── app/
    │   ├── router.tsx        # 路由表（§3.1）
    │   └── layout/
    │       ├── AppShell.tsx  # 侧栏 + 顶栏（GPU 徽章 / trust 徽章 / 活跃 run）
    │       ├── Sidebar.tsx
    │       └── Topbar.tsx
    ├── api/
    │   ├── client.ts         # fetch 封装：JSON、错误归一（409/400/502）、不做任何 key 缓存
    │   ├── types.ts          # §6 契约类型（RunSnapshot/JobConfig/CandidateRecord…）
    │   ├── queries.ts        # useHealth/useProblems/useRuns/useRunSnapshot(refetchInterval)
    │   └── mutations.ts      # useStartRun/useListModels（成功后清空 key 字段）
    ├── hooks/
    │   └── useRunSnapshot.ts # 轮询策略集中点（§5 表格），二期换 SSE 只改这里
    ├── store/
    │   ├── launchDraft.ts    # zustand：launch 表单（key 字段仅内存）
    │   └── prefs.ts          # zustand persist：筛选/轮询间隔/autoscroll（无 key）
    ├── components/
    │   ├── Badge.tsx         # §4.2 徽章（statusMap 驱动）
    │   ├── statusMap.ts      # 全部枚举→{label,color,icon} 唯一出处
    │   ├── BudgetBar.tsx     # reserved/settled 双层条
    │   ├── StageStepper.tsx  # §4.1.1 单候选六阶段
    │   ├── StageSwimlanes.tsx# §4.1.2 多候选泳道
    │   ├── CandidateTable.tsx# 候选对比（加速比 CI 列）
    │   ├── ChampionCard.tsx
    │   ├── ErrorPanel.tsx    # §4.3 三层错误呈现
    │   ├── LogViewer.tsx     # ANSI 渲染 + 智能跟随滚动 + auto-scroll 开关
    │   ├── CodeView.tsx      # highlight.js 包装
    │   ├── DiffView.tsx
    │   ├── StateDot.tsx
    │   └── NotRunBadge.tsx   # NOT_RUN/"—" 统一呈现
    ├── pages/
    │   ├── DashboardPage.tsx
    │   ├── RunsPage.tsx
    │   ├── RunDetailPage.tsx     # tabs: Overview/Candidates/Logs/Evidence
    │   ├── LaunchPage.tsx
    │   ├── BenchmarksPage.tsx
    │   ├── ProfilePage.tsx
    │   └── SettingsPage.tsx
    └── lib/
        ├── format.ts         # 相对时间、sha 截断、duration、tabular-nums
        └── states.ts         # TERMINAL_STATES 判定（§5）
```

### 8.3 路由表

```
/                      DashboardPage
/runs                  RunsPage
/runs/:runId           RunDetailPage        (tab 用 ?tab= query 同步)
/runs/:runId/profile   ProfilePage
/launch                LaunchPage           (?problem=kernelbench:l1:40 支持预填)
/benchmarks            BenchmarksPage
/settings              SettingsPage
*                      NotFound（含返回 Dashboard）
```

### 8.4 实现顺序建议（给实现 agent）

1. 脚手架 + 令牌 + AppShell + statusMap/Badge（纯函数先写测试）。
2. api/types + queries（§6 全部端点接通）→ Runs 列表 + Run 详情概览（StageStepper/泳道/预算条/错误面板）＝ 最小可用替换。
3. Launch 页（含模型拉取、409/400 错误展示、key 清空）。
4. Dashboard、Benchmarks、Settings。
5. 提案 P2-P5 后端联调 → Evidence/Logs/Profile tab。
6. 轮询退避、visibilitychange、空态/NOT_RUN 梳查。

### 8.5 验收清单（前端正例/反例）

- 正例：mock 一个 completed run 快照（含 champion + ratio_ci_95）渲染正确；launch 表单按 §6.6 发送完整 payload。
- 反例：409 时表单显示错误且不清空；api_key 不出现在 localStorage/URL；`candidates: []` + `in_flight` 时泳道正确显示进行中；`ratio_ci_95` 缺失显示 NOT_RUN 而非 0×；`journal_corrupt` 时显示错误横幅且不再轮询。
- 现网验证：`uv run python -m kernelagent.webapp`（或仓库现行启动方式）+ `demo-correct` 模式跑通全流程 UI。

---

## 9. 参考 URL

UCAgent：
- https://github.com/XS-MLVP/UCAgent
- https://github.com/XS-MLVP/UCAgent/blob/main/README.zh.md
- https://ucagent.open-verify.cc/content/02_usage/04_tui/ （TUI 开发者手册）
- https://ucagent.open-verify.cc/content/00_index/ （文档站目录；Web Master 使用指南在"功能介绍"）
- https://github.com/XS-MLVP/UCAgent/blob/main/docs/DOC_PATCH_BUNDLE_WEB_MASTER.md （Launch/Task/Agent 页面功能设计）
- https://github.com/XS-MLVP/hackathon2512 （UCAgent 人机协同验证黑客松，佐证交互定位）

暗色主题与仪表盘美观：
- https://m2.material.io/design/color/dark-theme.html （Material Dark Theme：分层表面/对比度）
- https://www.cmarix.com/blog/8-time-tested-dark-theme-design-tips-to-advance-dashboard-development/
- https://adminlte.io/blog/dark-dashboard-templates/ （不用纯黑/纯白、表面分层表达层级）
- https://www.designstudiouiux.com/blog/dark-mode-ui-design-best-practices/
- https://www.orbix.studio/blogs/saas-dark-mode-ui-design （WCAG 4.5:1 对比度）
- https://www.tech-rz.com/blog/best-dark-mode-dashboard-designs-2027/ （开发者色只留给日志/状态/指标）
- https://dribbble.com/search/swimlane （泳道图视觉参考）
- https://learn.microsoft.com/en-us/azure/devops/report/dashboards/overview?view=azure-devops （流水线仪表盘/泳道控件参照）

实时刷新（polling vs SSE）：
- https://piehost.com/websocket/http-polling-vs-sse
- https://pristren.com/blog/websockets-sse-polling-guide/ （~30s 以上低频更新用轮询的经验法则）
- https://blog.algomaster.io/p/polling-vs-long-polling-vs-sse-vs-websockets-webhooks
- https://javascript.plainenglish.io/short-polling-long-polling-vs-sse-vs-websockets-tradeoffs-use-cases-and-how-to-choose-907b48400cd8
- https://blog.openreplay.com/websockets-sse-long-polling/
- https://www.reddit.com/r/softwarearchitecture/comments/1okvxt2/polling_vs_websockets/ （从业者共识：仪表盘轮询够用，单向才考虑 SSE）

前端技术（选型依据，官方文档）：
- https://tanstack.com/query/latest （服务端状态/轮询/重试）
- https://github.com/pmndrs/zustand （轻量客户端状态）
- https://vitejs.dev/ 、https://tailwindcss.com/ 、https://reactrouter.com/
