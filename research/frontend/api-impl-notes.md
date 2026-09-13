# Webapp 后端 API 实现注记（P1–P5）

日期：2026-09-14 凌晨。实现者：C3（后端）。读者：前端 agent、集成者。
规格基线：`research/frontend/frontend-design-spec.md` §7 提案 P1–P6；本文件只记录**实现结果与偏离**。

全部改动位于 `src/kernelagent/webapp/server.py`（+ `tests/test_webapp_api.py`，38 条 HTTP 级测试）。
旧页 `webapp/static/index.html` 未改动；dist 不存在时行为与之前完全一致（`/` 与 `/index.html` 之外一律 404 JSON）。

## 端点清单（全部已用真实 run 目录 + curl 冒烟验证）

| 端点 | 状态 | 说明 |
|---|---|---|
| `GET /` `/index.html` | 200 | 有 `frontend/dist/index.html` 时返回 SPA dist；否则旧版单文件页 |
| `GET /assets/<path>` 及任意 dist 内静态文件 | 200 / 404 | MIME 白名单 + `Cache-Control: public, max-age=31536000, immutable`；缺失返回 `{"error": "no such asset: …"}` 404（**不做** SPA 回退） |
| `GET /<任意非 /api 路径>` | 200 / 404 | SPA 模式下回退 dist index（`Cache-Control: no-cache`）；含 `..` 或绝对路径的请求 404；无 dist 时保持旧行为 404 JSON |
| `GET /api/runs` | 200 | 摘要新增 `started_at`（float Unix 秒或 null） |
| `GET /api/runs/<id>` | 200 / 404 | 语义不变（快照） |
| `GET /api/runs/<id>/report` | 200 / 404 / 500 | report.json 原文；缺失 `{"error": "run X has no report.json yet"}`；坏 JSON → 500 `not valid JSON` |
| `GET /api/runs/<id>/journal?after=N` | 200 / 400 / 404 | 增量 journal 尾部（见下） |
| `GET /api/runs/<id>/records` | 200 / 404 | 列表（spec P4 之外的补充，见偏离 #3） |
| `GET /api/runs/<id>/records/<action>` | 200 / 404 / 500 | 单条 record（脱敏后） |
| `GET /api/runs/<id>/records/<action>/progress` | 200 / 404 / 500 | `{candidate, stage, ts, …}` |
| `GET /api/runs/<id>/workspace` | 200 / 404 | 顶层条目列表 |
| `GET /api/runs/<id>/workspace/file?path=<rel>` | 200 / 400 / 404 / 413 / 415 | 白名单文本文件 |
| `POST /api/runs` `/api/models` | 不变 | 409 单 GPU 语义未动；key 仍只驻内存 |

统一错误体：`{"error": "<human readable>"}`。

## journal 增量语义（P3，前端轮询协议）

- 游标 `after` = 已消费的**物理行数**（0 起）。`after=0` 全量；响应 `{"run_id", "next_after", "entries"}`；
  `next_after = after + len(entries)`（请求超过末尾时安全回落到总行数）。
- 每条 entry 原样透传（所有 kind，含 `stage_started`/`manifest_revised`，`entry_hash`/`prev_hash` 保留）+ 服务端附加 `seq`（= 文件行号，1 起）。
- **撕裂尾行**（正在写入的最后半行）会被扣发：游标停在最后一条完整行，写完后下次轮询自然补上。测试覆盖。
- 中部坏行 → 500；`after` 非整数/负数 → 400。
- 时间戳：只有 `stage_started` 自带 `ts`（optimization 写入）；其余 kind 无 ts，**后端不编造**（spec §7 已标记为待办）。

## 与 spec 的偏离项

1. **P1 dist 根目录**：spec 写"以 `webapp/static/` 为 root"，实际按任务要求使用 `frontend/dist`（默认
   `<repo>/frontend/dist`，env `KA_FRONTEND_DIST` 可覆盖；tests 即用该 env）。当前前端是
   vite-plugin-singlefile 单文件构建（dist 只有 index.html，无 assets 目录），`/assets/*` 现阶段必 404，
   属预期；未来多文件构建无需改后端。映射关系：`/assets/x.js` → `dist/assets/x.js`（Vite 默认 base）。
   另：任意非 `/api` 未匹配路径会**先尝试作为 dist 文件命中**（覆盖 favicon 等public文件），未命中才回退 index；`/assets/` 前缀未命中是硬 404（资源引用不该拿到 HTML）。
2. **P2 坏 JSON 状态码**：spec 只定义缺失 404；实现中"文件存在但非法 JSON/非对象"返回 **500**（404 语义会误导缓存层）。
3. **P4 新增列表端点**：spec 只有单项；任务要求"列表/单项"，故加 `GET …/records`，响应
   `{"run_id", "records": [{"record": "candidate-000", "variant": "final"|"progress", "file": "candidate-000.json", "size": 1234}]}`。
4. **P4 脱敏（有意偏离"原文"）**：record 响应不是字节级原文，服务端做三层清洗后才返回：
   - key 命中 `^(api_?key|authorization|secret|password|passwd)$`（忽略大小写）→ 值替换 `[REDACTED]`；
   - 字符串**恰好等于**任一环境变量值（长度 ≥ 8）→ 整串替换（防精确回显）；
   - 名字匹配 `key|secret|token|password|passwd|credential|auth` 的环境变量的值（长度 ≥ 8）在任意字符串中作为子串出现 → 子串替换（防 "Bearer <key>" 夹带）。
   P2 report / P5 workspace **不做**该清洗：key 只进过生成器客户端内存，不会出现在这些文件里，而 report 是身份链文档、必须保真（测试证明 job.json/error.txt 不落 key）。
5. **P5 列表排除符号链接**：workspace 顶层与目录内 `files` 均只列常规文件/目录，symlink 一律不展示（文件端点同样拒绝）。
6. **P6 SSE 未实现**（任务允许暂缓）：spec §5 已论证轮询是默认方案；stdlib ThreadingHTTPServer 手写 SSE 需要处理客户端断连与代理缓冲，成本不低、收益亚秒级；P3 游标轮询已覆盖"运行中增量"诉求。二期如需，再加 `GET /api/runs/<id>/events`。
7. **路由严格化（行为微变）**：旧实现 `/api/runs/<任何字符串>` 都命中快照路由；现在未知子路径（如 `/api/runs/<id>/whatever`）返回 404 `not found`。旧 UI 只调用已知端点，不受影响。

## workspace 文件端点的安全边界（P5）

- 拒绝顺序：缺 `path` → 400；绝对路径（`/`、`\`、盘符）→ 400；normpath 后仍含 `..` 组件 → 400；
  `resolve()` 后逃出 `resolve()` 后的 workspace 根（含符号链接逃逸）→ 400 `path escapes the workspace`；
  不存在 → 404；非普通文件（目录）→ 400；扩展名不在白名单 → 415；> 256 KiB → 413；前 8 KiB 含 NUL → 415（二进制）。
- 文本扩展白名单：`.bash .c .cfg .cpp .css .csv .cu .cuh .h .hpp .html .ini .js .json .jsonl .log .md .py .sh .toml .ts .tsv .txt .xml .yaml .yml`（无扩展名文件一律 415）。
- 端点只读：无任何 POST/写路径触碰 workspace。

## 穿越/攻击反例覆盖（tests/test_webapp_api.py，全部为真实 HTTP 请求）

- 静态：`/assets/..%2F..%2Fsecret.txt`、`/assets/%2e%2e/secret.txt`、`/../secret.txt`、`/..%2fsecret.txt` → 4xx 且响应不含目标文件内容；
  `/assets/payload.exe` → 404（MIME 白名单）；`/payload.exe` → 回退 index（exe 字节不出网）。
- records：`records/..%2F..%2Fjob` → 404（不回显 job.json）；`records/%2E%2E` → 404 invalid record id。
- workspace：`../../job.json`、`..%2F..%2Fjob.json`、`container-x/../../../etc/passwd` → 400；
  `/etc/passwd`、`%2Fetc%2Fpasswd` → 400 absolute；symlink 文件逃逸与 symlink 目录逃逸 → 400 escapes（且列表不展示 symlink）；
  二进制 NUL → 415；`.bin` → 415；256KiB+1 → 413。
- 脱敏：`MODEL_PROVIDER_API_KEY` 以精确叶子值/夹带子串/`api_key` 键三种形态写入 record → 响应零泄漏，其余字段原样。

## 测试与冒烟数字

- `tests/ -q`：558 passed, 3 skipped（其中本包新增 `tests/test_webapp_api.py` 38 条；`test_webapp.py` 原 10 条未改动仍绿）。
  注：任务给的基线是 519 passed；开工时仓库实际基线已有 +1（并行工作包所致），本包净增 38。
- `ruff check` + `ruff format --check`（webapp 全目录 + 两个测试文件）：通过。
- 冒烟（`python -m kernelagent.webapp --port 8599 --runs-root artifacts/webui`，真实 run 20260913-102604）：
  `/` → 200 text/html 378826 B（dist 单文件）；`/runs/20260913-102604`、`/launch` → 200 text/html（SPA 回退）；
  `/api/health` ok；`/api/runs[0].started_at = 1789266364.73`；`/report` 200；
  `journal?after=0` 17 条 / `next_after=17`，`after=2` 得 seq 3–17；`after=abc` → 400；
  `records` 列出 4 候选 × final+progress；`records/candidate-000/progress` → `{"stage": "generate", "ts": …}`；
  `workspace` 列出 container / timing-inputs 两目录；`workspace/file?path=…/stdout.log` → 200 523 B；
  穿越 4 连击全部 400/404；symlink 逃逸 → 400。测毕进程已停止、端口已关闭。
  legacy 模式（`KA_FRONTEND_DIST` 指向不存在目录）：`/` → 200 17194 B（旧页），`/runs/whatever` → 404 JSON，旧行为无回归。

## 给前端 agent 的对接提醒

1. 现在可以切 **BrowserRouter**（`frontend/src/App.tsx` 只需 `createHashRouter` → `createBrowserRouter`）；
   刷新/深链 `/runs/:id` 由后端回退托管。但当前 dist 是单文件构建，hash 路由同样可用，切换非紧急。
2. journal 轮询：保存 `next_after`，`GET /api/runs/:id/journal?after=<next_after>`；`entries` 直接进事件流渲染，
   泳道时间线用 `stage_started.ts`（唯一可靠时间戳）；终态后再拉一次收尾。
3. records 列表用 `variant` 区分 final/progress；Profile 页的 `batches_ms`、`async_leak` 从 final record 取。
4. 错误体统一 `{"error": string}`；`/report` 的 500 表示 report.json 损坏（配合快照的 `journal_corrupt` 横幅）。
5. `started_at` 在 `/api/runs` 摘要中可能为 null（无 job.json 且 journal 无 stage_started），显示 "—"，勿编造。
6. API key 依旧只允许出现在 POST body；GET 端点集合全部只读、无 key 通路，前端无需做额外脱敏。
