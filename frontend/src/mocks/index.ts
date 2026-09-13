/**
 * Mock 数据源入口。
 *
 * - MOCKS_ENABLED：开关（VITE_USE_MOCKS=1/0 优先，其次
 *   localStorage["ka-use-mocks"]，默认关）。
 * - mockRequest：与真实 HTTP 同形状的错误面（MockHttpError 由
 *   client.ts 转成 ApiError，页面代码无需感知差异）。
 *
 * 覆盖端点与切换说明见 README「mock 与真实数据切换」。
 */

export { MOCKS_ENABLED } from "./switch";
export { MockHttpError } from "./http";
export { mockRequest } from "./mockApi";
