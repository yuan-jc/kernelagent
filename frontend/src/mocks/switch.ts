/**
 * Mock 数据源开关（唯一出处）。
 *
 * 优先级：
 * 1. 构建期环境变量 VITE_USE_MOCKS=1 / 0（dev server 与 build 都生效）；
 * 2. 运行时 localStorage["ka-use-mocks"] = "1" / "0"（设置页可切换）；
 * 3. 默认关闭（走真实后端）。
 *
 * 打开后 src/api/client.ts 会把全部请求路由到本目录的内存后端，
 * 用于在 C3 的 P2-P5 端点落地前开发与演示。安全约束不变：
 * mock 层同样不读写任何 API key。
 */
function resolveMocksEnabled(): boolean {
  const raw: unknown = import.meta.env.VITE_USE_MOCKS;
  if (raw === "1") return true;
  if (raw === "0") return false;
  try {
    return window.localStorage.getItem("ka-use-mocks") === "1";
  } catch {
    return false;
  }
}

export const MOCKS_ENABLED = resolveMocksEnabled();
