import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// kernelagent 控制台后端（src/kernelagent/webapp/server.py）的默认端口。
// 可用环境变量 KA_BACKEND_PORT 覆盖。
const backendPort = Number(process.env.KA_BACKEND_PORT ?? 8501);

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    // 产物为单文件 dist/index.html（JS/CSS 全部内联），可直接被
    // server.py 的 GET / 静态托管路径使用；详见 README.md。
    target: "es2022",
    reportCompressedSize: false,
  },
});
