import { RouterProvider, createHashRouter } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { DashboardPage } from "./pages/DashboardPage";
import { LaunchPage } from "./pages/LaunchPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunProfilePage } from "./pages/RunProfilePage";
import { RunsPage } from "./pages/RunsPage";
import { SettingsPage } from "./pages/SettingsPage";

/**
 * 路由表（spec §8.3）。当前用 hash 路由：server.py 只托管 "/"（index.html）
 * 与 /api/*，没有 SPA 静态资源/回退路由（设计规范提案 P1 落地后可切换
 * BrowserRouter，仅需改这一处）。
 */
export const router = createHashRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "runs", element: <RunsPage /> },
      { path: "runs/:runId", element: <RunDetailPage /> },
      { path: "runs/:runId/profile", element: <RunProfilePage /> },
      { path: "launch", element: <LaunchPage /> },
      { path: "benchmarks", element: <BenchmarksPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}
