import { NavLink, Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

/** 小屏下的水平导航（侧边栏在 md 以下隐藏） */
function MobileNav() {
  const items = [
    { to: "/", label: "总览", end: true },
    { to: "/runs", label: "Runs" },
    { to: "/launch", label: "新建运行" },
    { to: "/benchmarks", label: "Benchmarks" },
    { to: "/settings", label: "设置" },
  ];
  return (
    <nav className="flex gap-1 overflow-x-auto border-b border-border bg-surface px-3 py-2 md:hidden">
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            `rounded-md px-2.5 py-1.5 text-xs whitespace-nowrap ${
              isActive ? "bg-primary-soft text-primary" : "text-muted"
            }`
          }
        >
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}

export function AppLayout() {
  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <MobileNav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 md:px-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
