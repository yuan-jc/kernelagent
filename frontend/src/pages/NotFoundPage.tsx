import { Link } from "react-router-dom";
import { EmptyState } from "../components/ui";
import { IconInbox } from "../components/icons";

export function NotFoundPage() {
  return (
    <EmptyState
      icon={<IconInbox className="h-8 w-8" />}
      title="页面不存在"
      description="请使用侧边栏导航，或返回总览。"
      action={
        <Link
          to="/"
          className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-fg hover:bg-primary-hover"
        >
          返回总览
        </Link>
      }
    />
  );
}
