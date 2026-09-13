import type { RunState } from "../api";
import { Badge } from "./ui/Badge";
import { runStateMeta } from "./statusMap";

/** Run 状态徽章：颜色/标签唯一出处为 statusMap.ts（spec §4.2） */
export function RunStateBadge({ state }: { state: RunState }) {
  const meta = runStateMeta(state);
  return (
    <Badge variant={meta.variant} dot={meta.dot}>
      {meta.label}
    </Badge>
  );
}
