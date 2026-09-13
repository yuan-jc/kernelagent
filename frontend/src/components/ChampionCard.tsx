import type { RunSnapshot } from "../api";
import { CopyButton } from "./CopyButton";
import { Badge, Card, CardContent, CardHeader, CardTitle } from "./ui";
import { shortHash } from "../lib/format";

/**
 * Champion 卡（spec §3.4/§4.3.3）：金色点缀 + cooperative 审查提醒
 * （语义不可弱化）。无 champion 时整卡不渲染（不画占位成功）。
 */
export function ChampionCard({ snapshot }: { snapshot: RunSnapshot }) {
  const champion = snapshot.champion;
  const sha = champion?.candidate_sha256 ?? null;
  if (!champion || !sha) return null;

  return (
    <Card className="border-champion/40">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span aria-hidden="true">🏆</span>
          <span className="text-champion">Champion（晋升候选）</span>
          <Badge variant="warning">cooperative · 需人工审查</Badge>
        </CardTitle>
        <CopyButton text={sha} label="复制 sha256" />
      </CardHeader>
      <CardContent className="space-y-1.5 text-xs">
        <p className="flex flex-wrap items-baseline gap-2">
          <span className="w-28 shrink-0 text-muted">candidate_sha256</span>
          <span className="break-all font-mono text-fg" title={sha}>
            {sha.length > 40 ? `${sha.slice(0, 40)}…` : sha}
          </span>
          <span className="font-mono text-text-3">({shortHash(sha)})</span>
        </p>
        <p className="flex flex-wrap items-baseline gap-2">
          <span className="w-28 shrink-0 text-muted">path</span>
          <span className="break-all font-mono text-fg">
            {champion.path ?? "未报告（—）"}
          </span>
        </p>
        <p className="rounded-md border border-border bg-surface-2 px-2.5 py-2 leading-relaxed text-muted">
          candidate_trust ={" "}
          <span className="font-mono text-warning">
            {snapshot.candidate_trust ?? "未报告"}
          </span>
          ，adversarially_secure ={" "}
          <span className="font-mono">
            {snapshot.adversarially_secure === undefined
              ? "未报告"
              : String(snapshot.adversarially_secure)}
          </span>
          。自动评估为协作式：champion 的正确性与性能结论需经人工审查后方可采信；
          本控制台不将 completed 渲染为"已验证可信"。
        </p>
      </CardContent>
    </Card>
  );
}
