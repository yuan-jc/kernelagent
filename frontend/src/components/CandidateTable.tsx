import { useState } from "react";
import type { CandidateRecord, RunSnapshot } from "../api";
import { Badge, NotRunBadge, Table, TableWrapper, TBody, TD, TH, THead, TR } from "./ui";
import { candidateStatusMeta } from "./statusMap";
import { shortHash, truncate } from "../lib/format";

/**
 * 候选对比表（spec §3.4）：champion vs baseline vs 全部候选。
 * - 加速比 CI 缺失 -> NOT_RUN 徽章（不显示 0×）；
 * - detail 截断 400 字符，可展开完整内容（候选级错误第二层呈现，spec §4.3.2）。
 */

const DETAIL_TRUNCATE = 400;

function DetailCell({ text }: { text: string | undefined }) {
  const [expanded, setExpanded] = useState(false);
  if (!text) return <span className="text-xs text-text-3">—</span>;
  const cut = expanded ? null : truncate(text, DETAIL_TRUNCATE);
  return (
    <div className="max-w-[24rem]">
      <pre
        className={`whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted ${
          expanded ? "" : "line-clamp-3"
        }`}
      >
        {cut !== null ? `${cut}…` : text}
      </pre>
      {(truncate(text, DETAIL_TRUNCATE) !== null || expanded) && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-0.5 text-[11px] text-primary hover:text-primary-hover"
        >
          {expanded ? "收起" : "展开完整 detail"}
        </button>
      )}
    </div>
  );
}

function CiCell({ value }: { value: [number, number] | null }) {
  if (!value) return <NotRunBadge what="速度比 95% CI（未到 confirm/timing）" />;
  return (
    <span className="font-mono text-xs tabular-nums text-ok" title="confirm 阶段速度比 95% 置信区间">
      {value[0].toFixed(3)}× – {value[1].toFixed(3)}×
    </span>
  );
}

export interface CandidateTableProps {
  snapshot: RunSnapshot;
  /** baseline 的 timing 记录（P4 可用或 mock 时提供；否则 baseline 行不出现） */
  baseline?: CandidateRecord | null;
  championSha: string | null | undefined;
}

export function CandidateTable({ snapshot, baseline, championSha }: CandidateTableProps) {
  const rows: Array<CandidateRecord & { _isBaseline?: boolean }> = [];
  if (baseline) rows.push({ ...baseline, _isBaseline: true });
  for (const c of snapshot.candidates ?? []) rows.push(c);

  return (
    <TableWrapper>
      <Table>
        <THead>
          <TR>
            <TH>候选</TH>
            <TH>sha256</TH>
            <TH>状态</TH>
            <TH>阶段</TH>
            <TH>速度比 95% CI</TH>
            <TH>详情</TH>
          </TR>
        </THead>
        <TBody>
          {rows.length === 0 ? (
            <TR>
              <TD colSpan={6} className="py-6 text-center text-xs text-muted">
                暂无候选记录（运行中的实时进度见「概览」泳道；终态候选来自 report.json）
              </TD>
            </TR>
          ) : (
            rows.map((c, i) => {
              const meta = candidateStatusMeta(c.status);
              const isChampion =
                championSha && c.candidate_sha256 && c.candidate_sha256 === championSha;
              return (
                <TR
                  key={`${c.candidate ?? "row"}-${i}`}
                  className={isChampion ? "bg-warning-soft/40" : undefined}
                >
                  <TD className="whitespace-nowrap font-mono text-xs">
                    {c._isBaseline ? (
                      <span className="text-muted" title="baseline 记录来自 records/baseline-eager（P4）">
                        baseline-eager（基准）
                      </span>
                    ) : (
                      c.candidate ?? "—"
                    )}
                    {isChampion && (
                      <span className="ml-1 text-champion" title="champion（candidate_trust=cooperative，需人工审查）">
                        🏆
                      </span>
                    )}
                  </TD>
                  <TD className="font-mono text-xs text-muted">{shortHash(c.candidate_sha256)}</TD>
                  <TD>
                    <Badge variant={meta.variant}>{meta.label}</Badge>
                  </TD>
                  <TD className="font-mono text-xs text-muted">{c.stage ?? "—"}</TD>
                  <TD>
                    <CiCell
                      value={
                        Array.isArray(c.ratio_ci_95)
                          ? [c.ratio_ci_95[0]!, c.ratio_ci_95[1]!]
                          : null
                      }
                    />
                  </TD>
                  <TD>
                    <DetailCell text={c.detail} />
                  </TD>
                </TR>
              );
            })
          )}
        </TBody>
      </Table>
    </TableWrapper>
  );
}
