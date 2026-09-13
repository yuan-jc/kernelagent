/**
 * 泳道 / stepper 状态推导（纯函数）。
 *
 * 数据来源优先级（高 -> 低）：
 * 1. report.candidates[]（终态候选，来自服务端 report.json）；
 * 2. journal 事件（P3 stage_started/stage_finished；仅在端点可用时传入）；
 * 3. in_flight[] + progress{}（运行中唯一的一期数据源）。
 *
 * 诚实性约束：单阶段耗时没有时间戳依据，一律显示 "—"（spec §4.1.1，
 * 二期由 P3 + 后端补 ts 后才允许显示数字）。
 */

import type { ActionRecord, CandidateRecord, RunSnapshot } from "../api";
import type { JournalEntry } from "../api";
import { isTerminalRunState } from "./states";
import { STAGES, normalizeStage } from "../components/statusMap";

export type StageNodeState =
  | "done"
  | "running"
  | "failed"
  | "skipped"
  | "pending"
  | "unknown";

export interface Lane {
  id: string;
  /** 数据来源：report=终态记录；live=in_flight/progress；journal=事件流 */
  source: "report" | "live" | "journal";
  status?: string;
  stage?: string;
  detail?: string;
  ratioCi: [number, number] | null;
  sha256?: string;
  stages: StageNodeState[];
  active: boolean;
}

const STAGE_KEYS = STAGES.map((s) => s.key as string);

function blankStages(): StageNodeState[] {
  return STAGE_KEYS.map(() => "pending");
}

function stageIndex(stage: string | undefined): number {
  const key = normalizeStage(stage);
  const idx = STAGE_KEYS.indexOf(key);
  return idx;
}

/** report 终态候选 -> stages 投影 */
function stagesFromReport(c: CandidateRecord): StageNodeState[] {
  const stages = blankStages();
  const status = c.status ?? "unknown";
  const idx = stageIndex(c.stage);

  if (status === "failed") {
    // 失败发生在 idx 阶段（generation 已归并为 generate）
    for (let i = 0; i < idx; i++) stages[i] = "done";
    if (idx >= 0) stages[idx] = "failed";
    else stages[0] = "failed";
    return stages;
  }
  if (status === "rejected") {
    // policy 门拒绝：gate 本身按设计工作，后续阶段未运行（skipped）
    if (idx >= 0) {
      for (let i = 0; i < idx; i++) stages[i] = "done";
      stages[idx] = "skipped";
    }
    return stages;
  }
  if (["promoted", "retained", "measured", "completed", "passed"].includes(status)) {
    for (let i = 0; i <= idx && i < stages.length; i++) stages[i] = "done";
    return stages;
  }
  // 未知状态：在该阶段标 unknown，不猜
  if (idx >= 0) stages[idx] = "unknown";
  return stages;
}

interface JournalStageHistory {
  started: string[]; // 已 stage_started 的阶段（按序）
  finished: Set<string>;
  interrupted: boolean;
  finishedAction: boolean;
}

/** journal 事件 -> 每个候选的阶段历史（仅当 P3 端点/事件可用时调用） */
export function journalStageHistory(
  entries: JournalEntry[],
): Map<string, JournalStageHistory> {
  const map = new Map<string, JournalStageHistory>();
  const get = (id: string): JournalStageHistory => {
    let h = map.get(id);
    if (!h) {
      h = { started: [], finished: new Set(), interrupted: false, finishedAction: false };
      map.set(id, h);
    }
    return h;
  };
  for (const e of entries) {
    const id = e.action_id;
    if (!id) continue;
    if (e.kind === "stage_started" && e.stage) {
      const h = get(id);
      const key = normalizeStage(e.stage);
      if (!h.started.includes(key)) h.started.push(key);
    } else if (e.kind === "stage_finished" && e.stage) {
      get(id).finished.add(normalizeStage(e.stage));
    } else if (e.kind === "action_interrupted") {
      get(id).interrupted = true;
    } else if (e.kind === "action_finished") {
      get(id).finishedAction = true;
    }
  }
  return map;
}

function stagesFromJournal(
  history: JournalStageHistory,
  runTerminal: boolean,
): StageNodeState[] {
  const stages = blankStages();
  for (const key of history.started) {
    const idx = STAGE_KEYS.indexOf(key);
    if (idx < 0) continue;
    if (history.finished.has(key)) {
      stages[idx] = "done";
    } else if (history.finishedAction || history.interrupted || runTerminal) {
      // action 已收尾/被中断，或 run 已到终态，却仍没有 stage_finished（真实
      // 后端目前不回写 stage_finished）：段不能再画"运行中"，标 unknown
      // （无 done/failed 依据，不猜）。
      stages[idx] = "unknown";
    } else {
      stages[idx] = "running";
    }
  }
  return stages;
}

/**
 * 汇总全部泳道。
 * @param snapshot 服务端 run_snapshot
 * @param journal P3 journal 条目（不可用时传 null）
 * @param baselineRecord P4 records/baseline-eager 的 final 记录（可选）。
 *   baseline 不出现在 report.candidates 里，其泳道只能来自 journal/in_flight；
 *   拿到 record 后用它的 status/stage 覆盖，终态徽章才不会误显"运行中"。
 */
export function buildLanes(
  snapshot: RunSnapshot,
  journal: JournalEntry[] | null,
  baselineRecord?: ActionRecord | null,
): Lane[] {
  const lanes = new Map<string, Lane>();
  const make = (id: string, source: Lane["source"]): Lane => {
    let lane = lanes.get(id);
    if (!lane) {
      lane = {
        id,
        source,
        ratioCi: null,
        stages: blankStages(),
        active: false,
      };
      lanes.set(id, lane);
    }
    return lane;
  };

  // 1) report 终态候选
  for (const c of snapshot.candidates ?? []) {
    const id = c.candidate ?? `unknown-${String(c.candidate_sha256 ?? "").slice(0, 8)}`;
    const lane = make(id, "report");
    lane.status = c.status;
    lane.stage = c.stage;
    lane.detail = c.detail;
    lane.sha256 = c.candidate_sha256;
    lane.ratioCi = Array.isArray(c.ratio_ci_95) ? c.ratio_ci_95 : null;
    lane.stages = stagesFromReport(c);
  }

  // 2) journal 事件（补充运行中候选的阶段轨迹）
  const runTerminal = isTerminalRunState(snapshot.state);
  if (journal && journal.length > 0) {
    const history = journalStageHistory(journal);
    for (const [id, h] of history) {
      // 真实 journal 的多数条目是 budget/action 级（无 stage 字段）：
      // 没有任何 stage_started/interrupted 信息时不建泳道，避免画出空管线
      if (h.started.length === 0 && !h.interrupted) continue;
      if (lanes.has(id) && lanes.get(id)!.source === "report") continue;
      const lane = make(id, "journal");
      const stages = stagesFromJournal(h, runTerminal);
      // journal 比 in_flight 快照更细：仅在能提供信息时覆盖
      if (lane.source !== "live" || stages.some((s) => s !== "pending")) {
        lane.stages = stages;
      }
      if (h.interrupted && !lane.status) lane.status = "interrupted";
    }
  }

  // 3) in_flight + progress（一期运行中数据源）
  for (const id of snapshot.in_flight ?? []) {
    const lane = make(id, "live");
    lane.active = true;
    const stage = snapshot.progress?.[id];
    if (stage && lane.source !== "journal") {
      const idx = stageIndex(stage);
      if (idx >= 0 && lane.stages[idx] === "pending") {
        for (let i = 0; i < idx; i++) if (lane.stages[i] === "pending") lane.stages[i] = "done";
        lane.stages[idx] = "running";
      }
    } else if (!stage && lane.source !== "journal") {
      // stage 未知：不猜，保持 pending
    }
  }

  // 4) P4 baseline 记录覆盖（baseline 不在 report.candidates，泳道终态只能
  //    来自 records/baseline-eager；report 来源的泳道不受影响）
  if (baselineRecord?.candidate) {
    const lane = lanes.get(baselineRecord.candidate);
    if (lane && lane.source !== "report") {
      lane.status = baselineRecord.status ?? lane.status;
      lane.stage = baselineRecord.stage ?? lane.stage;
      lane.detail = baselineRecord.detail ?? lane.detail;
      if (runTerminal) {
        // 终态 run：以记录为准重画阶段（无 in_flight 依据，不保留"运行中"段）
        lane.stages = stagesFromReport(baselineRecord as CandidateRecord);
      }
    }
  }

  const list = [...lanes.values()];
  // 活跃泳道置顶，其余按名称稳定排序（baseline 在前）
  list.sort((a, b) => {
    if (a.active !== b.active) return a.active ? -1 : 1;
    return a.id.localeCompare(b.id);
  });
  return list;
}

/** 顶层 run 阶段（单候选视图）：取第一个活跃泳道；无活跃取最后一条 */
export function primaryLane(lanes: Lane[]): Lane | null {
  return lanes.find((l) => l.active) ?? lanes[lanes.length - 1] ?? null;
}
