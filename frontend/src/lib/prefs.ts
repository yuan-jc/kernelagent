/**
 * 本地偏好（localStorage，唯一入口）。约束：
 * - 绝不存 API key 或任何凭据（key 只在 Launch 页内存 state）；
 * - 读取失败（隐私模式/禁用）时回退默认值，不让 UI 崩溃。
 */

export type PollInterval = 1500 | 3000 | 5000;

export interface LaunchDefaults {
  maxCandidates: number;
  maxRepairRounds: number;
  gpuBudgetSeconds: number;
  tokenBudget: number;
}

export interface ProviderPreset {
  name: string;
  baseUrl: string;
}

const KEYS = {
  pollInterval: "ka-poll-interval",
  logAutoscroll: "ka-log-autoscroll",
  runsFilter: "ka-runs-filter",
  launchDefaults: "ka-launch-defaults",
  providerPresets: "ka-provider-presets",
  theme: "ka-theme",
  useMocks: "ka-use-mocks",
} as const;

function read<T>(key: string, parse: (raw: string) => T | null, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    const value = parse(raw);
    return value === null ? fallback : value;
  } catch {
    return fallback;
  }
}

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // localStorage 不可用时仅本次会话生效
  }
}

export const prefs = {
  keys: KEYS,

  getPollInterval(): PollInterval {
    return read<PollInterval>(
      KEYS.pollInterval,
      (raw) => {
        const n = Number(raw);
        return ([1500, 3000, 5000] as readonly number[]).includes(n) ? (n as PollInterval) : null;
      },
      1500,
    );
  },
  setPollInterval(value: PollInterval): void {
    write(KEYS.pollInterval, String(value));
  },

  getLogAutoscroll(): boolean {
    return read(KEYS.logAutoscroll, (raw) => (raw === "1" ? true : raw === "0" ? false : null), true);
  },
  setLogAutoscroll(value: boolean): void {
    write(KEYS.logAutoscroll, value ? "1" : "0");
  },

  getRunsFilter<T>(fallback: T, parse: (raw: string) => T | null): T {
    return read(KEYS.runsFilter, parse, fallback);
  },
  setRunsFilter(value: unknown): void {
    try {
      write(KEYS.runsFilter, JSON.stringify(value));
    } catch {
      // ignore
    }
  },

  getLaunchDefaults(): LaunchDefaults {
    return read<LaunchDefaults>(
      KEYS.launchDefaults,
      (raw) => {
        try {
          const obj = JSON.parse(raw) as Partial<LaunchDefaults>;
          if (
            typeof obj.maxCandidates === "number" &&
            typeof obj.maxRepairRounds === "number" &&
            typeof obj.gpuBudgetSeconds === "number" &&
            typeof obj.tokenBudget === "number"
          ) {
            return obj as LaunchDefaults;
          }
          return null;
        } catch {
          return null;
        }
      },
      { maxCandidates: 5, maxRepairRounds: 2, gpuBudgetSeconds: 1800, tokenBudget: 200000 },
    );
  },
  setLaunchDefaults(value: LaunchDefaults): void {
    try {
      write(KEYS.launchDefaults, JSON.stringify(value));
    } catch {
      // ignore
    }
  },

  getProviderPresets(): ProviderPreset[] {
    return read<ProviderPreset[]>(
      KEYS.providerPresets,
      (raw) => {
        try {
          const arr = JSON.parse(raw) as unknown;
          if (!Array.isArray(arr)) return null;
          const cleaned = arr.filter(
            (p): p is ProviderPreset =>
              typeof p === "object" &&
              p !== null &&
              typeof (p as ProviderPreset).name === "string" &&
              typeof (p as ProviderPreset).baseUrl === "string",
          );
          return cleaned;
        } catch {
          return null;
        }
      },
      [{ name: "DeepSeek", baseUrl: "https://api.deepseek.com" }],
    );
  },
  setProviderPresets(value: ProviderPreset[]): void {
    try {
      write(KEYS.providerPresets, JSON.stringify(value));
    } catch {
      // ignore
    }
  },
};
