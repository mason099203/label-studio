export const ACTIVE_RUN_STATUSES = new Set([
  "queued",
  "preparing",
  "starting",
  "running",
  "started",
  "deferred",
]);

export const RUN_STATUS_LABELS = {
  queued: "排隊中",
  preparing: "準備資料",
  starting: "啟動中",
  running: "訓練中",
  started: "訓練中",
  finished: "已完成",
  failed: "失敗",
  unknown: "未知",
};

/** @param {{ status?: string, epoch?: number, total_epochs?: number, progress_pct?: number }} run */
export function formatRunStatus(run) {
  const key = run?.status ?? "unknown";
  const label = RUN_STATUS_LABELS[key] ?? key;
  if (run?.progress_pct != null && ACTIVE_RUN_STATUSES.has(key)) {
    const epochPart =
      run.epoch != null && run.total_epochs != null ? ` · Epoch ${run.epoch}/${run.total_epochs}` : "";
    return `${label}${epochPart} · ${Math.round(run.progress_pct)}%`;
  }
  return label;
}

/** @param {{ status?: string }} run */
export function isActiveRun(run) {
  return ACTIVE_RUN_STATUSES.has(run?.status ?? "");
}

/** @param {string | null | undefined} value */
export function formatTrainingTime(value) {
  if (!value) return "時間未知";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "時間未知";

  return new Intl.DateTimeFormat("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

/** @param {Record<string, unknown> | null | undefined} metrics */
export function pickSummaryMetrics(metrics) {
  if (!metrics) return [];

  if (metrics.top1 != null || metrics.top5 != null) {
    return [
      ["Top-1", metrics.top1],
      ["Top-5", metrics.top5],
      ["Fitness", metrics.fitness],
    ];
  }

  return [
    ["mAP@0.5", metrics.map50],
    ["mAP@0.5:0.95", metrics.map],
    ["mAP@0.75", metrics.map75],
    ["Precision (mp)", metrics.mp],
    ["Recall (mr)", metrics.mr],
  ];
}

export function formatMetricPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
}
