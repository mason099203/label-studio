import { useState, useEffect, useCallback, useMemo } from "react";
import { Typography, Button, Spinner } from "@humansignal/ui";
import {
  IconExternal,
  IconTerminal,
  IconCode,
  IconInfoOutline,
  IconAnalytics,
  IconTrash,
  IconChevronRight,
  IconCopy,
} from "@humansignal/icons";
import { ToggleItems } from "../../components";
import { Select } from "../../components/Form";
import { cn } from "../../utils/bem";
import {
  TRITON_PLAYGROUND_STATE_KEY,
  extractHostFromUrl,
  buildTritonMetricsUrl,
  readManualMonitorServers,
  persistManualMonitorServers,
} from "./tritonUrlState";
import { useAPI } from "../../providers/ApiProvider";
import { useProject } from "../../providers/ProjectProvider";
import "./ModelDeployment.scss";

const rootClass = cn("logs-metrics-tab");
const VIEW_MODES = {
  mine: "我的部署",
  project: "專案全部",
};
const ALL_DEPLOYERS = "__all__";
/**
 * 數據統計磁貼元件
 */
function StatCard({ label, value, unit, icon: Icon, color = "#6366f1" }) {
  return (
    <div className={rootClass.elem("stat-card").toClassName()}>
      <div className={rootClass.elem("stat-icon").toClassName()} style={{ color }}>
        {typeof Icon === "function" ? <Icon size={24} /> : <IconInfoOutline size={24} />}
      </div>
      <div className={rootClass.elem("stat-label").toClassName()}>{label}</div>
      <div className={rootClass.elem("stat-value").toClassName()}>
        {value} <small>{unit}</small>
      </div>
    </div>
  );
}

/**
 * 區塊容器，統一各段落的標題、說明與卡片外觀。
 * @param {{
 *   title: string,
 *   description?: string,
 *   children: import("react").ReactNode
 * }} props
 * @returns {JSX.Element}
 */
function SectionBlock({ title, description, children }) {
  return (
    <div className={rootClass.elem("section").toClassName()}>
      <div className={rootClass.elem("section-header").toClassName()}>
        <Typography variant="title" size="medium">
          {title}
        </Typography>
        {description ? (
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            {description}
          </Typography>
        ) : null}
      </div>
      {children}
    </div>
  );
}

/**
 * 格式化數值，避免畫面顯示過長小數。
 * @param {number | string | null | undefined} value
 * @param {number} decimals
 * @returns {string | number}
 */
function formatValue(value, decimals = 2) {
  if (typeof value !== "number") return value ?? 0;
  return Number.isInteger(value) ? value : value.toFixed(decimals);
}

/**
 * 轉換 Triton 健康狀態顯示樣式。
 * @param {string | null | undefined} status
 * @returns {{ label: string, background: string, color: string }}
 */
function getHealthBadge(status) {
  if (status === "online" || status === "Online") {
    return { label: "Online", background: "#dcfce7", color: "#166534" };
  }

  if (status === "degraded" || status === "Degraded") {
    return { label: "Degraded", background: "#fef9c3", color: "#854d0e" };
  }

  return { label: "Offline", background: "#fee2e2", color: "#991b1b" };
}

/**
 * 建立監控 API 查詢參數。
 * @param {{ pk?: number, scope: string, selectedUserId: string }} params
 * @returns {{ pk?: number, scope: string, user_id?: number, username?: string }}
 */
/**
 * 自 Triton HTTP 基底推導常見 metrics 位址（同主機、埠 8002）。
 * @param {string} tritonBase
 * @returns {string}
 */
function defaultMetricsUrlFromTritonBase(tritonBase) {
  const b = (tritonBase || "").trim();
  if (!b) return "http://localhost:8002/metrics";
  try {
    const u = new URL(b);
    return `${u.protocol}//${u.hostname}:8002/metrics`;
  } catch {
    return "http://localhost:8002/metrics";
  }
}

function buildMetricsParams({ pk, scope, selectedUserId }) {
  const params = { pk, scope };

  if (scope === "project" && selectedUserId && selectedUserId !== ALL_DEPLOYERS) {
    if (selectedUserId.startsWith("id:")) {
      params.user_id = Number(selectedUserId.replace("id:", ""));
    } else if (selectedUserId.startsWith("username:")) {
      params.username = selectedUserId.replace("username:", "");
    }
  }

  return params;
}

/**
 * 產生部署者下拉選單項目。
 * @param {{ id?: number | null, username?: string | null, model_count?: number }[]} deployers
 * @returns {{ label: string, value: string }[]}
 */
function buildDeployerOptions(deployers = []) {
  return [
    { label: "全部部署者", value: ALL_DEPLOYERS },
    ...deployers.map((item) => ({
      label: `${item.username ?? "未記錄"}${item.model_count ? ` (${item.model_count})` : ""}`,
      value: item.id != null ? `id:${item.id}` : `username:${item.username ?? ""}`,
    })),
  ];
}

/**
 * 從 models 陣列中提取所有唯一的 Triton 伺服器清單（含各台上的模型名稱）。
 * 優先使用 triton_servers 陣列；若無則退回 triton_public_base_url 單一值。
 * @param {Array<{name?: string, triton_servers?: Array<{url:string}>, triton_public_base_url?: string}>} models
 * @returns {Array<{url: string, ip: string, models: string[]}>}
 */
function extractServersFromModels(models) {
  /** @type {Map<string, {url: string, ip: string, models: string[]}>} */
  const map = new Map();

  for (const m of models || []) {
    const serverList =
      Array.isArray(m.triton_servers) && m.triton_servers.length > 0
        ? m.triton_servers
        : m.triton_public_base_url
          ? [{ url: m.triton_public_base_url }]
          : [];

    for (const s of serverList) {
      const url = (s.url || "").trim().replace(/\/+$/, "");
      if (!url) continue;
      if (!map.has(url)) {
        map.set(url, { url, ip: extractHostFromUrl(url), models: [] });
      }
      if (m.name) map.get(url).models.push(m.name);
    }
  }

  return Array.from(map.values());
}

/**
 * 單台 Triton 伺服器的即時效能面板。
 * @param {{
 *   ip: string,
 *   models: string[],
 *   metricsData: Object | null,
 *   isManual?: boolean,
 * }} props
 */
function ServerPerformancePanel({ ip, models, metricsData, isManual = false }) {
  const health = metricsData?.health ?? {};
  const hardware = metricsData?.hardware ?? {};
  const inference = metricsData?.inference ?? {};
  const badge = getHealthBadge(metricsData ? (health.status ?? "Offline") : null);
  const isLoading = !metricsData;
  const hasError = metricsData?.error;
  const tritonHasCpuMetrics = hardware.metrics_available && hardware.cpu > 0;
  // 以後端旗標判斷 VRAM / RAM 指標是否實際可用，避免顯示 0 GB / 0%
  const tritonHasVramMetrics =
    hardware.metrics_available && (hardware.vram_available === true || hardware.vram_total_gb > 0);
  const tritonHasRamMetrics =
    hardware.metrics_available && (hardware.ram_mem_available === true || hardware.ram_used_gb > 0);

  const hardwareStats = [
    { label: "GPU 使用率", value: formatValue(hardware.gpu), unit: "%", icon: IconAnalytics, color: "#10b981" },
    {
      label: "VRAM 佔用",
      value: tritonHasVramMetrics ? formatValue(hardware.vram_used_gb) : "—",
      unit: tritonHasVramMetrics
        ? hardware.vram_total_gb > 0
          ? `GB / ${formatValue(hardware.vram_total_gb)} GB`
          : "GB"
        : "",
      icon: IconTerminal,
      color: "#6366f1",
    },
    {
      label: "CPU 使用率",
      value: tritonHasCpuMetrics ? formatValue(hardware.cpu) : "—",
      unit: tritonHasCpuMetrics ? "%" : "",
      icon: IconCode,
      color: "#f59e0b",
    },
    {
      label: "系統記憶體",
      value: tritonHasRamMetrics ? formatValue(hardware.ram) : "—",
      unit: tritonHasRamMetrics ? `% (${formatValue(hardware.ram_used_gb)} GB)` : "",
      icon: IconInfoOutline,
      color: "#3b82f6",
    },
  ];

  const perfStats = [
    { label: "平均延遲", value: formatValue(inference.latency), unit: "ms", icon: IconAnalytics },
    { label: "每秒請求", value: formatValue(inference.rps), unit: "req/s", icon: IconExternal },
    { label: "成功率", value: formatValue(inference.success_rate), unit: "%", icon: IconAnalytics },
    {
      label: "啟用模型數",
      value: formatValue(inference.active_models ?? 0, 0),
      unit: `/ ${formatValue(inference.total_models ?? 0, 0)}`,
      icon: IconCode,
      color: "#10b981",
    },
  ];

  const rootClass = cn("logs-metrics-tab");

  return (
    <div className={rootClass.elem("server-panel").toClassName()}>
      <div className={rootClass.elem("server-panel-header").toClassName()}>
        <span className={rootClass.elem("server-badge-ip").toClassName()}>{ip}</span>
        <span
          className={rootClass.elem("service-badge").toClassName()}
          style={{ background: badge.background, color: badge.color }}
        >
          {isLoading ? "載入中…" : badge.label}
        </span>
        <span className={rootClass.elem("meta-chip").toClassName()}>
          {isLoading ? "—" : health.metrics_available ? "Metrics 已連線" : "Metrics 未連線"}
        </span>
        {isManual && <span className={rootClass.elem("manual-badge").toClassName()}>手動新增</span>}
        {models.length > 0 && (
          <span className={rootClass.elem("meta-chip").toClassName()} title={models.join(", ")}>
            模型：{models.length <= 3 ? models.join(", ") : `${models.slice(0, 3).join(", ")} …+${models.length - 3}`}
          </span>
        )}
        {hasError && (
          <span style={{ color: "var(--color-negative-content, #991b1b)", fontSize: 12 }}>連線失敗：{hasError}</span>
        )}
      </div>
      {isLoading ? (
        <div className={rootClass.elem("server-panel-loading").toClassName()}>
          <Spinner size={20} />
          <span>正在查詢伺服器指標…</span>
        </div>
      ) : (
        <>
          <div className={rootClass.elem("server-panel-group-label").toClassName()}>硬體資源</div>
          <div className={rootClass.elem("stats-grid").toClassName()}>
            {hardwareStats.map((s) => (
              <StatCard key={s.label} {...s} />
            ))}
          </div>
          <div className={rootClass.elem("server-panel-group-label").toClassName()}>推論效能</div>
          <div className={rootClass.elem("stats-grid").toClassName()}>
            {perfStats.map((s) => (
              <StatCard key={s.label} {...s} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/**
 * 格式化 ISO 時間字串。
 * @param {string | null | undefined} value
 * @returns {string}
 */
function formatDateTime(value) {
  if (!value) return "未記錄";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

/**
 * 讀取最近使用的專案 ID，避免全域頁面缺少專案上下文時完全無法操作。
 * @returns {string}
 */
function readSavedProjectId() {
  if (typeof window === "undefined") return "";

  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed?.projectId ? String(parsed.projectId) : "";
  } catch (_) {
    return "";
  }
}

/**
 * 儲存最近查詢的專案 ID，讓全域監控頁可直接回填。
 * @param {string} projectId
 */
function persistProjectId(projectId) {
  if (typeof window === "undefined") return;

  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    window.localStorage.setItem(
      TRITON_PLAYGROUND_STATE_KEY,
      JSON.stringify({
        ...parsed,
        projectId: projectId || "",
      }),
    );
  } catch (_) {
    // Ignore storage errors to keep the monitor view usable.
  }
}

export function LogsAndMetricsTab() {
  const api = useAPI();
  const project = useProject()?.project;
  const [projectId, setProjectId] = useState(() => String(project?.id ?? readSavedProjectId() ?? ""));
  /**
   * 所有可選專案清單，用於下拉篩選。
   * @type {[Array<{id: number, title: string}>, Function]}
   */
  const [projects, setProjects] = useState([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  /**
   * 從部署 metadata 自動偵測到的 Triton 伺服器清單。
   * 每項包含 { url, ip, models[] }，無需使用者手動輸入 IP。
   * @type {[Array<{url:string, ip:string, models:string[]}>, Function]}
   */
  const [autoServers, setAutoServers] = useState([]);
  /**
   * 各 Triton 伺服器的即時指標快取（serverUrl → metrics 回應物件）。
   * @type {[Map<string, Object>, Function]}
   */
  const [serverMetricsMap, setServerMetricsMap] = useState(new Map());
  const [loading, setLoading] = useState(true);
  /** 輪詢更新時（資料已存在的背景刷新）顯示小型載入指示器。 */
  const [refreshing, setRefreshing] = useState(false);
  const [data, setData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [scope, setScope] = useState("mine");
  const [selectedUserId, setSelectedUserId] = useState(ALL_DEPLOYERS);
  /** @type {[Set<string>, Function]} 目前展開的模型名稱集合 */
  const [expandedModels, setExpandedModels] = useState(new Set());
  /** @type {[string | null, Function]} 正在刪除中的模型名稱 */
  const [deletingModel, setDeletingModel] = useState(null);
  /**
   * 正在刪除中的版本鍵，格式為 `${modelName}::${version}`。
   * @type {[Set<string>, Function]}
   */
  const [deletingVersionKeys, setDeletingVersionKeys] = useState(new Set());
  /** 剛複製成功的列 rowKey，短暫顯示「已複製」反饋後自動清除。 */
  const [copiedKey, setCopiedKey] = useState(null);

  /**
   * 使用者手動新增的 Triton 監控伺服器清單，從 localStorage 初始化。
   * @type {[Array<{ url: string }>, Function]}
   */
  const [manualServers, setManualServers] = useState(() => readManualMonitorServers());
  /** 新增伺服器表單：使用者輸入的 URL 文字 */
  const [newServerUrl, setNewServerUrl] = useState("");
  /** 新增伺服器表單：驗證錯誤訊息 */
  const [addServerError, setAddServerError] = useState("");

  /** 元件掛載時拉取所有專案，供名稱下拉篩選使用。 */
  useEffect(() => {
    setProjectsLoading(true);
    api
      .callApi("projects")
      .then((res) => {
        const list = Array.isArray(res) ? res : (res?.results ?? []);
        setProjects(list);
      })
      .catch(() => {
        setProjects([]);
      })
      .finally(() => {
        setProjectsLoading(false);
      });
  }, [api]);

  useEffect(() => {
    if (project?.id) {
      setProjectId(String(project.id));
    }
  }, [project?.id]);

  useEffect(() => {
    persistProjectId(projectId);
  }, [projectId]);

  /**
   * 由 projectId 查找對應的專案名稱。
   * @type {string}
   */
  const selectedProjectTitle = useMemo(() => {
    if (!projectId) return "";
    const found = projects.find((p) => String(p.id) === String(projectId));
    return found?.title ?? projectId;
  }, [projectId, projects]);

  /**
   * 供下拉選單使用的專案選項。
   * @type {{ label: string, value: string }[]}
   */
  const projectOptions = useMemo(
    () => [
      { label: "— 選擇專案 —", value: "" },
      ...projects.map((p) => ({ label: p.title || `Project ${p.id}`, value: String(p.id) })),
    ],
    [projects],
  );

  /**
   * 自動偵測伺服器與手動新增伺服器合併後的完整清單（去重）。
   * 自動偵測的伺服器優先（相同 URL 時保留自動版本）。
   * @type {Array<{ url: string, ip: string, models: string[], _manual?: boolean }>}
   */
  const allServers = useMemo(() => {
    const map = new Map(autoServers.map((s) => [s.url, s]));
    for (const ms of manualServers) {
      const u = (ms.url || "").trim().replace(/\/+$/, "");
      if (u && !map.has(u)) {
        map.set(u, { url: u, ip: extractHostFromUrl(u), models: [], _manual: true });
      }
    }
    return Array.from(map.values());
  }, [autoServers, manualServers]);

  const fetchMetrics = useCallback(async () => {
    if (!projectId) {
      setLoading(false);
      setData(null);
      setHistoryData(null);
      return;
    }
    // 資料已存在時進入「背景刷新」模式，顯示小型指示器而非全頁 Spinner
    setRefreshing(true);
    try {
      const params = buildMetricsParams({ pk: projectId, scope, selectedUserId });

      // ── Step 1：取得模型清單（含 triton_servers）、推論事件與快照 ──────────
      const [metricsRes, metricsHistoryRes] = await Promise.all([
        // 不帶 triton_url，由伺服器預設值提供 health check（取部署 metadata 為主）
        api.callApi("trainingMetrics", { params }),
        api.callApi("trainingMetricsHistory", {
          params: { ...params, event_limit: 12, snapshot_limit: 12 },
        }),
      ]);
      setData(metricsRes);
      setHistoryData(metricsHistoryRes);

      // ── Step 2：從模型 triton_servers 欄位提取唯一伺服器清單 ──────────────
      const allModels = metricsRes?.models ?? [];
      const servers = extractServersFromModels(allModels);
      setAutoServers(servers);

      // ── Step 3：合併自動偵測 + 手動新增伺服器，對所有台平行查詢即時 Metrics ──
      // 使用 Map 去重：相同 URL 的自動偵測版本優先保留
      const serverMap = new Map(servers.map((s) => [s.url, s]));
      for (const ms of manualServers) {
        const u = (ms.url || "").trim().replace(/\/+$/, "");
        if (u && !serverMap.has(u)) {
          serverMap.set(u, { url: u, ip: extractHostFromUrl(u), models: [] });
        }
      }
      const allServersToQuery = Array.from(serverMap.values());

      if (allServersToQuery.length > 0) {
        // 先將所有伺服器標記為「載入中」（null 表示載入中，讓 panel 顯示 spinner）
        setServerMetricsMap((prev) => {
          const next = new Map(prev);
          for (const { url } of allServersToQuery) {
            if (!next.has(url)) next.set(url, null);
          }
          return next;
        });

        const serverResults = await Promise.all(
          allServersToQuery.map(async ({ url }) => {
            const metricsUrl = buildTritonMetricsUrl(extractHostFromUrl(url));
            try {
              const r = await api.callApi("trainingMetrics", {
                params: {
                  ...params,
                  triton_url: url,
                  ...(metricsUrl ? { triton_metrics_url: metricsUrl } : {}),
                },
              });
              return { url, result: r, error: null };
            } catch (err) {
              return { url, result: null, error: err?.message ?? String(err) };
            }
          }),
        );

        setServerMetricsMap(
          new Map(serverResults.map(({ url, result, error }) => [url, { ...(result ?? {}), error }])),
        );
      } else {
        setServerMetricsMap(new Map());
      }
    } catch (e) {
      console.error("Failed to fetch metrics", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [api, projectId, scope, selectedUserId, manualServers]);

  /**
   * 新增手動監控伺服器。
   * 驗證 URL 格式、防止重複，並寫入 localStorage。
   */
  const handleAddServer = useCallback(() => {
    const raw = newServerUrl.trim();
    if (!raw) return;

    if (!/^https?:\/\//i.test(raw)) {
      setAddServerError("請輸入完整的 URL（需以 http:// 或 https:// 開頭）");
      return;
    }

    // 正規化：若未指定 port 則補上 Triton 預設推論埠 18000
    let normalized = raw.replace(/\/+$/, "");
    try {
      const parsed = new URL(raw);
      if (!parsed.port) {
        parsed.port = "18000";
      }
      normalized = parsed.origin;
    } catch (_) {
      setAddServerError("URL 格式無效，請確認主機與埠號");
      return;
    }

    if (manualServers.some((s) => s.url === normalized)) {
      setAddServerError("此伺服器已在監控清單中");
      return;
    }

    const updated = [...manualServers, { url: normalized }];
    setManualServers(updated);
    persistManualMonitorServers(updated);
    setNewServerUrl("");
    setAddServerError("");
  }, [newServerUrl, manualServers]);

  /**
   * 移除手動監控伺服器。
   * @param {string} url - 要移除的伺服器 URL
   */
  const handleRemoveServer = useCallback(
    (url) => {
      const updated = manualServers.filter((s) => s.url !== url);
      setManualServers(updated);
      persistManualMonitorServers(updated);
      // 同步清除 serverMetricsMap 中已移除伺服器的快取
      setServerMetricsMap((prev) => {
        const next = new Map(prev);
        next.delete(url);
        return next;
      });
    },
    [manualServers],
  );

  /**
   * 切換單一列（模型 × 伺服器）的展開狀態。
   * @param {string} rowKey - 格式為 `${modelName}::${serverUrl}`
   */
  const toggleExpand = useCallback((rowKey) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(rowKey)) {
        next.delete(rowKey);
      } else {
        next.add(rowKey);
      }
      return next;
    });
  }, []);

  /**
   * 從指定伺服器移除已部署模型。
   * - 帶 triton_url：只從該台伺服器移除（其他伺服器不受影響）
   * - 不帶 triton_url：完整刪除（向後相容）
   * @param {string} modelName   - Triton 模型名稱
   * @param {string} serverUrl   - 目標 Triton 基底 URL（空字串表示刪除全部）
   * @param {string} rowKey      - 當前列唯一鍵，用於清除展開狀態
   */
  const handleDeleteModelFromServer = useCallback(
    async (modelName, serverUrl, rowKey) => {
      const serverIp = serverUrl ? extractHostFromUrl(serverUrl) : "";
      const confirmMsg = serverUrl
        ? `確定要從伺服器 [${serverIp}] 移除部署模型「${modelName}」嗎？\n其餘伺服器的部署不受影響，此操作不可復原。`
        : `確定要完整移除部署模型「${modelName}」嗎？\n此操作不可復原。`;
      if (!window.confirm(confirmMsg)) return;

      setDeletingModel(rowKey);
      try {
        const params = { pk: projectId, model_name: modelName };
        if (serverUrl) params.triton_url = serverUrl;
        await api.callApi("trainingTritonModelDelete", { params });
        await fetchMetrics();
        setExpandedModels((prev) => {
          const next = new Set(prev);
          next.delete(rowKey);
          return next;
        });
      } catch (err) {
        window.alert(`刪除失敗：${err?.message ?? err}`);
      } finally {
        setDeletingModel(null);
      }
    },
    [api, projectId, fetchMetrics],
  );

  /**
   * 刪除指定模型的單一版本目錄。
   * 刪除後重新整理指標資料；若模型所有版本都刪光則同時清除展開狀態。
   *
   * @param {string} modelName  - Triton 模型名稱
   * @param {number} version    - 要刪除的版本號
   * @param {string} rowKey     - 該模型列的唯一鍵（用於清除展開狀態）
   */
  const handleDeleteVersion = useCallback(
    async (modelName, version, rowKey) => {
      if (
        !window.confirm(
          `確定要刪除模型「${modelName}」的版本 ${version} 嗎？\n此操作不可復原，Triton 將無法再載入此版本。`,
        )
      )
        return;

      const vKey = `${modelName}::${version}`;
      setDeletingVersionKeys((prev) => new Set([...prev, vKey]));
      try {
        // 使用與 handleDeleteModelFromServer 相同的呼叫模式（不用 errorFilter，以 try/catch 捕捉錯誤）
        const res = await api.callApi("trainingTritonVersionDelete", {
          params: { pk: projectId, model_name: modelName, version },
        });
        if (res?.detail) {
          window.alert(`刪除失敗：${res.detail}`);
          return;
        }
        // 若整個模型都已移除，清除展開狀態
        if (res?.model_removed) {
          setExpandedModels((prev) => {
            const next = new Set(prev);
            next.delete(rowKey);
            return next;
          });
        }
        await fetchMetrics();
      } catch (err) {
        window.alert(`刪除版本失敗：${err?.message ?? err}`);
      } finally {
        setDeletingVersionKeys((prev) => {
          const next = new Set(prev);
          next.delete(vKey);
          return next;
        });
      }
    },
    [api, projectId, fetchMetrics],
  );

  useEffect(() => {
    if (scope === "mine" && selectedUserId !== ALL_DEPLOYERS) {
      setSelectedUserId(ALL_DEPLOYERS);
    }
  }, [scope, selectedUserId]);

  useEffect(() => {
    fetchMetrics();
    const timer = setInterval(fetchMetrics, 5000); // 5 sec poll
    return () => clearInterval(timer);
  }, [fetchMetrics]);

  if (loading && !data && projectId) return <Spinner size={32} centered />;

  if (!projectId) {
    return (
      <section className={rootClass.toClassName()}>
        <div className={rootClass.elem("hero").toClassName()}>
          <div className={rootClass.elem("hero-copy").toClassName()}>
            <Typography variant="headline" size="medium">
              日誌與效能
            </Typography>
            <Typography variant="body" size="small" className="text-neutral-content-subtle">
              查看模型部署、推論請求與系統監控資料。
            </Typography>
          </div>
        </div>
        <div className={rootClass.elem("panel").mod({ compact: true }).toClassName()}>
          <Typography variant="title" size="medium" className="mb-tight">
            需要先指定專案
          </Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle mb-wide">
            這個頁面是全域頁面，若目前不在專案內，請先從下拉選單選擇要查詢的專案。
          </Typography>
          <div className={rootClass.elem("project-form").toClassName()}>
            <div className={rootClass.elem("field").toClassName()}>
              <label className={rootClass.elem("field-label").toClassName()}>專案</label>
              <select
                className={rootClass.elem("text-input").toClassName()}
                value={projectId}
                onChange={(event) => setProjectId(event.target.value)}
                disabled={projectsLoading}
              >
                {projectOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <Button variant="primary" onClick={fetchMetrics} disabled={!projectId || projectsLoading}>
              載入監控資料
            </Button>
          </div>
        </div>
      </section>
    );
  }

  const modelUsage = data?.models ?? [];
  const deployerOptions = buildDeployerOptions(data?.filters?.deployers);
  const selectedUser = data?.filters?.selected_user;
  const snapshots = historyData?.snapshots ?? [];
  const events = historyData?.events ?? [];

  /**
   * 將 modelUsage 展開為「模型 × 伺服器」資料列陣列。
   * 每筆 model 若有多台 triton_servers，就產生多列（各列可獨立刪除）。
   * 指標優先從對應伺服器的 serverMetricsMap 取得；若未命中則沿用 data.models 的彙總值。
   *
   * @type {Array<{
   *   _rowKey: string,
   *   _serverUrl: string,
   *   _serverIp: string | null,
   *   _serverDeployedAt: string,
   *   name: string,
   *   deployed_by_username: string,
   *   request_count: number,
   *   rps: number,
   *   latency_ms: number,
   *   success_count: number,
   *   error_count: number,
   *   status: string,
   * }>}
   */
  const tableRows = modelUsage.flatMap((m) => {
    const serverList =
      Array.isArray(m.triton_servers) && m.triton_servers.length > 0
        ? m.triton_servers
        : [{ url: m.triton_public_base_url ?? "", deployed_at: m.deployed_at ?? "" }];

    return serverList.map((s) => {
      const serverUrl = (s.url || "").trim().replace(/\/+$/, "");
      const serverIp = serverUrl ? extractHostFromUrl(serverUrl) : null;
      const rowKey = `${m.name}::${serverUrl}`;

      // 從對應伺服器的 metrics 取得單台效能數據
      const srvData = serverUrl ? serverMetricsMap.get(serverUrl) : null;
      const srvModel = srvData?.models?.find((sm) => sm.name === m.name);

      return {
        ...m,
        // 用每台伺服器的 metrics 覆蓋，若無則使用彙總值（僅單台時準確）
        request_count: srvModel?.request_count ?? (serverList.length === 1 ? m.request_count : 0),
        rps: srvModel?.rps ?? (serverList.length === 1 ? m.rps : 0),
        latency_ms: srvModel?.latency_ms ?? (serverList.length === 1 ? m.latency_ms : 0),
        success_count: srvModel?.success_count ?? (serverList.length === 1 ? m.success_count : 0),
        error_count: srvModel?.error_count ?? (serverList.length === 1 ? m.error_count : 0),
        status: srvModel?.status ?? (serverUrl ? "—" : m.status),
        // 列識別
        _rowKey: rowKey,
        _serverUrl: serverUrl,
        _serverIp: serverIp,
        _serverDeployedAt: s.deployed_at ?? m.deployed_at ?? "",
        _uploadServerUrl: s.upload_server_url ?? "",
      };
    });
  });

  return (
    <section className={rootClass.toClassName()}>
      <div className={rootClass.elem("hero").toClassName()}>
        <div className={rootClass.elem("hero-copy").toClassName()}>
          <Typography variant="headline" size="medium">
            儀錶板
          </Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            即時查看部署模型、推論事件、效能快照與監控入口。
          </Typography>
          <div className={rootClass.elem("hero-meta").toClassName()}>
            <span className={rootClass.elem("meta-chip").toClassName()}>{selectedProjectTitle || projectId}</span>
            <span className={rootClass.elem("meta-chip").toClassName()}>
              檢視範圍: {VIEW_MODES[scope]}
              {data?.viewer?.username ? ` (${data.viewer.username})` : ""}
            </span>
            {scope === "project" && selectedUser?.username ? (
              <span className={rootClass.elem("meta-chip").toClassName()}>部署者: {selectedUser.username}</span>
            ) : null}
            <span className={rootClass.elem("meta-chip").toClassName()}>已偵測伺服器: {autoServers.length} 台</span>
          </div>
        </div>
        <div className={rootClass.elem("toolbar").toClassName()}>
          <div className={rootClass.elem("field").mod({ project: true }).toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()}>
              專案
              {refreshing && data ? (
                <span className={rootClass.elem("refresh-indicator").toClassName()} title="資料更新中…">
                  <span className={rootClass.elem("refresh-dot").toClassName()} />
                  <span className={rootClass.elem("refresh-label").toClassName()}>更新中</span>
                </span>
              ) : null}
            </label>

            <select
              className={rootClass.elem("text-input").toClassName()}
              value={projectId}
              onChange={(event) => setProjectId(event.target.value)}
              disabled={projectsLoading}
            >
              {projectOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <Button variant="neutral" look="outlined" onClick={fetchMetrics} disabled={!projectId || refreshing}>
            重新載入
          </Button>
          {/* 輪詢更新時顯示小型旋轉指示器（初次全頁載入不顯示，由 Spinner 處理） */}

          <div className={rootClass.elem("filters").toClassName()}>
            <ToggleItems items={VIEW_MODES} active={scope} onSelect={setScope} />
            <div className={rootClass.elem("select-wrap").toClassName()}>
              <Select
                value={selectedUserId}
                onChange={setSelectedUserId}
                options={deployerOptions}
                placeholder="依部署者篩選"
                disabled={scope !== "project"}
              />
            </div>
          </div>
        </div>
      </div>

      <SectionBlock
        title={`Triton 伺服器效能監控${allServers.length > 0 ? `（${autoServers.length} 自動 + ${manualServers.length} 手動，共 ${allServers.length} 台）` : ""}`}
        description="每 5 秒自動更新。硬體指標（GPU / VRAM / CPU / RAM）需 Triton Prometheus Metrics（埠 8002）可存取。可手動新增任意 Triton 伺服器以納入監控。"
      >
        {/* ── 手動新增伺服器表單 ──────────────────────────────────────────────── */}
        <div className={rootClass.elem("manual-server-form").toClassName()}>
          <div className={rootClass.elem("manual-server-input-row").toClassName()}>
            <input
              type="text"
              className={rootClass.elem("text-input").toClassName()}
              placeholder="http://10.214.57.20:18000"
              value={newServerUrl}
              onChange={(e) => {
                setNewServerUrl(e.target.value);
                setAddServerError("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAddServer();
              }}
            />
            <Button variant="primary" look="outlined" onClick={handleAddServer} disabled={!newServerUrl.trim()}>
              新增監控
            </Button>
          </div>
          {addServerError && (
            <div className={rootClass.elem("manual-server-error").toClassName()}>{addServerError}</div>
          )}
          {/* 已新增的手動伺服器 tag 列表 */}
          {manualServers.length > 0 && (
            <div className={rootClass.elem("manual-server-list").toClassName()}>
              {manualServers.map((s) => (
                <span key={s.url} className={rootClass.elem("manual-server-item").toClassName()}>
                  <span className={rootClass.elem("server-badge-ip").toClassName()}>{extractHostFromUrl(s.url)}</span>
                  <span className={rootClass.elem("manual-server-url").toClassName()}>{s.url}</span>
                  <button
                    type="button"
                    className={rootClass.elem("manual-server-remove").toClassName()}
                    onClick={() => handleRemoveServer(s.url)}
                    title={`移除 ${s.url}`}
                  >
                    <IconTrash size={12} />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* ── 伺服器效能面板 ──────────────────────────────────────────────────── */}
        {allServers.length === 0 ? (
          <div
            className={rootClass.elem("empty-cell").toClassName()}
            style={{ padding: "28px 0", textAlign: "center" }}
          >
            尚未偵測到 Triton 伺服器；請輸入 Triton 位址或先完成模型部署
          </div>
        ) : (
          <div className={rootClass.elem("server-panels").toClassName()}>
            {allServers.map(({ url, ip, models: serverModels, _manual }) => (
              <ServerPerformancePanel
                key={url}
                ip={ip}
                models={serverModels}
                isManual={Boolean(_manual)}
                metricsData={serverMetricsMap.has(url) ? serverMetricsMap.get(url) : null}
              />
            ))}
          </div>
        )}
      </SectionBlock>

      <SectionBlock
        title="部署模型"
        description="每列對應一個模型部署至單台 Triton 伺服器；點擊列可展開詳情，每列可獨立移除。"
      >
        <div className={rootClass.elem("table-wrap").toClassName()}>
          <table className={rootClass.elem("table").mod({ compact: true }).toClassName()}>
            <thead>
              <tr>
                {/* 展開箭頭 */}
                <th style={{ width: 28 }} />
                {/* 模型名稱：最寬，含副標題 */}
                <th style={{ width: 170 }}>模型名稱</th>
                {/* 部署者 */}
                <th style={{ width: 96 }}>部署者</th>
                {/* Triton 伺服器 IP badge */}
                <th style={{ width: 148 }}>Triton 伺服器</th>
                {/* 數字欄：右對齊 */}
                <th style={{ width: 80, textAlign: "right" }}>請求總數</th>
                <th style={{ width: 76, textAlign: "right" }}>RPS (req/s)</th>
                <th style={{ width: 92, textAlign: "right" }}>平均延遲 (ms)</th>
                <th style={{ width: 104, textAlign: "right" }}>成功 / 失敗</th>
                {/* 狀態 badge：置中 */}
                <th style={{ width: 82, textAlign: "center" }}>狀態</th>
                {/* 操作按鈕：置中 */}
                <th style={{ width: 72, textAlign: "center" }}>操作</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.length > 0 ? (
                tableRows.map((row) => {
                  const badge = getHealthBadge(row.status);
                  const isExpanded = expandedModels.has(row._rowKey);
                  const isDeleting = deletingModel === row._rowKey;
                  /** 該模型共部署於幾台伺服器（用於顯示提示） */
                  const totalServers = (Array.isArray(row.triton_servers) ? row.triton_servers : []).length || 1;

                  return [
                    /* ── 主列：(模型, 伺服器) 一列 ── */
                    <tr
                      key={row._rowKey}
                      className={rootClass.elem("model-row").mod({ expanded: isExpanded }).toClassName()}
                      style={{ cursor: "pointer" }}
                      onClick={() => toggleExpand(row._rowKey)}
                    >
                      <td style={{ paddingRight: 0 }}>
                        <IconChevronRight
                          size={14}
                          style={{
                            transition: "transform 0.2s",
                            transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                            color: "var(--color-neutral-content-subtle)",
                          }}
                        />
                      </td>
                      <td className={rootClass.elem("table-primary").toClassName()}>
                        <div className={rootClass.elem("model-name-row").toClassName()}>
                          <span className={rootClass.elem("model-name").toClassName()}>{row.name}</span>
                          {/* 複製模型名稱按鈕：滑鼠移入列時顯示 */}
                          <button
                            type="button"
                            className={rootClass
                              .elem("copy-btn")
                              .mod({ copied: copiedKey === row._rowKey })
                              .toClassName()}
                            title={copiedKey === row._rowKey ? "已複製！" : `複製「${row.name}」`}
                            onClick={(e) => {
                              e.stopPropagation();
                              navigator.clipboard.writeText(row.name).then(() => {
                                setCopiedKey(row._rowKey);
                                setTimeout(() => setCopiedKey(null), 1500);
                              });
                            }}
                          >
                            {copiedKey === row._rowKey ? (
                              <span className={rootClass.elem("copy-check").toClassName()}>✓</span>
                            ) : (
                              <IconCopy size={12} />
                            )}
                          </button>
                        </div>
                        <small className={rootClass.elem("model-meta").toClassName()}>
                          {row.owned_by_current_user ? "目前使用者部署" : "其他成員部署"}
                          {totalServers > 1 ? ` · 共 ${totalServers} 台伺服器` : ""}
                        </small>
                      </td>
                      <td>{row.deployed_by_username ?? "未記錄"}</td>
                      <td>
                        {row._serverIp ? (
                          <span className={rootClass.elem("server-badge-ip").toClassName()} title={row._serverUrl}>
                            {row._serverIp}
                          </span>
                        ) : (
                          <span style={{ color: "var(--color-neutral-content-subtle)" }}>—</span>
                        )}
                      </td>
                      <td style={{ textAlign: "right" }}>{formatValue(row.request_count, 0)}</td>
                      <td style={{ textAlign: "right" }}>{formatValue(row.rps, 1)}</td>
                      <td style={{ textAlign: "right" }}>{formatValue(row.latency_ms, 1)}</td>
                      <td style={{ textAlign: "right" }}>
                        {formatValue(row.success_count, 0)} / {formatValue(row.error_count, 0)}
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <span
                          className={rootClass.elem("status-badge").toClassName()}
                          style={{ background: badge.background, color: badge.color }}
                        >
                          {badge.label}
                        </span>
                      </td>
                      <td style={{ textAlign: "center" }} onClick={(e) => e.stopPropagation()}>
                        <Button
                          size="small"
                          look="outlined"
                          variant="negative"
                          icon={<IconTrash size={14} />}
                          waiting={isDeleting}
                          disabled={isDeleting}
                          onClick={() => handleDeleteModelFromServer(row.name, row._serverUrl, row._rowKey)}
                          title={
                            row._serverUrl ? `從 ${row._serverIp} 移除模型 ${row.name}` : `完整移除模型 ${row.name}`
                          }
                        >
                          移除
                        </Button>
                      </td>
                    </tr>,

                    /* ── 展開詳情列（單台伺服器詳情） ── */
                    isExpanded ? (
                      <tr key={`${row._rowKey}--detail`} className={rootClass.elem("model-detail-row").toClassName()}>
                        <td />
                        <td colSpan={9}>
                          <div className={rootClass.elem("model-detail").toClassName()}>
                            <div className={rootClass.elem("model-detail-grid").toClassName()}>
                              <span className={rootClass.elem("detail-label").toClassName()}>部署至此台時間</span>
                              <span>{formatDateTime(row._serverDeployedAt || row.deployed_at)}</span>

                              <span className={rootClass.elem("detail-label").toClassName()}>Triton 伺服器</span>
                              <span>
                                {row._serverIp ? (
                                  <span className={rootClass.elem("server-list-item").toClassName()}>
                                    <span className={rootClass.elem("server-badge-ip").toClassName()}>
                                      {row._serverIp}
                                    </span>
                                    <span className={rootClass.elem("detail-mono").toClassName()}>
                                      {row._serverUrl}
                                    </span>
                                  </span>
                                ) : (
                                  <span className={rootClass.elem("detail-mono").toClassName()}>—</span>
                                )}
                              </span>

                              <span className={rootClass.elem("detail-label").toClassName()}>推論端點</span>
                              <span className={rootClass.elem("detail-mono").toClassName()}>
                                {row._serverUrl
                                  ? `${row._serverUrl}/v2/models/${row.name}/infer`
                                  : (row.infer_url ?? "—")}
                              </span>

                              <span className={rootClass.elem("detail-label").toClassName()}>Run ID</span>
                              <span className={rootClass.elem("detail-mono").toClassName()}>{row.run_id ?? "—"}</span>

                              <span className={rootClass.elem("detail-label").toClassName()}>輸入尺寸</span>
                              <span>{row.imgsz ? `${row.imgsz} × ${row.imgsz}` : "—"}</span>
                            </div>

                            {/* ── 版本管理區塊 ── */}
                            {Array.isArray(row.available_versions) && row.available_versions.length > 0 && (
                              <div className={rootClass.elem("version-manager").toClassName()}>
                                <div className={rootClass.elem("version-manager-title").toClassName()}>
                                  版本管理
                                  <span className={rootClass.elem("version-manager-hint").toClassName()}>
                                    （共 {row.available_versions.length} 個版本，最新：v{row.latest_version}）
                                  </span>
                                </div>
                                <div className={rootClass.elem("version-list").toClassName()}>
                                  {row.available_versions.map((ver) => {
                                    const vKey = `${row.name}::${ver}`;
                                    const isLatest = ver === row.latest_version;
                                    const isDeleting = deletingVersionKeys.has(vKey);
                                    return (
                                      <div key={ver} className={rootClass.elem("version-item").toClassName()}>
                                        <span
                                          className={rootClass
                                            .elem("version-badge")
                                            .mod({ latest: isLatest })
                                            .toClassName()}
                                        >
                                          v{ver}
                                          {isLatest && (
                                            <span className={rootClass.elem("version-latest-tag").toClassName()}>
                                              最新
                                            </span>
                                          )}
                                        </span>
                                        <span className={rootClass.elem("version-infer-url").toClassName()}>
                                          {row._serverUrl
                                            ? `${row._serverUrl}/v2/models/${row.name}/versions/${ver}/infer`
                                            : `…/v2/models/${row.name}/versions/${ver}/infer`}
                                        </span>
                                        <Button
                                          size="small"
                                          look="outlined"
                                          variant="negative"
                                          icon={<IconTrash size={12} />}
                                          waiting={isDeleting}
                                          disabled={isDeleting}
                                          onClick={() => handleDeleteVersion(row.name, ver, row._rowKey)}
                                          title={`刪除版本 ${ver}`}
                                        >
                                          刪除版本
                                        </Button>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    ) : null,
                  ];
                })
              ) : (
                <tr>
                  <td colSpan={10} className={rootClass.elem("empty-cell").toClassName()}>
                    目前沒有符合此檢視條件的部署模型。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </SectionBlock>

      <SectionBlock title="最近推論事件" description="顯示近期推論請求的呼叫者、部署者、耗時與成功狀態。">
        <div className={rootClass.elem("table-wrap").toClassName()}>
          <table className={rootClass.elem("table").toClassName()}>
            <thead>
              <tr>
                <th>時間</th>
                <th>模型</th>
                <th>呼叫者</th>
                <th>部署者</th>
                <th>耗時</th>
                <th>狀態</th>
                <th>輸入 / 輸出</th>
              </tr>
            </thead>
            <tbody>
              {events.length > 0 ? (
                events.map((event) => (
                  <tr key={`${event.captured_at}-${event.model_name}-${event.status_code}`}>
                    <td>{formatDateTime(event.captured_at)}</td>
                    <td className={rootClass.elem("table-primary").toClassName()}>{event.model_name ?? "未記錄"}</td>
                    <td>{event.requested_by_username ?? "未記錄"}</td>
                    <td>{event.deployed_by_username ?? "未記錄"}</td>
                    <td>
                      {formatValue(event.duration_ms)} <small>ms</small>
                    </td>
                    <td>
                      <span
                        className={rootClass
                          .elem("status-text")
                          .mod({ success: event.ok, error: !event.ok })
                          .toClassName()}
                      >
                        {event.ok ? "成功" : `失敗 (${event.status_code ?? "-"})`}
                      </span>
                    </td>
                    <td>
                      {formatValue(event?.request?.input_count ?? 0, 0)} /{" "}
                      {formatValue(event?.request?.output_count ?? 0, 0)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className={rootClass.elem("empty-cell").toClassName()}>
                    目前沒有可用的推論事件歷史。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </SectionBlock>

      <SectionBlock title="歷史監控快照" description="追蹤近一段時間的硬體與服務效能變化。">
        <div className={rootClass.elem("table-wrap").toClassName()}>
          <table className={rootClass.elem("table").toClassName()}>
            <thead>
              <tr>
                <th>時間</th>
                <th>CPU</th>
                <th>RAM</th>
                <th>GPU</th>
                <th>VRAM</th>
                <th>RPS</th>
                <th>平均延遲</th>
                <th>成功率</th>
              </tr>
            </thead>
            <tbody>
              {snapshots.length > 0 ? (
                snapshots.map((snapshot) => (
                  <tr key={`${snapshot.captured_at}-${snapshot.scope}`}>
                    <td>{formatDateTime(snapshot.captured_at)}</td>
                    <td>
                      {formatValue(snapshot?.hardware?.cpu)} <small>%</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.hardware?.ram)} <small>%</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.hardware?.gpu)} <small>%</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.hardware?.vram_used_gb)} / {formatValue(snapshot?.hardware?.vram_total_gb)}{" "}
                      <small>GB</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.inference?.rps)} <small>req/s</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.inference?.latency)} <small>ms</small>
                    </td>
                    <td>
                      {formatValue(snapshot?.inference?.success_rate)} <small>%</small>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className={rootClass.elem("empty-cell").toClassName()}>
                    目前沒有可用的監控快照歷史。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </SectionBlock>
    </section>
  );
}
