import { useState, useEffect, useCallback, useMemo } from "react";
import { Typography, Button, buttonVariant, Spinner } from "@humansignal/ui";
import { IconExternal, IconTerminal, IconCode, IconInfoOutline, IconAnalytics, IconTrash, IconChevronRight } from "@humansignal/icons";
import { ToggleItems } from "../../components";
import { Select } from "../../components/Form";
import { cn } from "../../utils/bem";
import { MONITORING_ENDPOINTS } from "./config";
import {
  readTritonUrlState,
  persistTritonUrlFields,
  verifyTritonConnection,
  TRITON_PLAYGROUND_STATE_KEY,
  extractHostFromUrl,
  buildTritonBaseUrl,
  buildTritonMetricsUrl,
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
  /** Triton 伺服器主機 IP（如 `192.168.1.10`）；port 18000/8002 固定。空值表示使用伺服器 TRITON_SERVER_URL。 */
  const [tritonServerHost, setTritonServerHost] = useState(
    () => extractHostFromUrl(readTritonUrlState().tritonServerUrl),
  );
  /** 後端查詢 Triton 健康與轉發之完整 HTTP URL（固定埠 18000）；由 tritonServerHost 自動組成。 */
  const tritonServerUrl = useMemo(() => buildTritonBaseUrl(tritonServerHost), [tritonServerHost]);
  /** Prometheus Metrics 完整 URL（固定埠 8002）；由 tritonServerHost 自動組成。 */
  const tritonMetricsUrl = useMemo(() => buildTritonMetricsUrl(tritonServerHost), [tritonServerHost]);
  const [tritonVerifyLoading, setTritonVerifyLoading] = useState(false);
  const [tritonVerifyResult, setTritonVerifyResult] = useState(null);
  const [iframeUrl, setIframeUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [scope, setScope] = useState("mine");
  const [selectedUserId, setSelectedUserId] = useState(ALL_DEPLOYERS);
  /** @type {[Set<string>, Function]} 目前展開的模型名稱集合 */
  const [expandedModels, setExpandedModels] = useState(new Set());
  /** @type {[string | null, Function]} 正在刪除中的模型名稱 */
  const [deletingModel, setDeletingModel] = useState(null);

  /** 元件掛載時拉取所有專案，供名稱下拉篩選使用。 */
  useEffect(() => {
    setProjectsLoading(true);
    api.callApi("projects").then((res) => {
      const list = Array.isArray(res) ? res : (res?.results ?? []);
      setProjects(list);
    }).catch(() => {
      setProjects([]);
    }).finally(() => {
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

  useEffect(() => {
    persistTritonUrlFields({ tritonServerUrl, tritonMetricsUrl });
  }, [tritonServerUrl, tritonMetricsUrl]);

  useEffect(() => {
    setTritonVerifyResult(null);
  }, [tritonServerUrl]);

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
  const projectOptions = useMemo(() => [
    { label: "— 選擇專案 —", value: "" },
    ...projects.map((p) => ({ label: p.title || `Project ${p.id}`, value: String(p.id) })),
  ], [projects]);

  /**
   * 驗證目前 Triton 基底是否可由後端連線（與載入監控資料相同之權限）。
   */
  const handleVerifyTriton = useCallback(async () => {
    if (!projectId.trim()) {
      setTritonVerifyResult({ level: "error", text: "請先選擇專案。" });
      return;
    }
    setTritonVerifyResult(null);
    setTritonVerifyLoading(true);
    try {
      const r = await verifyTritonConnection(api, projectId, tritonServerUrl);
      setTritonVerifyResult({ level: r.level, text: r.message });
    } finally {
      setTritonVerifyLoading(false);
    }
  }, [api, projectId, tritonServerUrl]);

  const monitoringEndpoints = useMemo(() => {
    const metricsUrl = tritonMetricsUrl.trim() || defaultMetricsUrlFromTritonBase(tritonServerUrl);
    return [
      ...MONITORING_ENDPOINTS,
      {
        name: "Triton Metrics",
        url: metricsUrl,
        description: "Triton 推論指標（Prometheus）",
      },
    ];
  }, [tritonServerUrl, tritonMetricsUrl]);

  const fetchMetrics = useCallback(async () => {
    if (!projectId) {
      setLoading(false);
      setData(null);
      setHistoryData(null);
      return;
    }
    try {
      const params = buildMetricsParams({ pk: projectId, scope, selectedUserId });
      const tritonParams = {};
      const tu = tritonServerUrl.trim();
      if (tu) tritonParams.triton_url = tu;
      const tm = tritonMetricsUrl.trim();
      if (tm) tritonParams.triton_metrics_url = tm;
      const merged = { ...params, ...tritonParams };
      const [metricsRes, metricsHistoryRes] = await Promise.all([
        api.callApi("trainingMetrics", { params: merged }),
        api.callApi("trainingMetricsHistory", {
          params: {
            ...merged,
            event_limit: 12,
            snapshot_limit: 12,
          },
        }),
      ]);
      setData(metricsRes);
      setHistoryData(metricsHistoryRes);
    } catch (e) {
      console.error("Failed to fetch metrics", e);
    } finally {
      setLoading(false);
    }
  }, [api, projectId, scope, selectedUserId, tritonServerUrl, tritonMetricsUrl]);

  /**
   * 切換單一模型列的展開狀態。
   * @param {string} modelName
   */
  const toggleExpand = useCallback((modelName) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelName)) {
        next.delete(modelName);
      } else {
        next.add(modelName);
      }
      return next;
    });
  }, []);

  /**
   * 呼叫後端刪除指定 Triton 部署模型，並重新整理資料。
   * @param {string} modelName - 要刪除的 Triton 模型名稱
   */
  const handleDeleteModel = useCallback(async (modelName) => {
    if (!window.confirm(`確定要移除部署模型「${modelName}」嗎？\n此操作不可復原。`)) return;
    setDeletingModel(modelName);
    try {
      await api.callApi("trainingTritonModelDelete", {
        params: { pk: projectId, model_name: modelName },
      });
      await fetchMetrics();
      setExpandedModels((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
    } catch (err) {
      window.alert(`刪除失敗：${err?.message ?? err}`);
    } finally {
      setDeletingModel(null);
    }
  }, [api, projectId, fetchMetrics]);

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
              <label className={rootClass.elem("field-label").toClassName()}>
                專案
              </label>
              <select
                className={rootClass.elem("text-input").toClassName()}
                value={projectId}
                onChange={(event) => setProjectId(event.target.value)}
                disabled={projectsLoading}
              >
                {projectOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
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

  const health = data?.health ?? {};
  const hardware = data?.hardware ?? {};
  const inference = data?.inference ?? {};
  const modelUsage = data?.models ?? [];
  const deployerOptions = buildDeployerOptions(data?.filters?.deployers);
  const selectedUser = data?.filters?.selected_user;
  const snapshots = historyData?.snapshots ?? [];
  const events = historyData?.events ?? [];
  const serviceBadge = getHealthBadge(health.status);

  /** Triton Prometheus 有無回傳 CPU 指標（nv_cpu_utilization）。 */
  const tritonHasCpuMetrics = hardware.metrics_available && hardware.cpu > 0;
  const hardwareStats = [
    { label: "GPU 使用率", value: formatValue(hardware.gpu), unit: "%", icon: IconAnalytics, color: "#10b981" },
    {
      label: "VRAM 佔用",
      value: formatValue(hardware.vram_used_gb),
      unit: hardware.vram_total_gb > 0
        ? `GB / ${formatValue(hardware.vram_total_gb)} GB`
        : "GB",
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
      value: tritonHasCpuMetrics ? formatValue(hardware.ram) : "—",
      unit: tritonHasCpuMetrics ? `% (${formatValue(hardware.ram_used_gb)} GB)` : "",
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
            <span className={rootClass.elem("meta-chip").toClassName()}>
              {selectedProjectTitle || projectId}
            </span>
            <span className={rootClass.elem("meta-chip").toClassName()}>
              檢視範圍: {VIEW_MODES[scope]}{data?.viewer?.username ? ` (${data.viewer.username})` : ""}
            </span>
            {scope === "project" && selectedUser?.username ? (
              <span className={rootClass.elem("meta-chip").toClassName()}>
                部署者: {selectedUser.username}
              </span>
            ) : null}
            <span
              className={rootClass.elem("service-badge").toClassName()}
              style={{
                background: serviceBadge.background,
                color: serviceBadge.color,
              }}
            >
              Triton {serviceBadge.label}
            </span>
            <span className={rootClass.elem("meta-chip").toClassName()}>
              {health.metrics_available ? "Metrics 已連線" : "Metrics 未連線"}
            </span>
          </div>
        </div>
        <div className={rootClass.elem("toolbar").toClassName()}>
          <div className={rootClass.elem("field").mod({ project: true }).toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()}>
              專案
            </label>
            <select
              className={rootClass.elem("text-input").toClassName()}
              value={projectId}
              onChange={(event) => setProjectId(event.target.value)}
              disabled={projectsLoading}
            >
              {projectOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className={rootClass.elem("field").mod({ project: true }).toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()}>
              Triton 伺服器 IP
            </label>
            <input
              className={rootClass.elem("text-input").toClassName()}
              value={tritonServerHost}
              onChange={(event) => setTritonServerHost(event.target.value.trim())}
              placeholder="192.168.1.10（空＝伺服器預設）"
              type="text"
              title={tritonServerHost.trim() ? `HTTP: ${tritonServerUrl}  Metrics: ${tritonMetricsUrl}` : ""}
            />
          </div>
          <Button variant="neutral" look="outlined" onClick={fetchMetrics} disabled={!projectId}>
            重新載入
          </Button>
          <Button
            variant="neutral"
            look="outlined"
            type="button"
            onClick={handleVerifyTriton}
            disabled={!projectId || tritonVerifyLoading}
          >
            {tritonVerifyLoading ? "驗證中…" : "驗證 Triton"}
          </Button>
          {tritonVerifyResult ? (
            <Typography
              variant="body"
              size="small"
              style={{
                maxWidth: 360,
                color:
                  tritonVerifyResult.level === "success"
                    ? "var(--color-positive-content, #166534)"
                    : tritonVerifyResult.level === "warning"
                      ? "var(--color-warning-content, #92400e)"
                      : "var(--color-negative-content, #991b1b)",
              }}
            >
              {tritonVerifyResult.text}
            </Typography>
          ) : null}
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

      <div className={rootClass.elem("overview").toClassName()}>
        <SectionBlock
          title="Triton 伺服器硬體監控"
          description={
            health.metrics_available
              ? "即時顯示目標 Triton 伺服器的 GPU、VRAM、CPU 與記憶體使用量（資料來源：Triton Prometheus Metrics）。"
              : "Triton Metrics 尚未連線，硬體數據暫不可用。請確認 Triton 伺服器 IP 與 Metrics 埠（8002）是否正確。"
          }
        >
          <div className={rootClass.elem("stats-grid").toClassName()}>
            {hardwareStats.map((s) => <StatCard key={s.label} {...s} />)}
          </div>
        </SectionBlock>

        <SectionBlock title="推論效能指標" description="摘要展示目前模型服務的延遲、吞吐與成功率。">
          <div className={rootClass.elem("stats-grid").toClassName()}>
            {perfStats.map((s) => <StatCard key={s.label} {...s} />)}
          </div>
        </SectionBlock>
      </div>

      <SectionBlock title="部署模型" description="以模型為單位查看部署者、請求數量與目前狀態；點擊列可展開詳細資訊。">
        <div className={rootClass.elem("table-wrap").toClassName()}>
          <table className={rootClass.elem("table").mod({ compact: true }).toClassName()}>
            <thead>
              <tr>
                <th style={{ width: 24 }} />
                <th>模型名稱</th>
                <th>部署者</th>
                <th>請求總數</th>
                <th>RPS</th>
                <th>平均延遲</th>
                <th>成功 / 失敗</th>
                <th>狀態</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {modelUsage.length > 0 ? modelUsage.map((m) => {
                const badge = getHealthBadge(m.status);
                const isExpanded = expandedModels.has(m.name);
                const isDeleting = deletingModel === m.name;

                return [
                  /* ── 主列 ── */
                  <tr
                    key={m.name}
                    className={rootClass.elem("model-row").mod({ expanded: isExpanded }).toClassName()}
                    style={{ cursor: "pointer" }}
                    onClick={() => toggleExpand(m.name)}
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
                      <div className={rootClass.elem("model-name").toClassName()}>{m.name}</div>
                      <small className={rootClass.elem("model-meta").toClassName()}>
                        {m.owned_by_current_user ? "目前使用者部署" : "其他成員部署"}
                      </small>
                    </td>
                    <td>{m.deployed_by_username ?? "未記錄"}</td>
                    <td>{formatValue(m.request_count, 0)}</td>
                    <td>{formatValue(m.rps)} <small>req/s</small></td>
                    <td>{formatValue(m.latency_ms)} <small>ms</small></td>
                    <td>
                      {formatValue(m.success_count, 0)} / {formatValue(m.error_count, 0)}
                    </td>
                    <td>
                      <span
                        className={rootClass.elem("status-badge").toClassName()}
                        style={{ background: badge.background, color: badge.color }}
                      >
                        {badge.label}
                      </span>
                    </td>
                    <td onClick={(e) => e.stopPropagation()}>
                      <Button
                        size="small"
                        look="outlined"
                        variant="negative"
                        icon={<IconTrash size={14} />}
                        waiting={isDeleting}
                        disabled={isDeleting}
                        onClick={() => handleDeleteModel(m.name)}
                        title={`移除部署模型 ${m.name}`}
                      >
                        移除
                      </Button>
                    </td>
                  </tr>,

                  /* ── 展開詳情列 ── */
                  isExpanded ? (
                    <tr key={`${m.name}--detail`} className={rootClass.elem("model-detail-row").toClassName()}>
                      <td />
                      <td colSpan={8}>
                        <div className={rootClass.elem("model-detail").toClassName()}>
                          <div className={rootClass.elem("model-detail-grid").toClassName()}>
                            <span className={rootClass.elem("detail-label").toClassName()}>部署時間</span>
                            <span>{formatDateTime(m.deployed_at)}</span>

                            <span className={rootClass.elem("detail-label").toClassName()}>Run ID</span>
                            <span className={rootClass.elem("detail-mono").toClassName()}>
                              {m.run_id ?? "—"}
                            </span>

                            <span className={rootClass.elem("detail-label").toClassName()}>輸入尺寸</span>
                            <span>{m.imgsz ? `${m.imgsz} × ${m.imgsz}` : "—"}</span>

                            <span className={rootClass.elem("detail-label").toClassName()}>Triton URL</span>
                            <span className={rootClass.elem("detail-mono").toClassName()}>
                              {m.triton_public_base_url ?? m.infer_url ?? "—"}
                            </span>

                            <span className={rootClass.elem("detail-label").toClassName()}>推論端點</span>
                            <span className={rootClass.elem("detail-mono").toClassName()}>
                              {m.infer_url ?? "—"}
                            </span>

                            <span className={rootClass.elem("detail-label").toClassName()}>模型目錄</span>
                            <span className={rootClass.elem("detail-mono").toClassName()}>
                              {m.model_dir ?? "—"}
                            </span>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ) : null,
                ];
              }) : (
                <tr>
                  <td colSpan={9} className={rootClass.elem("empty-cell").toClassName()}>
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
              {events.length > 0 ? events.map((event) => (
                <tr key={`${event.captured_at}-${event.model_name}-${event.status_code}`}>
                  <td>{formatDateTime(event.captured_at)}</td>
                  <td className={rootClass.elem("table-primary").toClassName()}>{event.model_name ?? "未記錄"}</td>
                  <td>{event.requested_by_username ?? "未記錄"}</td>
                  <td>{event.deployed_by_username ?? "未記錄"}</td>
                  <td>{formatValue(event.duration_ms)} <small>ms</small></td>
                  <td>
                    <span
                      className={rootClass.elem("status-text").mod({ success: event.ok, error: !event.ok }).toClassName()}
                    >
                      {event.ok ? "成功" : `失敗 (${event.status_code ?? "-"})`}
                    </span>
                  </td>
                  <td>
                    {formatValue(event?.request?.input_count ?? 0, 0)} / {formatValue(event?.request?.output_count ?? 0, 0)}
                  </td>
                </tr>
              )) : (
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
              {snapshots.length > 0 ? snapshots.map((snapshot) => (
                <tr key={`${snapshot.captured_at}-${snapshot.scope}`}>
                  <td>{formatDateTime(snapshot.captured_at)}</td>
                  <td>{formatValue(snapshot?.hardware?.cpu)} <small>%</small></td>
                  <td>{formatValue(snapshot?.hardware?.ram)} <small>%</small></td>
                  <td>{formatValue(snapshot?.hardware?.gpu)} <small>%</small></td>
                  <td>
                    {formatValue(snapshot?.hardware?.vram_used_gb)} / {formatValue(snapshot?.hardware?.vram_total_gb)} <small>GB</small>
                  </td>
                  <td>{formatValue(snapshot?.inference?.rps)} <small>req/s</small></td>
                  <td>{formatValue(snapshot?.inference?.latency)} <small>ms</small></td>
                  <td>{formatValue(snapshot?.inference?.success_rate)} <small>%</small></td>
                </tr>
              )) : (
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

      <SectionBlock title="監控與分析入口" description="快速進入外部監控工具，查看更完整的圖表與即時資料。">
        <div className={rootClass.elem("grid").toClassName()}>
          {monitoringEndpoints.map((item) => (
            <div key={item.name} className={rootClass.elem("item-card").toClassName()}>
              <div>
                <Typography variant="title" size="small" className={rootClass.elem("item-title").toClassName()}>
                  {item.name}
                </Typography>
                <Typography variant="body" size="small" className="text-neutral-content-subtle">
                  {item.description}
                </Typography>
              </div>
              <div className={rootClass.elem("item-actions").toClassName()}>
                <Button
                  variant="neutral"
                  look="outlined"
                  size="small"
                  onClick={() => setIframeUrl(item.url)}
                >
                  內嵌檢視
                </Button>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={buttonVariant({ variant: "neutral", look: "outlined", size: "small" })}
                >
                  <IconExternal size={14} style={{ marginRight: 4 }} />
                  新分頁
                </a>
              </div>
            </div>
          ))}
        </div>
      </SectionBlock>

      {iframeUrl && (
        <div className={rootClass.elem("iframe-wrap").toClassName()}>
          <div className={rootClass.elem("iframe-header").toClassName()}>
            <Typography variant="title" size="small">
              {monitoringEndpoints.find((e) => e.url === iframeUrl)?.name ?? "預覽模式"}
            </Typography>
            <Button variant="neutral" look="outlined" size="small" onClick={() => setIframeUrl(null)}>
              關閉視窗
            </Button>
          </div>
          <div className={rootClass.elem("iframe-container").toClassName()}>
            <iframe
              className={rootClass.elem("iframe").toClassName()}
              title="監控預覽"
              src={iframeUrl}
            />
          </div>
        </div>
      )}
    </section>
  );
}
