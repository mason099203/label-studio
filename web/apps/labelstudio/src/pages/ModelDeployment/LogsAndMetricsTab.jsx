import { useState, useEffect, useCallback } from "react";
import { Typography, Button, buttonVariant, Spinner } from "@humansignal/ui";
import { IconExternal, IconTerminal, IconCode, IconInfoOutline, IconAnalytics } from "@humansignal/icons";
import { ToggleItems } from "../../components";
import { Select } from "../../components/Form";
import { cn } from "../../utils/bem";
import { MONITORING_ENDPOINTS } from "./config";
import { useAPI } from "../../providers/ApiProvider";
import { useProject } from "../../providers/ProjectProvider";
import "./ModelDeployment.scss";

const rootClass = cn("logs-metrics-tab");
const VIEW_MODES = {
  mine: "我的部署",
  project: "專案全部",
};
const ALL_DEPLOYERS = "__all__";
const PLAYGROUND_STATE_KEY = "labelstudio.triton.playground";

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
    const raw = window.localStorage.getItem(PLAYGROUND_STATE_KEY);
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
    const raw = window.localStorage.getItem(PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    window.localStorage.setItem(
      PLAYGROUND_STATE_KEY,
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
  const [iframeUrl, setIframeUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [scope, setScope] = useState("mine");
  const [selectedUserId, setSelectedUserId] = useState(ALL_DEPLOYERS);

  useEffect(() => {
    if (project?.id) {
      setProjectId(String(project.id));
    }
  }, [project?.id]);

  useEffect(() => {
    persistProjectId(projectId);
  }, [projectId]);

  const fetchMetrics = useCallback(async () => {
    if (!projectId) {
      setLoading(false);
      setData(null);
      setHistoryData(null);
      return;
    }
    try {
      const params = buildMetricsParams({ pk: projectId, scope, selectedUserId });
      const [metricsRes, metricsHistoryRes] = await Promise.all([
        api.callApi("trainingMetrics", { params }),
        api.callApi("trainingMetricsHistory", {
          params: {
            ...params,
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
  }, [api, projectId, scope, selectedUserId]);

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
            這個頁面是全域頁面，若目前不在專案內，請先輸入要查詢的專案 ID。
          </Typography>
          <div className={rootClass.elem("project-form").toClassName()}>
            <div className={rootClass.elem("field").toClassName()}>
              <label className={rootClass.elem("field-label").toClassName()}>
                專案 ID
              </label>
              <input
                className={rootClass.elem("text-input").toClassName()}
                value={projectId}
                onChange={(event) => setProjectId(event.target.value.replace(/[^\d]/g, ""))}
                placeholder="例如 1"
              />
            </div>
            <Button variant="primary" onClick={fetchMetrics} disabled={!projectId}>
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

  const hardwareStats = [
    { label: "GPU 使用率", value: formatValue(hardware.gpu), unit: "%", icon: IconAnalytics, color: "#10b981" },
    {
      label: "VRAM 佔用",
      value: formatValue(hardware.vram_used_gb),
      unit: `GB / ${formatValue(hardware.vram_total_gb)}`,
      icon: IconTerminal,
      color: "#6366f1",
    },
    { label: "CPU 使用率", value: formatValue(hardware.cpu), unit: "%", icon: IconCode, color: "#f59e0b" },
    {
      label: "系統記憶體",
      value: formatValue(hardware.ram),
      unit: `% (${formatValue(hardware.ram_used_gb)}GB)`,
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
            日誌與效能
          </Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            即時查看部署模型、推論事件、效能快照與監控入口。
          </Typography>
          <div className={rootClass.elem("hero-meta").toClassName()}>
            <span className={rootClass.elem("meta-chip").toClassName()}>
              專案 ID: {projectId}
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
              專案 ID
            </label>
            <input
              className={rootClass.elem("text-input").toClassName()}
              value={projectId}
              onChange={(event) => setProjectId(event.target.value.replace(/[^\d]/g, ""))}
              placeholder="例如 1"
            />
          </div>
          <Button variant="neutral" look="outlined" onClick={fetchMetrics} disabled={!projectId}>
            重新載入
          </Button>
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
        <SectionBlock title="硬體資源監控" description="即時顯示目前 Triton 服務所在主機的資源使用情況。">
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

      <SectionBlock title="個別模型使用情況" description="以模型為單位查看部署者、請求數量與目前狀態。">
        <div className={rootClass.elem("table-wrap").toClassName()}>
          <table className={rootClass.elem("table").mod({ compact: true }).toClassName()}>
            <thead>
              <tr>
                <th>模型名稱</th>
                <th>部署者</th>
                <th>請求總數</th>
                <th>RPS</th>
                <th>平均延遲</th>
                <th>成功 / 失敗</th>
                <th>狀態</th>
              </tr>
            </thead>
            <tbody>
              {modelUsage.length > 0 ? modelUsage.map((m) => {
                const badge = getHealthBadge(m.status);

                return (
                  <tr key={m.name}>
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
                        style={{
                          background: badge.background,
                          color: badge.color,
                        }}
                      >
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                );
              }) : (
                <tr>
                  <td colSpan={7} className={rootClass.elem("empty-cell").toClassName()}>
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
          {MONITORING_ENDPOINTS.map((item) => (
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
              {MONITORING_ENDPOINTS.find((e) => e.url === iframeUrl)?.name ?? "預覽模式"}
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
