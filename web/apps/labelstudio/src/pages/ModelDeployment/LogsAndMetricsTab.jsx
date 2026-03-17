import { useState, useEffect, useCallback } from "react";
import { Typography, Button, buttonVariant, Spinner } from "@humansignal/ui";
import { IconExternal, IconTerminal, IconCode, IconInfoOutline, IconAnalytics } from "@humansignal/icons";
import { ToggleItems } from "../../components";
import { cn } from "../../utils/bem";
import { MONITORING_ENDPOINTS } from "./config";
import { useAPI } from "../../providers/ApiProvider";
import { useProject } from "../../providers/ProjectProvider";
import "./ModelDeployment.module.scss";

const rootClass = cn("logs-metrics-tab");
const VIEW_MODES = {
  mine: "我的部署",
  project: "專案全部",
};

/**
 * 數據統計磁貼元件
 */
function StatCard({ label, value, unit, icon: Icon, color = "#6366f1" }) {
  return (
    <div className={rootClass.elem("stat-card").toClassName()}>
      <div className="icon" style={{ color: color, marginBottom: 8 }}>
        {typeof Icon === "function" ? <Icon size={24} /> : <IconInfoOutline size={24} />}
      </div>
      <div className="label">{label}</div>
      <div className="value">
        {value} <small style={{ fontSize: 14, fontWeight: 400 }}>{unit}</small>
      </div>
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

export function LogsAndMetricsTab() {
  const api = useAPI();
  const project = useProject()?.project;
  const [iframeUrl, setIframeUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [scope, setScope] = useState("mine");

  const fetchMetrics = useCallback(async () => {
    if (!project?.id) return;
    try {
      const res = await api.callApi("trainingMetrics", { params: { pk: project.id, scope } });
      setData(res);
    } catch (e) {
      console.error("Failed to fetch metrics", e);
    } finally {
      setLoading(false);
    }
  }, [api, project?.id, scope]);

  useEffect(() => {
    fetchMetrics();
    const timer = setInterval(fetchMetrics, 5000); // 5 sec poll
    return () => clearInterval(timer);
  }, [fetchMetrics]);

  if (loading && !data) return <Spinner size={32} centered />;

  const health = data?.health ?? {};
  const hardware = data?.hardware ?? {};
  const inference = data?.inference ?? {};
  const modelUsage = data?.models ?? [];
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
      <Typography variant="headline" size="medium" className="mb-wide">
        性能指標與監控日誌
      </Typography>

      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", marginBottom: 24, flexWrap: "wrap" }}>
        <div>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            目前檢視：{VIEW_MODES[scope]}{data?.viewer?.username ? ` (${data.viewer.username})` : ""}
          </Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            Triton 服務：{" "}
            <span
              style={{
                padding: "2px 8px",
                borderRadius: 999,
                background: serviceBadge.background,
                color: serviceBadge.color,
                fontWeight: 600,
              }}
            >
              {serviceBadge.label}
            </span>
            {health.metrics_available ? " / Metrics 已連線" : " / Metrics 未連線"}
          </Typography>
        </div>
        <ToggleItems items={VIEW_MODES} active={scope} onSelect={setScope} />
      </div>

      <Typography variant="title" size="medium" className="mb-tight">硬體資源監控</Typography>
      <div className={rootClass.elem("stats-grid").toClassName()} style={{ marginBottom: 32 }}>
        {hardwareStats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      <Typography variant="title" size="medium" className="mb-tight">推論效能指標</Typography>
      <div className={rootClass.elem("stats-grid").toClassName()} style={{ marginBottom: 32 }}>
        {perfStats.map(s => <StatCard key={s.label} {...s} />)}
      </div>

      <Typography variant="title" size="medium" className="mb-tight">個別模型使用情況</Typography>
      <div className={rootClass.elem("usage-table-wrap").toClassName()} style={{
        background: "rgba(255, 255, 255, 0.6)",
        backdropFilter: "blur(10px)",
        padding: 24, borderRadius: 16, border: '1px solid #e2e8f0',
        marginBottom: 32,
        overflowX: "auto",
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 760 }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1" }}>
              <th style={{ padding: "12px 0", color: "#64748b" }}>模型名稱</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>部署者</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>請求總數</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>RPS</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>平均延遲</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>成功 / 失敗</th>
              <th style={{ padding: "12px 0", color: "#64748b" }}>狀態</th>
            </tr>
          </thead>
          <tbody>
            {modelUsage.length > 0 ? modelUsage.map((m) => {
              const badge = getHealthBadge(m.status);

              return (
                <tr key={m.name} style={{ borderBottom: "1px solid #f1f5f9" }}>
                  <td style={{ padding: "16px 0", fontWeight: 500 }}>
                    <div>{m.name}</div>
                    <small style={{ color: "#64748b" }}>{m.owned_by_current_user ? "目前使用者部署" : "其他成員部署"}</small>
                  </td>
                  <td style={{ padding: "16px 0" }}>{m.deployed_by_username ?? "未記錄"}</td>
                  <td style={{ padding: "16px 0" }}>{formatValue(m.request_count, 0)}</td>
                  <td style={{ padding: "16px 0" }}>{formatValue(m.rps)} <small>req/s</small></td>
                  <td style={{ padding: "16px 0" }}>{formatValue(m.latency_ms)} <small>ms</small></td>
                  <td style={{ padding: "16px 0" }}>
                    {formatValue(m.success_count, 0)} / {formatValue(m.error_count, 0)}
                  </td>
                  <td style={{ padding: "16px 0" }}>
                    <span style={{
                      padding: "4px 8px",
                      borderRadius: 4,
                      fontSize: 12,
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
                <td colSpan={7} style={{ padding: "24px 0", color: "#64748b", textAlign: "center" }}>
                  目前沒有符合此檢視條件的部署模型。
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Typography variant="title" size="medium" className="mb-tight">
        監控與分析入口
      </Typography>
      <Typography variant="body" size="small" className="text-neutral-content-subtle mb-wide">
        點擊下方入口進入專業監控後台，檢視完整的分佈式日誌與詳細指標。
      </Typography>

      <div className={rootClass.elem("grid").toClassName()} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
        {MONITORING_ENDPOINTS.map((item) => (
          <div key={item.name} className={rootClass.elem("item-card").toClassName()} style={{ 
            background: 'white', padding: 20, borderRadius: 12, border: '1px solid #e2e8f0',
            display: 'flex', flexDirection: 'column', gap: 12
          }}>
            <div>
              <Typography variant="title" size="small" style={{ marginBottom: 4 }}>
                {item.name}
              </Typography>
              <Typography variant="body" size="small" className="text-neutral-content-subtle">
                {item.description}
              </Typography>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 'auto' }}>
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

      {iframeUrl && (
        <div className={rootClass.elem("iframe-wrap").toClassName()} style={{ marginTop: 32 }}>
          <div className={rootClass.elem("iframe-header").toClassName()} style={{ 
            background: '#f1f5f9', padding: '8px 16px', borderTopLeftRadius: 16, borderTopRightRadius: 16,
            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
          }}>
            <Typography variant="title" size="small">
              {MONITORING_ENDPOINTS.find((e) => e.url === iframeUrl)?.name ?? "預覽模式"}
            </Typography>
            <Button variant="neutral" look="outlined" size="small" onClick={() => setIframeUrl(null)}>
              關閉視窗
            </Button>
          </div>
          <div className={rootClass.elem("iframe-container").toClassName()}>
            <iframe
              title="監控預覽"
              src={iframeUrl}
              style={{ width: '100%', height: '600px', border: 'none' }}
            />
          </div>
        </div>
      )}
    </section>
  );
}
