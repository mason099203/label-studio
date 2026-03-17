import { useState } from "react";
import { Typography, Button, buttonVariant } from "@humansignal/ui";
import { IconExternal, IconTerminal, IconCode, IconInfoOutline } from "@humansignal/icons";
import { cn } from "../../utils/bem";
import { MONITORING_ENDPOINTS } from "./config";
import "./ModelDeployment.module.scss";

const rootClass = cn("logs-metrics-tab");

/**
 * 數據統計磁貼元件
 */
function StatCard({ label, value, unit, icon: Icon }) {
  return (
    <div className={rootClass.elem("stat-card").toClassName()}>
      <div className="icon" style={{ color: "#6366f1", marginBottom: 8 }}>
        {typeof Icon === "function" ? <Icon size={24} /> : <IconInfoOutline size={24} />}
      </div>
      <div className="label">{label}</div>
      <div className="value">
        {value} <small style={{ fontSize: 14, fontWeight: 400 }}>{unit}</small>
      </div>
    </div>
  );
}

export function LogsAndMetricsTab() {
  const [iframeUrl, setIframeUrl] = useState(null);

  // 模擬數據：實際應從 Triton Metrics 或 Prometheus 獲取
  const stats = [
    { label: "平均延遲 (Lat)", value: "42.5", unit: "ms", icon: IconTerminal },
    { label: "每秒請求 (RPS)", value: "128", unit: "req/s", icon: IconExternal },
    { label: "GPU 使用率", value: "65", unit: "%", icon: IconCode },
    { label: "VRAM 佔用", value: "4.2", unit: "GB", icon: IconTerminal },
  ];

  return (
    <section className={rootClass.toClassName()}>
      <Typography variant="headline" size="medium" className="mb-wide">
        性能指標與監控日誌
      </Typography>

      {/* 數據概覽磁貼 */}
      <div className={rootClass.elem("stats-grid").toClassName()}>
        {stats.map(s => <StatCard key={s.label} {...s} />)}
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
