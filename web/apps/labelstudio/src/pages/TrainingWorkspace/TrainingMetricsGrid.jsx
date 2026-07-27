import { cn } from "../../utils/bem";
import { formatMetricPercent, pickSummaryMetrics } from "./trainingRunUtils";

/**
 * 訓練指標網格（Training / Models / Progress 共用視覺）。
 * @param {{ block: string, metrics?: Record<string, unknown> | null, className?: string }} props
 */
export const TrainingMetricsGrid = ({ block, metrics, className }) => {
  const entries = pickSummaryMetrics(metrics).filter(([, value]) => value != null);
  if (!entries.length) return null;

  const root = cn(block);

  return (
    <div className={root.elem("metrics").mix(className).toClassName()}>
      {entries.map(([label, value]) => (
        <div key={label} className={root.elem("metric").toClassName()}>
          <span className={root.elem("metric-label").toClassName()}>{label}</span>
          <span className={root.elem("metric-value").toClassName()}>{formatMetricPercent(value)}</span>
        </div>
      ))}
    </div>
  );
};
