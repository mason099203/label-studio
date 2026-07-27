import { useState } from "react";
import { useHistory } from "react-router";
import { Button } from "@humansignal/ui";
import { absoluteURL } from "../../utils/helpers";
import { cn } from "../../utils/bem";
import { TrainingMetricsGrid } from "./TrainingMetricsGrid";
import { formatRunStatus, formatTrainingTime, isActiveRun } from "./trainingRunUtils";

const BLOCK = "training-page";

/**
 * 訓練歷史紀錄列表（Training modal 精簡版，與 Models 頁 run-card 視覺一致）。
 * @param {{
 *   projectId: string | number,
 *   runs: Array<Record<string, unknown>>,
 *   limit?: number,
 *   showViewAll?: boolean,
 * }} props
 */
export const TrainingRunHistoryList = ({ projectId, runs, limit = 5, showViewAll = true }) => {
  const history = useHistory();
  const [expandedRunId, setExpandedRunId] = useState(null);
  const root = cn(BLOCK);
  const visibleRuns = runs.slice(0, limit);

  if (!visibleRuns.length) {
    return (
      <div className={root.elem("empty").toClassName()}>
        尚無歷史訓練紀錄。完成一次訓練後會顯示於此。
      </div>
    );
  }

  return (
    <div className={root.elem("run-list").toClassName()}>
      {visibleRuns.map((run) => {
        const isExpanded = expandedRunId === run.run_id;
        const charts = (run.artifacts ?? []).filter((a) => /\.(png|jpg|jpeg|webp)$/i.test(a.name));

        return (
          <div key={run.run_id} className={root.elem("run-card").toClassName()}>
            <button
              type="button"
              className={root.elem("run-summary").toClassName()}
              onClick={() => setExpandedRunId(isExpanded ? null : run.run_id)}
            >
              <div className={root.elem("run-summary-main").toClassName()}>
                <div className={root.elem("run-summary-title").toClassName()}>{run.name || run.run_id}</div>
                <div className={root.elem("hint").toClassName()}>{formatTrainingTime(run.created_at ?? run.ended_at)}</div>
              </div>
              <div className={root.elem("run-summary-side").toClassName()}>
                <div
                  className={root
                    .elem("run-status")
                    .mod({ [run.status ?? "unknown"]: true })
                    .toClassName()}
                >
                  {formatRunStatus(run)}
                </div>
                <div className={root.elem("run-toggle").toClassName()}>{isExpanded ? "收合" : "展開"}</div>
              </div>
            </button>

            {isExpanded && (
              <div className={root.elem("run-content").toClassName()}>
                {run.message && <div className={root.elem("hint").toClassName()}>{run.message}</div>}
                {run.error && <div className={root.elem("run-error").toClassName()}>{run.error}</div>}

                <TrainingMetricsGrid block={BLOCK} metrics={run.metrics} />

                {charts.length > 0 && (
                  <div className={root.elem("charts").toClassName()}>
                    {charts.map((chart) => (
                      <a
                        key={chart.download_url}
                        href={absoluteURL(chart.download_url)}
                        target="_blank"
                        rel="noreferrer"
                        className={root.elem("chart-link").toClassName()}
                      >
                        <img src={absoluteURL(chart.download_url)} alt={chart.name} />
                        <div className={root.elem("chart-name").toClassName()}>{chart.name}</div>
                      </a>
                    ))}
                  </div>
                )}

                <div className={root.elem("run-actions").toClassName()}>
                  {isActiveRun(run) && (
                    <Button
                      look="outlined"
                      size="small"
                      onClick={() =>
                        history.push(`/projects/${projectId}/data/training/progress/${run.run_id}`)
                      }
                    >
                      查看進度
                    </Button>
                  )}
                  <Button
                    look="outlined"
                    size="small"
                    onClick={() => history.push(`/projects/${projectId}/models`)}
                  >
                    在模型紀錄中管理
                  </Button>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {showViewAll && runs.length > limit && (
        <div className={root.elem("hint").toClassName()}>
          另有 {runs.length - limit} 筆紀錄。{" "}
          <button
            type="button"
            className={root.elem("text-link").toClassName()}
            onClick={() => history.push(`/projects/${projectId}/models`)}
          >
            前往模型紀錄查看全部 →
          </button>
        </div>
      )}
    </div>
  );
};
