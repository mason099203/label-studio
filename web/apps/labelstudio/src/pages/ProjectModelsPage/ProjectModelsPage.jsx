import { useEffect, useState } from "react";
import { IconAnalytics, IconFileDownload, IconWarningCircleFilled } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { useAPI } from "../../providers/ApiProvider";
import { useParams } from "../../providers/RoutesProvider";
import { useProject } from "../../providers/ProjectProvider";
import { cn } from "../../utils/bem";
import { absoluteURL } from "../../utils/helpers";
import "./ProjectModelsPage.scss";

/**
 * Project models page:
 * - View previously trained models without entering labeling
 * - Show run status (running / failed / finished)
 * - Display metrics, error message, and training charts inline
 */
export const ProjectModelsPage = () => {
  const api = useAPI();
  const params = useParams();
  const { project } = useProject();
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);
  const [expandedRunId, setExpandedRunId] = useState(null);

  useEffect(() => {
    if (!params?.id) return;
    api
      .callApi("trainingHistory", {
        params: { pk: params.id },
        errorFilter: () => true,
      })
      .then((res) => setHistory(res ?? null))
      .catch((err) => setError(err?.message ?? "Failed to load training history"));
  }, [params?.id]);

  const runs = history?.runs ?? [];
  const datasets = history?.datasets ?? [];

  /**
   * Format training time for the collapsed run summary.
   * @param {string | null | undefined} value
   * @returns {string}
   */
  const formatTrainingTime = (value) => {
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
  };

  return (
    <div className={cn("project-models-page").toClassName()}>
      <div className={cn("project-models-page").elem("header").toClassName()}>
        <div>
          <h2 className={cn("project-models-page").elem("title").toClassName()}>Models</h2>
          <div className={cn("project-models-page").elem("subtitle").toClassName()}>
            查看先前訓練過的模型、狀態、結果與圖表，不需進入 Labeling。
          </div>
        </div>
        <Button to={`/projects/${params.id}/data/training`} look="outlined" size="small" data-external>
          開啟 Training
        </Button>
      </div>

      {error && (
        <div className={cn("project-models-page").elem("error").toClassName()}>
          <IconWarningCircleFilled /> {error}
        </div>
      )}

      {datasets.length > 0 && (
        <section className={cn("project-models-page").elem("section").toClassName()}>
          <div className={cn("project-models-page").elem("section-title").toClassName()}>Datasets</div>
          <div className={cn("project-models-page").elem("dataset-list").toClassName()}>
            {datasets.slice(0, 10).map((dataset) => (
              <div key={dataset.dataset_id} className={cn("project-models-page").elem("dataset-card").toClassName()}>
                <div className={cn("project-models-page").elem("dataset-id").toClassName()}>{dataset.dataset_id}</div>
                <div className={cn("project-models-page").elem("dataset-meta").toClassName()}>
                  train {dataset.meta?.train_count ?? "—"} / val {dataset.meta?.val_count ?? "—"}
                </div>
                <div className={cn("project-models-page").elem("dataset-path").toClassName()}>
                  {dataset.data_yaml ?? "No data.yaml"}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className={cn("project-models-page").elem("section").toClassName()}>
        <div className={cn("project-models-page").elem("section-title").toClassName()}>Trained Models</div>
        <div className={cn("project-models-page").elem("run-list").toClassName()}>
          {runs.length === 0 && (
            <div className={cn("project-models-page").elem("empty").toClassName()}>尚無歷史模型紀錄。</div>
          )}

          {runs.map((run) => {
            const charts = (run.artifacts ?? []).filter((a) => /\.(png|jpg|jpeg|webp)$/i.test(a.name));
            const isExpanded = expandedRunId === run.run_id;
            const trainingTime = formatTrainingTime(run.created_at ?? run.ended_at);

            return (
              <div key={run.run_id} className={cn("project-models-page").elem("run-card").toClassName()}>
                <button
                  type="button"
                  className={cn("project-models-page").elem("run-summary").toClassName()}
                  onClick={() => setExpandedRunId(isExpanded ? null : run.run_id)}
                >
                  <div className={cn("project-models-page").elem("run-summary-main").toClassName()}>
                    <div className={cn("project-models-page").elem("run-summary-title").toClassName()}>
                      {project?.title ?? `Project ${params.id}`}
                    </div>
                    <div className={cn("project-models-page").elem("run-summary-time").toClassName()}>{trainingTime}</div>
                  </div>
                  <div className={cn("project-models-page").elem("run-summary-side").toClassName()}>
                    <div className={cn("project-models-page").elem("run-status").mod({ [run.status ?? "unknown"]: true }).toClassName()}>
                      {run.status ?? "unknown"}
                    </div>
                    <div className={cn("project-models-page").elem("run-toggle").toClassName()}>
                      {isExpanded ? "收合" : "展開"}
                    </div>
                  </div>
                </button>

                {isExpanded && (
                  <div className={cn("project-models-page").elem("run-content").toClassName()}>
                    <div className={cn("project-models-page").elem("run-top").toClassName()}>
                      <div>
                        <div className={cn("project-models-page").elem("run-id").toClassName()}>{run.run_id}</div>
                      </div>
                      <div className={cn("project-models-page").elem("run-actions").toClassName()}>
                        <a className="no-go" href={absoluteURL(run.best_download_url)} target="_blank" rel="noreferrer">
                          <IconFileDownload /> best.pt
                        </a>
                        <a className="no-go" href={absoluteURL(run.last_download_url)} target="_blank" rel="noreferrer">
                          <IconFileDownload /> last.pt
                        </a>
                      </div>
                    </div>

                    <div className={cn("project-models-page").elem("run-description").toClassName()}>
                      {run.message && <div>{run.message}</div>}
                      {run.error && <div className={cn("project-models-page").elem("run-error").toClassName()}>{run.error}</div>}
                      {run.params?.base_weights && <div>Base: {run.params.base_weights}</div>}
                      {run.params?.data_yaml && <div>Dataset: {run.params.data_yaml}</div>}
                    </div>

                    {run.metrics && (
                      <div className={cn("project-models-page").elem("metrics").toClassName()}>
                        {[
                          ["mAP@0.5", run.metrics.map50],
                          ["mAP@0.5:0.95", run.metrics.map],
                          ["mAP@0.75", run.metrics.map75],
                          ["Precision", run.metrics.mp],
                          ["Recall", run.metrics.mr],
                        ].map(([label, value]) => (
                          <div key={label} className={cn("project-models-page").elem("metric").toClassName()}>
                            <span>{label}</span>
                            <strong>{typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—"}</strong>
                          </div>
                        ))}
                      </div>
                    )}

                    {charts.length > 0 && (
                      <div className={cn("project-models-page").elem("charts").toClassName()}>
                        {charts.map((chart) => (
                          <a
                            key={chart.download_url}
                            href={absoluteURL(chart.download_url)}
                            target="_blank"
                            rel="noreferrer"
                            className={cn("project-models-page").elem("chart-link").toClassName()}
                          >
                            <img src={absoluteURL(chart.download_url)} alt={chart.name} />
                            <div className={cn("project-models-page").elem("chart-name").toClassName()}>
                              <IconAnalytics /> {chart.name}
                            </div>
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};

ProjectModelsPage.path = "/models";
ProjectModelsPage.title = "Models";

