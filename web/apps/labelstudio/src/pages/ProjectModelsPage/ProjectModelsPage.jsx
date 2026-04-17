import { useEffect, useState } from "react";
import { IconAnalytics, IconFileDownload, IconWarningCircleFilled } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { Modal } from "../../components/Modal/Modal";
import { useAPI } from "../../providers/ApiProvider";
import { useParams } from "../../providers/RoutesProvider";
import { useProject } from "../../providers/ProjectProvider";
import { cn } from "../../utils/bem";
import { absoluteURL } from "../../utils/helpers";
import {
  readTritonUrlState,
  persistTritonUrlFields,
  verifyTritonConnection,
  TRITON_PLAYGROUND_STATE_KEY,
} from "../ModelDeployment/tritonUrlState";
import "./ProjectModelsPage.scss";

/**
 * 將最新部署的 Triton 模型寫入 Playground 預設值。
 * @param {string | number} projectId
 * @param {string} modelName
 */
function saveTritonPlaygroundState(projectId, modelName) {
  if (typeof window === "undefined") return;
  let existing = {};
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    existing = raw ? JSON.parse(raw) : {};
  } catch {
    existing = {};
  }
  window.localStorage.setItem(
    TRITON_PLAYGROUND_STATE_KEY,
    JSON.stringify({
      ...existing,
      projectId: String(projectId),
      modelName,
    }),
  );
}

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
  const [selectedChart, setSelectedChart] = useState(null);
  const [deployingRunId, setDeployingRunId] = useState(null);
  const [deployError, setDeployError] = useState(null);
  /**
   * 部署目標 Triton HTTP 基底（與模型測試／儀錶板共用 localStorage）。
   * 空字串表示後端使用 TRITON_SERVER_URL 環境變數。
   */
  const [deployTritonServerUrl, setDeployTritonServerUrl] = useState(() => readTritonUrlState().tritonServerUrl);
  /** 部署用 Triton 快速選項：default／本機 8000／自訂（與右側網址欄同步）。 */
  const [tritonPreset, setTritonPreset] = useState(() => {
    const u = (readTritonUrlState().tritonServerUrl || "").trim();
    if (!u) return "default";
    if (u === "http://localhost:8000" || u === "http://127.0.0.1:8000") return "local8000";
    return "custom";
  });
  const [tritonVerifyLoading, setTritonVerifyLoading] = useState(false);
  const [tritonVerifyResult, setTritonVerifyResult] = useState(null);

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
   * 由網址推斷預設選項（供手動輸入後同步下拉）。
   * @param {string} url
   * @returns {"default" | "local8000" | "custom"}
   */
  const presetFromUrl = (url) => {
    const u = (url || "").trim();
    if (!u) return "default";
    if (u === "http://localhost:8000" || u === "http://127.0.0.1:8000") return "local8000";
    return "custom";
  };

  /**
   * 變更部署用 Triton 位址並寫入與模型測試頁相同之儲存。
   * @param {string} nextUrl
   */
  const commitDeployTritonUrl = (nextUrl) => {
    setDeployTritonServerUrl(nextUrl);
    setTritonPreset(presetFromUrl(nextUrl));
    persistTritonUrlFields({ tritonServerUrl: nextUrl });
  };

  useEffect(() => {
    setTritonVerifyResult(null);
  }, [deployTritonServerUrl]);

  /**
   * 手動驗證目前選定之 Triton 是否可由後端連線。
   */
  const handleVerifyDeployTriton = async () => {
    if (!params?.id) return;
    setTritonVerifyResult(null);
    setTritonVerifyLoading(true);
    try {
      const r = await verifyTritonConnection(api, params.id, deployTritonServerUrl);
      setTritonVerifyResult({ level: r.level, text: r.message });
    } finally {
      setTritonVerifyLoading(false);
    }
  };

  const handleDeployToTriton = async (runId) => {
    if (!params?.id || !runId) return;
    setDeployError(null);
    const tu = (deployTritonServerUrl || "").trim();
    if (tu) {
      setTritonVerifyLoading(true);
      const check = await verifyTritonConnection(api, params.id, tu);
      setTritonVerifyLoading(false);
      setTritonVerifyResult({ level: check.level, text: check.message });
      // degraded（live=true，ready=false）仍允許部署；只有完全無法連線才阻擋
      if (check.level === "error") {
        setDeployError(`Triton 無法連線，請確認伺服器狀態：${check.message}`);
        return;
      }
    }
    setDeployingRunId(runId);
    const body = {};
    if (tu) body.triton_url = tu;
    api
      .callApi("trainingRunDeployToTriton", {
        params: { pk: params.id, run_id: runId },
        body,
        errorFilter: () => true,
      })
      .then((res) => {
        setDeployingRunId(null);
        if (!res || res.error || res.detail) {
          setDeployError(res?.detail || res?.error || "部署至 Triton 失敗");
          return;
        }
        saveTritonPlaygroundState(params.id, res.model_name);
        // 簡單提示成功與模型名稱
        if (res.model_name) {
          // eslint-disable-next-line no-alert
          window.alert(`已部署至 Triton：${res.model_name}\n可前往 Playground 直接測試。`);
        } else {
          // eslint-disable-next-line no-alert
          window.alert("已部署至 Triton。");
        }
      })
      .catch((err) => {
        setDeployingRunId(null);
        setDeployError(err?.message || "部署至 Triton 失敗");
      });
  };

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
          <h2 className={cn("project-models-page").elem("title").toClassName()}>模型紀錄</h2>
          <div className={cn("project-models-page").elem("subtitle").toClassName()}>
            查看先前訓練過的模型、狀態、結果與圖表。
          </div>
        </div>
      </div>

      {error && (
        <div className={cn("project-models-page").elem("error").toClassName()}>
          <IconWarningCircleFilled /> {error}
        </div>
      )}

      {deployError && (
        <div className={cn("project-models-page").elem("error").toClassName()}>
          <IconWarningCircleFilled /> {deployError}
        </div>
      )}

      {/* {datasets.length > 0 && (
        <section className={cn("project-models-page").elem("section").toClassName()}>
          <div className={cn("project-models-page").elem("section-title").toClassName()}>資料集</div>
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
      )} */}

      <section className={cn("project-models-page").elem("section").toClassName()}>
        <div className={cn("project-models-page").elem("section-title").toClassName()}>模型訓練紀錄</div>
        <div className={cn("project-models-page").elem("triton-deploy-config").toClassName()}>
          <div className={cn("project-models-page").elem("triton-deploy-label").toClassName()}>
            部署目標 Triton 伺服器
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-row").toClassName()}>
            <select
              className={cn("project-models-page").elem("triton-deploy-select").toClassName()}
              aria-label="選擇 Triton 伺服器預設"
              value={tritonPreset}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "default") commitDeployTritonUrl("");
                else if (v === "local8000") commitDeployTritonUrl("http://localhost:8000");
                else setTritonPreset("custom");
              }}
            >
              <option value="default">後端環境預設（不指定 URL）</option>
              <option value="local8000">本機 Triton（localhost:8000）</option>
              <option value="custom">自訂網址…</option>
            </select>
            <input
              type="url"
              className={cn("project-models-page").elem("triton-deploy-input").toClassName()}
              value={deployTritonServerUrl}
              onChange={(e) => {
                const next = e.target.value;
                setDeployTritonServerUrl(next);
                setTritonPreset(presetFromUrl(next));
              }}
              onBlur={() => persistTritonUrlFields({ tritonServerUrl: deployTritonServerUrl })}
              placeholder="http://主機:8000"
              aria-label="Triton HTTP 基底網址"
            />
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-actions").toClassName()}>
            <Button
              type="button"
              look="outlined"
              size="small"
              disabled={tritonVerifyLoading}
              onClick={handleVerifyDeployTriton}
            >
              {tritonVerifyLoading ? "驗證中…" : "驗證連線"}
            </Button>
            {tritonVerifyResult ? (
              <span
                className={cn("project-models-page")
                  .elem("triton-verify-msg")
                  .mod({
                    success: tritonVerifyResult.level === "success",
                    warning: tritonVerifyResult.level === "warning",
                    error: tritonVerifyResult.level === "error",
                  })
                  .toClassName()}
              >
                {tritonVerifyResult.text}
              </span>
            ) : null}
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-hint").toClassName()}>
            選擇預設或手動輸入；部署時會將此前綴寫入模型後設，並供模型測試／儀錶板轉發使用。指定 URL 時會先經後端連線檢查再部署。空值表示由伺服器{' '}
            <code>TRITON_SERVER_URL</code> 決定。
          </div>
        </div>
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
                    {/* <div className={cn("project-models-page").elem("run-summary-time").toClassName()}>{trainingTime}</div> */}
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
                          <IconFileDownload /> 模型
                        </a>
                        {/* <a className="no-go" href={absoluteURL(run.last_download_url)} target="_blank" rel="noreferrer">
                          <IconFileDownload /> last.pt
                        </a> */}
                        {run.status === "finished" && (
                          <Button
                            look="outlined"
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeployToTriton(run.run_id);
                            }}
                            disabled={tritonVerifyLoading || deployingRunId === run.run_id}
                            aria-label="部署模型"
                          >
                            {deployingRunId === run.run_id ? "部署中…" : "部署模型"}
                          </Button>
                        )}
                      </div>
                    </div>

                    <div className={cn("project-models-page").elem("run-description").toClassName()}>
                      {run.message && <div>{run.message}</div>}
                      {run.error && <div className={cn("project-models-page").elem("run-error").toClassName()}>{run.error}</div>}
                      {/* {run.params?.base_weights && <div>Base: {run.params.base_weights}</div>} */}
                      {/* {run.params?.data_yaml && <div>Dataset: {run.params.data_yaml}</div>} */}
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
                          <button
                            key={chart.download_url}
                            type="button"
                            onClick={() => setSelectedChart(chart)}
                            className={cn("project-models-page").elem("chart-link").toClassName()}
                          >
                            <img src={absoluteURL(chart.download_url)} alt={chart.name} />
                            <div className={cn("project-models-page").elem("chart-name").toClassName()}>
                              <IconAnalytics /> {chart.name}
                            </div>
                          </button>
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

      {selectedChart && (
        <Modal
          visible
          title={selectedChart.name}
          style={{ width: "min(96vw, 1200px)" }}
          onHide={() => setSelectedChart(null)}
        >
          <div className={cn("project-models-page").elem("preview").toClassName()}>
            <div className={cn("project-models-page").elem("preview-actions").toClassName()}>
              <a
                className={cn("project-models-page").elem("download-button").toClassName()}
                href={absoluteURL(selectedChart.download_url)}
                download={selectedChart.name}
                target="_blank"
                rel="noreferrer"
              >
                <IconFileDownload /> 下載圖片
              </a>
            </div>
            <div className={cn("project-models-page").elem("preview-image-wrap").toClassName()}>
              <img
                className={cn("project-models-page").elem("preview-image").toClassName()}
                src={absoluteURL(selectedChart.download_url)}
                alt={selectedChart.name}
              />
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

ProjectModelsPage.path = "/models";
ProjectModelsPage.title = "Models";

