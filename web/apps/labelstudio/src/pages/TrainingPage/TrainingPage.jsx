import { useCallback, useEffect, useState } from "react";
import { useHistory } from "react-router";
import {
  IconAnalytics,
  IconFileDownload,
  IconPlay,
  IconWarningCircleFilled,
} from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { cn } from "../../utils/bem";
import { absoluteURL, isDefined } from "../../utils/helpers";
import "./TrainingPage.scss";

/**
 * 訓練模組頁面：使用當前專案與已標註資料進行模型訓練，
 * 展示訓練效能指標並提供輸出模型下載以供部署。
 *
 * 後續要修改「訓練跑哪裡 / 模型目錄 / 可選模型清單」：
 * - **後端**：`label_studio/training/api.py`（_get_original_models_dir / _get_training_output_root）
 * - **前端**：此檔案的模型選擇與表單欄位
 */
export const TrainingPage = () => {
  const history = useHistory();
  const location = useFixedLocation();
  const pageParams = useParams();
  const api = useAPI();

  const [project, setProject] = useState(null);
  const [taskStats, setTaskStats] = useState({ annotated: 0, total: 0 });
  const [trainingState, setTrainingState] = useState("idle"); // idle | running | done | error
  const [metrics, setMetrics] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [outputModelUrl, setOutputModelUrl] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [localModels, setLocalModels] = useState([]);
  const [selectedBaseWeights, setSelectedBaseWeights] = useState(null);
  const [dataYamlPath, setDataYamlPath] = useState("");
  const [preparingDataset, setPreparingDataset] = useState(false);
  const [datasetMeta, setDatasetMeta] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobInfo, setJobInfo] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);

  const closeAndBack = useCallback(() => {
    const path = location.pathname.replace(TrainingPage.path, "");
    const search = location.search;
    history.replace(`${path}${search !== "?" ? search : ""}`);
  }, [history, location.pathname, location.search]);

  useEffect(() => {
    if (!isDefined(pageParams?.id)) return;
    let cancelled = false;

    api
      .callApi("project", {
        params: { pk: pageParams.id },
        errorFilter: () => true,
      })
      .then((proj) => {
        if (cancelled) return;
        setProject(proj);
        setTaskStats({
          annotated: proj?.num_tasks_with_annotations ?? 0,
          total: proj?.task_number ?? 0,
        });
      });

    api
      .callApi("trainingLocalModels", {
        params: { pk: pageParams.id },
        errorFilter: () => true,
      })
      .then((res) => {
        if (cancelled) return;
        if (!res) {
          setErrorMessage("無法取得本機模型清單（training/models API 回傳空值或請求失敗）");
          return;
        }
        const models = res?.models ?? [];

        setLocalModels(Array.isArray(models) ? models : []);
        if (models?.length) setSelectedBaseWeights(models[0].path);
      });

    api
      .callApi("trainingHistory", {
        params: { pk: pageParams.id },
        errorFilter: () => true,
      })
      .then((res) => {
        if (cancelled) return;
        setTrainingHistory(res ?? null);
      });

    return () => {
      cancelled = true;
    };
  }, [pageParams?.id]);

  /**
   * 啟動訓練（後端 RQ job，在本機 server 進行）。
   */
  const startTraining = useCallback(async () => {
    setErrorMessage(null);
    setTrainingState("running");
    setMetrics(null);
    setArtifacts([]);
    setOutputModelUrl(null);
    setJobId(null);
    setJobInfo(null);

    try {
      if (!pageParams?.id) return;
      if (!selectedBaseWeights) throw new Error("請先選擇 base 模型（權重檔）");
      if (!dataYamlPath) throw new Error("請輸入 data.yaml 路徑");

      const res = await api.callApi("trainingCreateJob", {
        params: { pk: pageParams.id },
        body: {
          base_weights: selectedBaseWeights,
          data_yaml: dataYamlPath,
          epochs: 50,
          imgsz: 640,
          batch: 16,
        },
      });

      setJobId(res?.job_id);
    } catch (err) {
      setErrorMessage(err?.message ?? "Training failed");
      setTrainingState("error");
    }
  }, [pageParams?.id, selectedBaseWeights, dataYamlPath]);

  const prepareDatasetFromExport = useCallback(async () => {
    setErrorMessage(null);
    setPreparingDataset(true);
    setDatasetMeta(null);
    try {
      if (!pageParams?.id) return;
      const meta = await api.callApi("trainingPrepareDataset", {
        params: { pk: pageParams.id },
        body: {
          train_ratio: 0.8,
          seed: 42,
        },
      });
      setDatasetMeta(meta);
      if (meta?.data_yaml) setDataYamlPath(meta.data_yaml);
    } catch (err) {
      setErrorMessage(err?.message ?? "Dataset preparation failed");
    } finally {
      setPreparingDataset(false);
    }
  }, [pageParams?.id]);

  const canStart =
    localModels.length > 0 &&
    trainingState === "idle" &&
    Boolean(selectedBaseWeights) &&
    Boolean(dataYamlPath);

  const hasDatasetReady = Boolean(dataYamlPath);

  const disabledReason = (() => {
    if (trainingState !== "idle") return "目前已有任務進行中";
    if (localModels.length === 0) return "找不到本機權重檔（請確認 data/training/models/original/）";
    if (!selectedBaseWeights) return "請先選擇 base 模型";
    if (!dataYamlPath) return "請先輸入或從 Export 產生 data.yaml";
    return null;
  })();

  // Poll job status while running
  useEffect(() => {
    if (!pageParams?.id || !jobId) return;
    let cancelled = false;
    let timer = null;

    const poll = async () => {
      try {
        const info = await api.callApi("trainingJob", {
          params: { pk: pageParams.id, job_id: jobId },
          errorFilter: () => true,
        });
        if (cancelled) return;

        setJobInfo(info);
        const st = info?.status;
        const meta = info?.meta ?? {};
        const finished = st === "finished" || meta?.status === "finished";
        const failed = st === "failed" || meta?.status === "failed";

        if (finished) {
          setTrainingState("done");
          setMetrics(meta?.metrics ?? null);

          const artifactsRes = await api.callApi("trainingJobArtifacts", {
            params: { pk: pageParams.id, job_id: jobId },
            errorFilter: () => true,
          });
          if (cancelled) return;
          setArtifacts(artifactsRes?.artifacts ?? []);

          const best = meta?.best_path ? "best.pt" : null;
          const last = meta?.last_path ? "last.pt" : null;
          const file = best ?? last;

          if (file) {
            setOutputModelUrl(
              absoluteURL(`/api/projects/${pageParams.id}/training/jobs/${jobId}/download?file=${file}`),
            );
          }
          return;
        }

        if (failed) {
          setTrainingState("error");
          setErrorMessage(meta?.message ?? "Training failed");
          return;
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, 2000);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [pageParams?.id, jobId]);

  return (
    <Modal
      onHide={closeAndBack}
      title="Training"
      style={{ width: 720 }}
      closeOnClickOutside={false}
      allowClose={trainingState !== "running"}
      visible
    >
      <div className={cn("training-page").toClassName()}>
        <div className={cn("training-page").elem("intro").toClassName()}>
          使用此專案資料在本機 server 進行訓練，並檢視效能指標與輸出模型。
        </div>

        {/* 專案與資料摘要 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>專案與資料</div>
          <div className={cn("training-page").elem("stats").toClassName()}>
            <span>專案：{project?.title ?? "—"}</span>
            <span>
              已標註任務：{taskStats.annotated} / {taskStats.total}
            </span>
          </div>
          {!hasDatasetReady && (
            <div className={cn("training-page").elem("warning").toClassName()}>
              <IconWarningCircleFilled />
              尚未準備訓練資料集，請先「從 Export 產生 data.yaml」或手動填入 data.yaml 路徑。
            </div>
          )}
        </div>

        {/* 選擇 base 模型 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>Base 模型（本機）</div>
          {localModels.length === 0 ? (
            <div className={cn("training-page").elem("hint").toClassName()}>
              找不到本機權重檔。請把 `.pt` 放到 `data/training/models/original/`，即可在此選擇。
            </div>
          ) : (
            <div className={cn("training-page").elem("backends").toClassName()}>
              {localModels.map((m) => (
                <div
                  key={m.path}
                  className={cn("training-page")
                    .elem("backend-item")
                    .mod({ selected: selectedBaseWeights === m.path })
                    .toClassName()}
                  onClick={() => setSelectedBaseWeights(m.path)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && setSelectedBaseWeights(m.path)}
                >
                  <span className={cn("training-page").elem("backend-title").toClassName()}>
                    {m.name}
                  </span>
                  {/* <span className={cn("training-page").elem("backend-desc").toClassName()}>{m.path}</span> */}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* data.yaml 路徑 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>資料集設定</div>
          <div className={cn("training-page").elem("hint").toClassName()}>
            你可以直接填入既有 `data.yaml` 路徑，或點下方按鈕從本專案 Export 產生（Label Studio JSON → YOLO_WITH_IMAGES → data.yaml）。
          </div>
          <div style={{ marginTop: 10 }}>
            <Button
              look="outlined"
              size="small"
              onClick={prepareDatasetFromExport}
              waiting={preparingDataset}
              disabled={!isDefined(pageParams?.id)}
              aria-label="Prepare dataset from export"
            >
              從 Export 產生 data.yaml
            </Button>
          </div>
          <div className={cn("training-page").elem("stats").toClassName()} style={{ marginTop: 8 }}>
            <input
              className="w-full"
              value={dataYamlPath}
              placeholder="例如：D:\\ai_test\\project\\label-studio\\data\\training\\datasets\\project_1\\data.yaml"
              onChange={(e) => setDataYamlPath(e.target.value)}
            />
          </div>
          {datasetMeta?.data_yaml && (
            <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
              已產生資料集：{datasetMeta.dataset_root}（train {datasetMeta.train_count} / val {datasetMeta.val_count}）
            </div>
          )}
        </div>

        {/* 訓練按鈕 */}
        <div className={cn("training-page").elem("actions").toClassName()}>
          <Button
            onClick={startTraining}
            disabled={!canStart}
            waiting={trainingState === "running"}
            icon={<IconPlay />}
            aria-label="Start training"
          >
            {trainingState === "running" ? "訓練中…" : "開始訓練"}
          </Button>
        </div>

        {!canStart && disabledReason && (
          <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
            {disabledReason}
          </div>
        )}

        {trainingState === "running" && (
          <div className={cn("training-page").elem("hint").toClassName()}>
            Job：{jobId ?? "建立中…"} {jobInfo?.meta?.message ? `— ${jobInfo.meta.message}` : ""}
          </div>
        )}

        {errorMessage && (
          <div className={cn("training-page").elem("error").toClassName()}>
            <IconWarningCircleFilled /> {errorMessage}
          </div>
        )}

        {/* 訓練效能指標 */}
        {metrics && (
          <div className={cn("training-page").elem("section").toClassName()}>
            <div className={cn("training-page").elem("section-title").toClassName()}>
              <IconAnalytics /> 訓練效能
            </div>
            <div className={cn("training-page").elem("metrics").toClassName()}>
              {[
                ["mAP@0.5", metrics.map50],
                ["mAP@0.5:0.95", metrics.map],
                ["mAP@0.75", metrics.map75],
                ["Precision (mp)", metrics.mp],
                ["Recall (mr)", metrics.mr],
              ].map(([label, value]) => (
                <div key={label} className={cn("training-page").elem("metric").toClassName()}>
                  <span className={cn("training-page").elem("metric-label").toClassName()}>{label}</span>
                  <span className={cn("training-page").elem("metric-value").toClassName()}>
                    {typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—"}
                  </span>
                </div>
              ))}
            </div>
            {metrics.warning && (
              <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
                {metrics.warning}
              </div>
            )}
          </div>
        )}

        {/* 輸出模型下載 / 部署 */}
        {outputModelUrl && (
          <div className={cn("training-page").elem("section").toClassName()}>
            <div className={cn("training-page").elem("section-title").toClassName()}>
              <IconFileDownload /> 輸出模型
            </div>
            <div className={cn("training-page").elem("output").toClassName()}>
              <p>訓練完成，可下載模型用於後續部署或更新 ML Backend。</p>
              <Button
                as="a"
                href={outputModelUrl}
                target="_blank"
                rel="noreferrer"
                icon={<IconFileDownload />}
                look="outlined"
                size="small"
              >
                下載模型
              </Button>
            </div>
          </div>
        )}

        {artifacts?.length > 0 && (
          <div className={cn("training-page").elem("section").toClassName()}>
            <div className={cn("training-page").elem("section-title").toClassName()}>Artifacts</div>
            <div className={cn("training-page").elem("hint").toClassName()}>
              {artifacts.map((a) => (
                <div key={a.download_url}>
                  <a className="no-go" href={absoluteURL(a.download_url)} target="_blank" rel="noreferrer">
                    {a.name}
                  </a>
                </div>
              ))}
            </div>
          </div>
        )}

        {(trainingHistory?.runs?.length || trainingHistory?.datasets?.length) && (
          <div className={cn("training-page").elem("section").toClassName()}>
            <div className={cn("training-page").elem("section-title").toClassName()}>歷史紀錄</div>

            {trainingHistory?.runs?.length > 0 && (
              <div className={cn("training-page").elem("hint").toClassName()} style={{ marginBottom: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>已訓練模型（runs）</div>
                {trainingHistory.runs.slice(0, 10).map((r) => (
                  <div key={r.run_id} style={{ marginBottom: 10 }}>
                    <div>
                      <span style={{ fontFamily: "monospace" }}>{r.run_id}</span>
                      {r.metrics?.map50 != null && (
                        <span style={{ marginLeft: 8, color: "var(--color-neutral-content-subtle)" }}>
                          mAP50: {(r.metrics.map50 * 100).toFixed(2)}%
                        </span>
                      )}
                    </div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 6 }}>
                      <a className="no-go" href={absoluteURL(r.best_download_url)} target="_blank" rel="noreferrer">
                        best.pt
                      </a>
                      <a className="no-go" href={absoluteURL(r.last_download_url)} target="_blank" rel="noreferrer">
                        last.pt
                      </a>
                      {(r.artifacts ?? []).slice(0, 6).map((a) => (
                        <a key={a.download_url} className="no-go" href={absoluteURL(a.download_url)} target="_blank" rel="noreferrer">
                          {a.name}
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {trainingHistory?.datasets?.length > 0 && (
              <div className={cn("training-page").elem("hint").toClassName()}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>已產生資料集（datasets）</div>
                {trainingHistory.datasets.slice(0, 10).map((d) => (
                  <div key={d.dataset_id} style={{ marginBottom: 10 }}>
                    <div>
                      <span style={{ fontFamily: "monospace" }}>{d.dataset_id}</span>
                      {d.meta?.train_count != null && (
                        <span style={{ marginLeft: 8, color: "var(--color-neutral-content-subtle)" }}>
                          train {d.meta.train_count} / val {d.meta.val_count}
                        </span>
                      )}
                    </div>
                    <div style={{ marginTop: 6, color: "var(--color-neutral-content-subtle)" }}>
                      data.yaml: {d.data_yaml ?? "—"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className={cn("training-page").elem("footer").toClassName()}>
          <Space spread style={{ width: "100%" }}>
            <span className={cn("training-page").elem("footer-hint").toClassName()}>
              標註資料將以專案現有標註為訓練集；可先從 Export 匯出備份。
            </span>
            <Button onClick={closeAndBack} look="outlined" size="small" aria-label="Close">
              關閉
            </Button>
          </Space>
        </div>
      </div>
    </Modal>
  );
};

TrainingPage.path = "/training";
TrainingPage.modal = true;
