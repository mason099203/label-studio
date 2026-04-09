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
  const [trainingEngine, setTrainingEngine] = useState("auto"); // auto | yolo_classify | cnn_classify
  const [datasetConfigPath, setDatasetConfigPath] = useState("");
  const [preparingDataset, setPreparingDataset] = useState(false);
  const [exportFormat, setExportFormat] = useState("");
  const [datasetMeta, setDatasetMeta] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobInfo, setJobInfo] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [trainingSpec, setTrainingSpec] = useState(null);

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
        setTrainingSpec(res?.training_spec ?? null);
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
      if (!datasetConfigPath) throw new Error("請輸入 dataset_config.json 路徑");

      const res = await api.callApi("trainingCreateJob", {
        params: { pk: pageParams.id },
        body: {
          base_weights: selectedBaseWeights,
          dataset_config: datasetConfigPath,
          epochs: 50,
          imgsz: trainingEngine === "cnn_classify" ? 224 : 640,
          batch: trainingEngine === "cnn_classify" ? 32 : 16,
          training_model: trainingEngine !== "auto" ? trainingEngine : undefined,
        },
      });

      setJobId(res?.job_id);
    } catch (err) {
      setErrorMessage(err?.message ?? "Training failed");
      setTrainingState("error");
    }
  }, [pageParams?.id, selectedBaseWeights, datasetConfigPath]);

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
          export_format: exportFormat || null,
        },
      });
      setDatasetMeta(meta);
      if (meta?.dataset_config) setDatasetConfigPath(meta.dataset_config);
    } catch (err) {
      setErrorMessage(err?.message ?? "Dataset preparation failed");
    } finally {
      setPreparingDataset(false);
    }
  }, [pageParams?.id, exportFormat]);

  const canStart =
    localModels.length > 0 &&
    (trainingState === "idle" || trainingState === "error") &&
    Boolean(selectedBaseWeights) &&
    Boolean(datasetConfigPath);

  const hasDatasetReady = Boolean(datasetConfigPath);

  const disabledReason = (() => {
    if (trainingState === "running") return "目前已有任務進行中";
    if (trainingState === "done")
      return "本次訓練已完成；若要再次訓練請關閉此視窗後重新開啟 Training。";
    if (localModels.length === 0) return "找不到本機權重檔（請確認 data/training/models/original/）";
    if (!selectedBaseWeights) return "請先選擇 base 模型";
    if (!datasetConfigPath) return "請先輸入或從 Export 產生 dataset_config.json";
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
          const headline = meta?.message ?? "Training failed";
          const detail = meta?.error ? `\n詳情：${meta.error}` : "";
          setErrorMessage(headline + detail);
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
          使用此專案資料進行訓練，並檢視效能指標與輸出模型。
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
              尚未準備訓練資料集，請先「從 Export 產生 dataset_config.json」或手動填入 JSON 設定路徑。
            </div>
          )}
          {(datasetMeta?.task_type || trainingSpec?.task_type) && (
            <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
              任務類型：{datasetMeta?.task_type ?? trainingSpec?.task_type} / 訓練模型：
              {" "}
              {datasetMeta?.training_model ?? trainingSpec?.training_model}
            </div>
          )}
        </div>

        {/* 選擇 base 模型 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>基礎模型</div>
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

        {/* dataset_config.json 路徑 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>資料集設定</div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <select
              className={cn("training-page").elem("input").toClassName()}
              style={{ width: "auto", padding: "4px 8px" }}
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
            >
              <option value="">依專案預設</option>
              <option value="YOLO_WITH_IMAGES">YOLO_WITH_IMAGES (YOLO v8/v11 偵測)</option>
              <option value="JSON_MIN">JSON_MIN (輕量 / 分類資料結構)</option>
              <option value="YOLO">YOLO (僅座標，無圖片)</option>
              <option value="COCO">COCO</option>
            </select>
            <Button
              look="outlined"
              size="small"
              onClick={prepareDatasetFromExport}
              waiting={preparingDataset}
              disabled={!isDefined(pageParams?.id)}
              aria-label="Prepare dataset from export"
            >
              生成訓練資料集
            </Button>
          </div>
          <div className={cn("training-page").elem("stats").toClassName()} style={{ marginTop: 8 }}>
            <input
              className="w-full"
              value={datasetConfigPath}
              placeholder="例如：D:\\ai_test\\project\\label-studio\\data\\training\\datasets\\project_1\\20260313_120000\\dataset_config.json"
              onChange={(e) => setDatasetConfigPath(e.target.value)}
            />
          </div>


          {/* 訓練引擎選擇 (僅限分類任務) */}
          {(datasetMeta?.task_type === "classification" || trainingSpec?.task_type === "classification") && (
            <div style={{ marginTop: 12 }}>
              <div className={cn("training-page").elem("section-title").toClassName()} style={{ fontSize: 13 }}>訓練引擎</div>
              <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input type="radio" name="engine" value="auto" checked={trainingEngine === "auto"} onChange={() => setTrainingEngine("auto")} />
                  自動 (依資料集)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input type="radio" name="engine" value="yolo_classify" checked={trainingEngine === "yolo_classify"} onChange={() => setTrainingEngine("yolo_classify")} />
                  YOLO v8/v11
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input type="radio" name="engine" value="cnn_classify" checked={trainingEngine === "cnn_classify"} onChange={() => setTrainingEngine("cnn_classify")} />
                  ResNet18 (PyTorch)
                </label>
              </div>
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
              {(
                metrics?.top1 != null || metrics?.top5 != null
                  ? [
                      ["Top-1", metrics.top1],
                      ["Top-5", metrics.top5],
                      ["Fitness", metrics.fitness],
                    ]
                  : [
                      ["mAP@0.5", metrics.map50],
                      ["mAP@0.5:0.95", metrics.map],
                      ["mAP@0.75", metrics.map75],
                      ["Precision (mp)", metrics.mp],
                      ["Recall (mr)", metrics.mr],
                    ]
              ).map(([label, value]) => (
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
              <p>訓練完成。</p>
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

        <div className={cn("training-page").elem("footer").toClassName()}>
          <Space spread style={{ width: "100%" }}>
            <span className={cn("training-page").elem("footer-hint").toClassName()}>
              {/* 標註資料將以專案現有標註為訓練集；可先從 Export 匯出備份。 */}
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
