import { useCallback, useEffect, useRef, useState } from "react";
import { useHistory } from "react-router";
import { IconAnalytics, IconCheck, IconFileDownload, IconPlay, IconWarningCircleFilled } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { cn } from "../../utils/bem";
import { isDefined } from "../../utils/helpers";
import {
  downloadTrainingArtifact,
  getTrainServerBodyFields,
  getTrainServerQueryParams,
  getTrainServerSettings,
  hasTrainServerUrl,
  normalizeTrainServerUrl,
  saveTrainServerDraft,
  setTrainServerSettings,
} from "./trainServerStorage";
import {
  isAllowedTrainingTaskType,
  toUltralyticsTaskKey,
} from "../ModelDeployment/trainingTaskTypes";
import "./TrainingPage.scss";

const TRAINING_UNSUPPORTED_HINT =
  "目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。";

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
  const [outputModelFile, setOutputModelFile] = useState(null);
  const [downloadingOutput, setDownloadingOutput] = useState(false);
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
  const [trainServerMode, setTrainServerMode] = useState("local");
  const [taskDefaults, setTaskDefaults] = useState(null);
  const [epochs, setEpochs] = useState(100);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);
  const [patience, setPatience] = useState(100);
  const [runName, setRunName] = useState("");
  const [showAdvancedParams, setShowAdvancedParams] = useState(false);
  const [optimizer, setOptimizer] = useState("auto");
  const [lr0, setLr0] = useState("");
  const [lrf, setLrf] = useState("");
  const [mosaic, setMosaic] = useState("");
  const [mixup, setMixup] = useState("");
  const [trainServerUrl, setTrainServerUrl] = useState("");
  const [trainServerApiKey, setTrainServerApiKey] = useState("");
  const [trainServerVerified, setTrainServerVerified] = useState(false);
  const [trainServerVerifyDetail, setTrainServerVerifyDetail] = useState(null);
  const [verifyingTrainServer, setVerifyingTrainServer] = useState(false);
  const [unsupportedTraining, setUnsupportedTraining] = useState(false);
  const [unsupportedTrainingDetail, setUnsupportedTrainingDetail] = useState(null);
  const didRestoreTrainServerRef = useRef(false);

  const loadModels = useCallback(async (taskOverride) => {
    if (!pageParams?.id) return;
    const query = getTrainServerQueryParams(pageParams.id);
    const taskKey =
      taskOverride ??
      (trainingSpec?.task_type ? toUltralyticsTaskKey(trainingSpec.task_type) : undefined);
    const res = await api.callApi("trainingLocalModels", {
      params: { pk: pageParams.id, ...query, ...(taskKey ? { task: taskKey } : {}) },
      errorFilter: () => true,
    });
    if (!res) {
      setErrorMessage("無法取得模型清單");
      return;
    }
    const models = res?.models ?? [];
    setLocalModels(Array.isArray(models) ? models : []);
    setTrainingSpec(res?.training_spec ?? null);
    setTrainServerMode(res?.train_server ?? "local");
    setTaskDefaults(res?.task_defaults ?? null);
    if (res?.task_defaults?.epochs) setEpochs(res.task_defaults.epochs);
    if (res?.task_defaults?.imgsz) setImgsz(res.task_defaults.imgsz);
    if (res?.task_defaults?.batch) setBatch(res.task_defaults.batch);
    if (res?.task_defaults?.patience) setPatience(res.task_defaults.patience);
    if (res?.task_defaults?.optimizer) setOptimizer(res.task_defaults.optimizer);
    if (models?.length) {
      const preferred = models.find((m) => m.available) ?? models[0];
      setSelectedBaseWeights(preferred.path ?? preferred.name);
    }
  }, [api, pageParams?.id, trainingSpec?.task_type]);

  const verifyTrainServer = useCallback(async (override = {}) => {
    if (!pageParams?.id) return;
    const urlInput = override.url ?? trainServerUrl;
    const apiKeyInput = override.apiKey ?? trainServerApiKey;
    const normalized = normalizeTrainServerUrl(urlInput);
    if (!normalized) {
      setTrainServerVerified(false);
      setTrainServerVerifyDetail("請輸入 Train Server 位址");
      return;
    }
    setVerifyingTrainServer(true);
    setTrainServerVerifyDetail(null);
    setTrainServerSettings(pageParams.id, {
      url: normalized,
      apiKey: apiKeyInput,
      verified: false,
    });
    try {
      const res = await api.callApi("trainingTrainServerHealth", {
        params: {
          pk: pageParams.id,
          train_server_url: normalized,
          ...(apiKeyInput ? { train_server_api_key: apiKeyInput } : {}),
        },
        errorFilter: () => true,
      });
      const ok = Boolean(res?.ok);
      setTrainServerVerified(ok);
      const detail =
        res?.detail ??
        (ok
          ? `連線成功（${res?.base_url ?? normalized}）`
          : "連線失敗");
      setTrainServerVerifyDetail(detail);
      setTrainServerSettings(pageParams.id, {
        url: normalized,
        apiKey: apiKeyInput,
        verified: ok,
        verifiedAt: ok ? new Date().toISOString() : null,
        detail,
      });
      if (ok) await loadModels();
    } catch (err) {
      setTrainServerVerified(false);
      setTrainServerVerifyDetail(err?.message ?? "驗證失敗");
      saveTrainServerDraft(pageParams.id, { url: normalized, apiKey: apiKeyInput });
    } finally {
      setVerifyingTrainServer(false);
    }
  }, [api, pageParams?.id, trainServerUrl, trainServerApiKey, loadModels]);

  const handleTrainServerUrlChange = useCallback(
    (value) => {
      setTrainServerUrl(value);
      setTrainServerVerified(false);
      setTrainServerVerifyDetail(null);
      if (pageParams?.id) {
        saveTrainServerDraft(pageParams.id, { url: value, apiKey: trainServerApiKey });
      }
    },
    [pageParams?.id, trainServerApiKey],
  );

  const handleTrainServerApiKeyChange = useCallback(
    (value) => {
      setTrainServerApiKey(value);
      setTrainServerVerified(false);
      if (pageParams?.id) {
        saveTrainServerDraft(pageParams.id, { url: trainServerUrl, apiKey: value });
      }
    },
    [pageParams?.id, trainServerUrl],
  );

  const closeAndBack = useCallback(() => {
    const path = location.pathname.replace(TrainingPage.path, "");
    const search = location.search;
    history.replace(`${path}${search !== "?" ? search : ""}`);
  }, [history, location.pathname, location.search]);

  useEffect(() => {
    if (!isDefined(pageParams?.id)) return;
    didRestoreTrainServerRef.current = false;
  }, [pageParams?.id]);

  useEffect(() => {
    if (!isDefined(pageParams?.id) || didRestoreTrainServerRef.current) return;
    didRestoreTrainServerRef.current = true;

    const saved = getTrainServerSettings(pageParams.id);
    setTrainServerUrl(saved.url ?? "");
    setTrainServerApiKey(saved.apiKey ?? "");
    setTrainServerVerified(Boolean(saved.verified));
    setTrainServerVerifyDetail(saved.detail ?? null);
    if (saved.verified && hasTrainServerUrl(saved.url)) {
      verifyTrainServer({ url: saved.url, apiKey: saved.apiKey });
    }
  }, [pageParams?.id, verifyTrainServer]);

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

    loadModels().catch(() => {
      if (!cancelled) setErrorMessage("無法取得模型清單");
    });

    api
      .callApi("trainingInterface", {
        params: { pk: pageParams.id },
        errorFilter: () => true,
      })
      .then((res) => {
        if (cancelled) return;
        if (!res || res?.detail || !isAllowedTrainingTaskType(res.task_type)) {
          setUnsupportedTraining(true);
          setUnsupportedTrainingDetail(
            typeof res?.detail === "string" ? res.detail : TRAINING_UNSUPPORTED_HINT,
          );
          return;
        }
        setUnsupportedTraining(false);
        setUnsupportedTrainingDetail(null);
        setTrainingSpec((prev) => prev ?? res);
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
  }, [pageParams?.id, api, loadModels]);

  /**
   * 啟動訓練（後端 RQ job，在本機 server 進行）。
   */
  const startTraining = useCallback(async () => {
    setErrorMessage(null);
    setTrainingState("running");
    setMetrics(null);
    setArtifacts([]);
    setOutputModelFile(null);
    setJobId(null);
    setJobInfo(null);

    try {
      if (!pageParams?.id) return;
      if (!selectedBaseWeights) throw new Error("請先選擇 base 模型（權重檔）");
      if (!datasetConfigPath) throw new Error("請輸入 dataset_config.json 路徑");
      if (hasTrainServerUrl(trainServerUrl) && !trainServerVerified) {
        throw new Error("請先驗證 Train Server 連線後再開始訓練");
      }

      const extraTrainParams = {};
      if (mosaic !== "") extraTrainParams.mosaic = Number(mosaic);
      if (mixup !== "") extraTrainParams.mixup = Number(mixup);

      const res = await api.callApi("trainingCreateJob", {
        params: { pk: pageParams.id },
        body: {
          base_weights: selectedBaseWeights,
          dataset_config: datasetConfigPath,
          epochs,
          imgsz: trainingEngine === "cnn_classify" ? 224 : imgsz,
          batch: trainingEngine === "cnn_classify" ? 32 : batch,
          patience,
          run_name: runName.trim() || undefined,
          optimizer: optimizer !== "auto" ? optimizer : undefined,
          lr0: lr0 !== "" ? Number(lr0) : undefined,
          lrf: lrf !== "" ? Number(lrf) : undefined,
          train_params: Object.keys(extraTrainParams).length ? extraTrainParams : undefined,
          training_model: trainingEngine !== "auto" ? trainingEngine : undefined,
          ...getTrainServerBodyFields(pageParams.id, {
            url: trainServerUrl,
            apiKey: trainServerApiKey,
            verified: trainServerVerified,
          }),
        },
      });

      setJobId(res?.job_id);
      if (res?.job_id) {
        history.push(`/projects/${pageParams.id}/data/training/progress/${res.job_id}`);
      }
    } catch (err) {
      setErrorMessage(err?.message ?? "Training failed");
      setTrainingState("error");
    }
  }, [
    pageParams?.id,
    selectedBaseWeights,
    datasetConfigPath,
    epochs,
    imgsz,
    batch,
    patience,
    runName,
    optimizer,
    lr0,
    lrf,
    mosaic,
    mixup,
    trainingEngine,
    api,
    history,
    trainServerUrl,
    trainServerApiKey,
    trainServerVerified,
  ]);

  const prepareDatasetFromExport = useCallback(async () => {
    if (!pageParams?.id || unsupportedTraining) return;
    setErrorMessage(null);
    setPreparingDataset(true);
    setDatasetMeta(null);
    try {
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
      if (meta?.task_type) {
        await loadModels(toUltralyticsTaskKey(meta.task_type));
      }
    } catch (err) {
      setErrorMessage(err?.message ?? "Dataset preparation failed");
    } finally {
      setPreparingDataset(false);
    }
  }, [pageParams?.id, exportFormat, api, loadModels, unsupportedTraining]);

  const hasSelectableModel = localModels.some(
    (m) => m.path === selectedBaseWeights || m.name === selectedBaseWeights,
  );

  const customTrainServerUrl = normalizeTrainServerUrl(trainServerUrl);
  const requiresTrainServerVerification = hasTrainServerUrl(customTrainServerUrl);
  const trainServerReady = !requiresTrainServerVerification || trainServerVerified;

  const canStart =
    !unsupportedTraining &&
    hasSelectableModel &&
    (trainingState === "idle" || trainingState === "error") &&
    Boolean(selectedBaseWeights) &&
    Boolean(datasetConfigPath) &&
    trainServerReady;

  const hasDatasetReady = Boolean(datasetConfigPath);

  const disabledReason = (() => {
    if (unsupportedTraining) {
      return unsupportedTrainingDetail || TRAINING_UNSUPPORTED_HINT;
    }
    if (trainingState === "running") return "目前已有任務進行中";
    if (trainingState === "done") return "本次訓練已完成；若要再次訓練請關閉此視窗後重新開啟 Training。";
    if (requiresTrainServerVerification && !trainServerVerified) {
      return "請先輸入 Train Server 位址並點「驗證連線」，通過後才能在此訓練";
    }
    if (localModels.length === 0) return "找不到可用模型（請設定 Train Server 或放置權重於 data/training/models/original/）";
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
        const trainQuery = getTrainServerQueryParams(pageParams.id);
        const info = await api.callApi("trainingJob", {
          params: { pk: pageParams.id, job_id: jobId, ...trainQuery },
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
            params: { pk: pageParams.id, job_id: jobId, ...trainQuery },
            errorFilter: () => true,
          });
          if (cancelled) return;
          setArtifacts(artifactsRes?.artifacts ?? []);

          const best = meta?.best_path ? "best.pt" : null;
          const last = meta?.last_path ? "last.pt" : null;
          const file = best ?? last;

          if (file) {
            setOutputModelFile(file);
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

  const downloadOutputModel = useCallback(async () => {
    if (!pageParams?.id || !jobId || !outputModelFile) return;
    setDownloadingOutput(true);
    try {
      const trainQuery = getTrainServerQueryParams(pageParams.id);
      if (jobInfo?.train_server_url && !trainQuery.train_server_url) {
        trainQuery.train_server_url = jobInfo.train_server_url;
      }
      await downloadTrainingArtifact(api, {
        projectId: pageParams.id,
        jobId,
        file: outputModelFile,
        trainServerParams: trainQuery,
      });
    } catch (err) {
      setErrorMessage(err?.message ?? "下載模型失敗");
    } finally {
      setDownloadingOutput(false);
    }
  }, [api, jobId, jobInfo?.train_server_url, outputModelFile, pageParams?.id]);

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
          {trainServerMode === "remote" && (
            <span className={cn("training-page").elem("hint").toClassName()} style={{ marginLeft: 8 }}>
              （遠端 Train Server — 模型可自動下載）
            </span>
          )}
        </div>

        {/* Train Server 設定 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>Train Server</div>
          <div className={cn("training-page").elem("train-server-row").toClassName()}>
            <input
              className="w-full"
              placeholder="例如：192.168.1.10:8011 或 http://gpu-server:8011"
              value={trainServerUrl}
              onChange={(e) => handleTrainServerUrlChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  verifyTrainServer();
                }
              }}
            />
            <input
              className={cn("training-page").elem("input").toClassName()}
              style={{ width: 160 }}
              placeholder="API Key（可選）"
              type="password"
              value={trainServerApiKey}
              onChange={(e) => handleTrainServerApiKeyChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  verifyTrainServer();
                }
              }}
            />
            <Button
              look="outlined"
              size="small"
              waiting={verifyingTrainServer}
              onClick={() => verifyTrainServer()}
              icon={trainServerVerified ? <IconCheck /> : undefined}
            >
              {trainServerVerified ? "已驗證" : "驗證連線"}
            </Button>
          </div>
          <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
            留空則使用伺服器環境變數 TRAIN_SERVER_URL（本機 RQ 或預設遠端）。
            填寫遠端位址後請先驗證，通過後才能開始訓練。
            {trainServerVerifyDetail ? ` — ${trainServerVerifyDetail}` : ""}
          </div>
          {requiresTrainServerVerification && (
            <div
              className={cn("training-page")
                .elem("train-server-status")
                .mod({
                  ok: trainServerVerified,
                  pending: !trainServerVerified && !verifyingTrainServer,
                })
                .toClassName()}
              style={{ marginTop: 8 }}
            >
              {trainServerVerified
                ? "Train Server 可用，可在此專案進行遠端訓練。"
                : verifyingTrainServer
                  ? "正在驗證 Train Server…"
                  : "尚未驗證 — 請點「驗證連線」確認遠端 Train Server 可連線。"}
            </div>
          )}
          {trainServerMode === "remote" && trainServerVerified && (
            <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 4 }}>
              目前使用遠端 Train Server（模型可自動下載）。
            </div>
          )}
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
              任務類型：{datasetMeta?.task_type ?? trainingSpec?.task_type} / 訓練模型：{" "}
              {datasetMeta?.training_model ?? trainingSpec?.training_model}
            </div>
          )}
        </div>

        {/* 選擇 base 模型 */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>基礎模型</div>
          {localModels.length === 0 ? (
            <div className={cn("training-page").elem("hint").toClassName()}>
              找不到模型清單。請設定 TRAIN_SERVER_URL 或將 `.pt` 放到 `data/training/models/original/`。
            </div>
          ) : (
            <div className={cn("training-page").elem("backends").toClassName()}>
              {localModels.map((m) => (
                <div
                  key={m.path ?? m.name}
                  className={cn("training-page")
                    .elem("backend-item")
                    .mod({ selected: selectedBaseWeights === m.path || selectedBaseWeights === m.name })
                    .toClassName()}
                  onClick={() => setSelectedBaseWeights(m.path ?? m.name)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && setSelectedBaseWeights(m.path ?? m.name)}
                >
                  <span className={cn("training-page").elem("backend-title").toClassName()}>
                    {m.name}
                    {m.family ? ` (${m.family})` : ""}
                    {!m.available && trainServerMode === "remote" ? " · 將自動下載" : ""}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 訓練超參數（對齊 Ultralytics 第 3 步：epochs / batch / imgsz / run name） */}
        <div className={cn("training-page").elem("section").toClassName()}>
          <div className={cn("training-page").elem("section-title").toClassName()}>訓練參數</div>
          <div className={cn("training-page").elem("stats").toClassName()} style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <label>
              Run 名稱
              <input
                type="text"
                className={cn("training-page").elem("input").toClassName()}
                value={runName}
                placeholder="可選"
                onChange={(e) => setRunName(e.target.value)}
                style={{ width: 140, marginLeft: 6 }}
              />
            </label>
            <label>
              Epochs
              <input
                type="number"
                min={1}
                className={cn("training-page").elem("input").toClassName()}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value) || taskDefaults?.epochs || 100)}
                style={{ width: 80, marginLeft: 6 }}
              />
            </label>
            <label>
              Image size
              <input
                type="number"
                min={32}
                step={32}
                className={cn("training-page").elem("input").toClassName()}
                value={imgsz}
                onChange={(e) => setImgsz(Number(e.target.value) || taskDefaults?.imgsz || 640)}
                style={{ width: 80, marginLeft: 6 }}
              />
            </label>
            <label>
              Batch
              <input
                type="number"
                min={1}
                className={cn("training-page").elem("input").toClassName()}
                value={batch}
                onChange={(e) => setBatch(Number(e.target.value) || taskDefaults?.batch || 16)}
                style={{ width: 80, marginLeft: 6 }}
              />
            </label>
            <label>
              Patience
              <input
                type="number"
                min={1}
                className={cn("training-page").elem("input").toClassName()}
                value={patience}
                onChange={(e) => setPatience(Number(e.target.value) || taskDefaults?.patience || 100)}
                style={{ width: 80, marginLeft: 6 }}
              />
            </label>
          </div>
          <button
            type="button"
            className={cn("training-page").elem("hint").toClassName()}
            style={{ marginTop: 8, background: "none", border: "none", cursor: "pointer", padding: 0, textAlign: "left" }}
            onClick={() => setShowAdvancedParams((v) => !v)}
          >
            {showAdvancedParams ? "▾ 收合進階參數" : "▸ 進階參數（學習率、增強等）"}
          </button>
          {showAdvancedParams && (
            <div className={cn("training-page").elem("stats").toClassName()} style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
              <label>
                Optimizer
                <select
                  className={cn("training-page").elem("input").toClassName()}
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value)}
                  style={{ marginLeft: 6 }}
                >
                  {["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"].map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                lr0
                <input
                  type="number"
                  step="0.0001"
                  className={cn("training-page").elem("input").toClassName()}
                  value={lr0}
                  placeholder={taskDefaults?.lr0 ?? "0.01"}
                  onChange={(e) => setLr0(e.target.value)}
                  style={{ width: 90, marginLeft: 6 }}
                />
              </label>
              <label>
                lrf
                <input
                  type="number"
                  step="0.0001"
                  className={cn("training-page").elem("input").toClassName()}
                  value={lrf}
                  placeholder={taskDefaults?.lrf ?? "0.01"}
                  onChange={(e) => setLrf(e.target.value)}
                  style={{ width: 90, marginLeft: 6 }}
                />
              </label>
              <label>
                mosaic
                <input
                  type="number"
                  step="0.1"
                  min={0}
                  max={1}
                  className={cn("training-page").elem("input").toClassName()}
                  value={mosaic}
                  placeholder="預設"
                  onChange={(e) => setMosaic(e.target.value)}
                  style={{ width: 80, marginLeft: 6 }}
                />
              </label>
              <label>
                mixup
                <input
                  type="number"
                  step="0.1"
                  min={0}
                  max={1}
                  className={cn("training-page").elem("input").toClassName()}
                  value={mixup}
                  placeholder="預設"
                  onChange={(e) => setMixup(e.target.value)}
                  style={{ width: 80, marginLeft: 6 }}
                />
              </label>
            </div>
          )}
          {taskDefaults && (
            <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 6 }}>
              預設依 Ultralytics {taskDefaults.task ?? "detect"} 任務。參考{" "}
              <a href="https://docs.ultralytics.com/zh/platform/train/cloud-training" target="_blank" rel="noreferrer">
                官方訓練參數文件
              </a>
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
              <option value="YOLO_WITH_IMAGES">YOLO_WITH_IMAGES（偵測 / 分割 / 姿態）</option>
              <option value="YOLO_OBB_WITH_IMAGES">YOLO_OBB_WITH_IMAGES（旋轉框 OBB）</option>
              <option value="JSON_MIN">JSON_MIN（分類資料結構）</option>
              <option value="YOLO">YOLO（僅座標，無圖片）</option>
              <option value="YOLO_OBB">YOLO_OBB（僅 OBB 座標，無圖片）</option>
              <option value="COCO">COCO</option>
            </select>
            <Button
              look="outlined"
              size="small"
              onClick={prepareDatasetFromExport}
              waiting={preparingDataset}
              disabled={!isDefined(pageParams?.id) || unsupportedTraining || preparingDataset}
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
              <div className={cn("training-page").elem("section-title").toClassName()} style={{ fontSize: 13 }}>
                訓練引擎
              </div>
              <div style={{ display: "flex", gap: 12, marginTop: 4 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input
                    type="radio"
                    name="engine"
                    value="auto"
                    checked={trainingEngine === "auto"}
                    onChange={() => setTrainingEngine("auto")}
                  />
                  自動 (依資料集)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input
                    type="radio"
                    name="engine"
                    value="yolo_classify"
                    checked={trainingEngine === "yolo_classify"}
                    onChange={() => setTrainingEngine("yolo_classify")}
                  />
                  YOLO v8/v11
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input
                    type="radio"
                    name="engine"
                    value="cnn_classify"
                    checked={trainingEngine === "cnn_classify"}
                    onChange={() => setTrainingEngine("cnn_classify")}
                  />
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

        {trainingState === "running" && jobId && (
          <div className={cn("training-page").elem("hint").toClassName()} style={{ marginTop: 8 }}>
            Job：{jobId}{" "}
            <Button
              look="string"
              size="small"
              onClick={() => history.push(`/projects/${pageParams.id}/data/training/progress/${jobId}`)}
            >
              查看訓練進度與即時效果 →
            </Button>
          </div>
        )}

        {trainingState === "running" && !jobId && (
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
              {(metrics?.top1 != null || metrics?.top5 != null
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
        {outputModelFile && (
          <div className={cn("training-page").elem("section").toClassName()}>
            <div className={cn("training-page").elem("section-title").toClassName()}>
              <IconFileDownload /> 輸出模型
            </div>
            <div className={cn("training-page").elem("output").toClassName()}>
              <p>訓練完成。</p>
              <Button
                onClick={downloadOutputModel}
                waiting={downloadingOutput}
                icon={<IconFileDownload />}
                look="outlined"
                size="small"
              >
                下載 {outputModelFile}
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
TrainingPage.exact = true;
TrainingPage.modal = true;
