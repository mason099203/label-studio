import { useEffect, useState } from "react";
import { IconAnalytics, IconFileDownload, IconWarningCircleFilled, IconPencil, IconCheck, IconClose, IconTrash } from "@humansignal/icons";
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
  normalizeTritonUrl,
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
  /** @type {[string|null, Function]} 正在刪除中的 run_id */
  const [deletingRunId, setDeletingRunId] = useState(null);
  /**
   * 部署目標 Triton HTTP 基底（與模型測試／儀錶板共用 localStorage）。
   * 空字串表示後端使用 TRITON_SERVER_URL 環境變數。
   */
  const [deployTritonServerUrl, setDeployTritonServerUrl] = useState(() => readTritonUrlState().tritonServerUrl);
  /** 部署用 Triton 快速選項：default／本機 8000／自訂（與右側網址欄同步）。 */
  const [tritonPreset, setTritonPreset] = useState(() => {
    const u = (readTritonUrlState().tritonServerUrl || "").trim();
    if (!u) return "default";
    if (u === "http://localhost:18000" || u === "http://127.0.0.1:18000") return "local18000";
    return "custom";
  });
  const [tritonVerifyLoading, setTritonVerifyLoading] = useState(false);
  const [tritonVerifyResult, setTritonVerifyResult] = useState(null);

  /** 目前正在編輯名稱的 run_id（null 表示未進入編輯模式） */
  const [editingRunId, setEditingRunId] = useState(null);
  /** 編輯中的暫存名稱 */
  const [editingName, setEditingName] = useState("");

  /** 推論裝置種類："GPU" | "CPU" | "AUTO" */
  const [instanceKind, setInstanceKind] = useState("GPU");
  /** GPU 裝置 ID 字串，逗號分隔，例如 "0" 或 "0,1"。僅 instanceKind==="GPU" 時有效 */
  const [gpuIds, setGpuIds] = useState("0");
  /** 推論實例數量（Triton instance_group.count） */
  const [instanceCount, setInstanceCount] = useState(1);
  /**
   * 記憶體模式：
   * - true  → 常駐記憶體（config.pbtxt 加入 model_warmup，Triton 啟動時即預熱並保持載入）
   * - false → 即時載入（無 model_warmup，模型於首次推論請求時才完整初始化）
   */
  const [alwaysInMemory, setAlwaysInMemory] = useState(true);
  /**
   * 部署目標 Triton 版本號（對應模型倉庫中的版本子目錄 1/ 2/ 3/...）。
   * 同一模型名稱的多次訓練可透過遞增此值寫入新版本，原版本保留不覆蓋。
   */
  const [targetVersion, setTargetVersion] = useState(1);
  /**
   * 自動遞增版本：true 時部署前由後端查詢現有最新版本並 +1，
   * 不需要使用者手動維護版本號。
   */
  const [autoVersion, setAutoVersion] = useState(true);
  /**
   * 上次成功部署的版本號，部署成功後由後端回傳並顯示於 UI。
   * @type {[number | null, Function]}
   */
  const [lastDeployedVersion, setLastDeployedVersion] = useState(null);
  /** 是否展開自訂 config.pbtxt 編輯區 */
  const [showCustomPbtxt, setShowCustomPbtxt] = useState(false);
  /** 自訂 config.pbtxt 內容；空字串表示使用自動生成 */
  const [customPbtxt, setCustomPbtxt] = useState("");

  /**
   * 根據當前部署設定，在前端生成 config.pbtxt 預覽文字（與後端 _build_triton_pbtxt 邏輯一致）。
   * 用於「自訂 config.pbtxt」區塊的初始填入值。
   * @param {string} [modelName] - Triton 模型名稱（預覽用途可用佔位符）
   * @returns {string} config.pbtxt 內容字串
   */
  function generatePbtxtPreview(modelName = "<model_name>") {
    const kindMap = { GPU: "KIND_GPU", CPU: "KIND_CPU", AUTO: "KIND_AUTO" };
    const kindStr = kindMap[instanceKind] || "KIND_AUTO";
    const gpuLine =
      kindStr === "KIND_GPU"
        ? `\n    gpus: [${gpuIds
            .split(",")
            .map((s) => s.trim())
            .filter((s) => /^\d+$/.test(s))
            .join(", ")}]`
        : "";
    const count = Math.max(1, instanceCount);
    const imgsz = instanceKind === "AUTO" ? 640 : 640;
    const instanceGroupBlock = `instance_group [\n  {\n    kind: ${kindStr}\n    count: ${count}${gpuLine}\n  }\n]`;
    const warmupBlock = alwaysInMemory
      ? `model_warmup [\n  {\n    name: "warmup"\n    batch_size: 1\n    inputs {\n      key: "images"\n      value {\n        dims: 3\n        dims: ${imgsz}\n        dims: ${imgsz}\n        data_type: TYPE_FP32\n        zero_data: true\n      }\n    }\n  }\n]`
      : "";
    return [
      `name: "${modelName}"`,
      `platform: "pytorch_libtorch"`,
      `max_batch_size: 1`,
      ``,
      `input [`,
      `  {`,
      `    name: "images"`,
      `    data_type: TYPE_FP32`,
      `    dims: [3, ${imgsz}, ${imgsz}]`,
      `  }`,
      `]`,
      ``,
      `output [`,
      `  {`,
      `    name: "output0"`,
      `    data_type: TYPE_FP32`,
      `    dims: [-1, -1]`,
      `  }`,
      `]`,
      ``,
      instanceGroupBlock,
      warmupBlock ? `\n${warmupBlock}` : "",
    ]
      .filter((line) => line !== undefined)
      .join("\n");
  }

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
   * 開始編輯指定 run 的名稱。
   * @param {React.MouseEvent} e
   * @param {{ run_id: string, name?: string }} run
   */
  function startEditRunName(e, run) {
    e.stopPropagation();
    setEditingRunId(run.run_id);
    setEditingName(run.name || "");
  }

  /**
   * 取消編輯，還原為原名稱。
   * @param {React.MouseEvent} e
   */
  function cancelEditRunName(e) {
    e.stopPropagation();
    setEditingRunId(null);
    setEditingName("");
  }

  /**
   * 儲存 run 名稱：呼叫 PATCH API 寫入 run_meta.json，並樂觀更新本地 history state。
   * @param {React.MouseEvent | React.KeyboardEvent} e
   * @param {string} runId
   */
  async function saveRunName(e, runId) {
    e.stopPropagation();
    const trimmed = editingName.trim();
    try {
      await api.callApi("trainingRunRename", {
        params: { pk: params.id, run_id: runId },
        body: { name: trimmed },
      });
      setHistory((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          runs: prev.runs.map((r) => (r.run_id === runId ? { ...r, name: trimmed } : r)),
        };
      });
    } catch {
      /* 失敗時不更動 state，讓舊名稱保留 */
    }
    setEditingRunId(null);
    setEditingName("");
  }

  /**
   * 由網址推斷預設選項（供手動輸入後同步下拉）。
   * @param {string} url
   * @returns {"default" | "local8000" | "custom"}
   */
  /**
   * 由網址推斷預設選項（供手動輸入後同步下拉）。
   * @param {string} url
   * @returns {"default" | "local18000" | "custom"}
   */
  const presetFromUrl = (url) => {
    const u = (url || "").trim();
    if (!u) return "default";
    if (u === "http://localhost:18000" || u === "http://127.0.0.1:18000") return "local18000";
    return "custom";
  };

  /**
   * 變更部署用 Triton 位址並寫入與模型測試頁相同之儲存。
   * 若 URL 未包含埠號，自動補上預設 TRITON_HTTP_PORT（18000）。
   * @param {string} nextUrl
   */
  const commitDeployTritonUrl = (nextUrl) => {
    const normalized = normalizeTritonUrl(nextUrl);
    setDeployTritonServerUrl(normalized);
    setTritonPreset(presetFromUrl(normalized));
    persistTritonUrlFields({ tritonServerUrl: normalized });
  };

  useEffect(() => {
    setTritonVerifyResult(null);
  }, [deployTritonServerUrl]);

  /**
   * 手動驗證目前選定之 Triton 是否可由後端連線。
   * 驗證前先正規化 URL（補足缺少的埠號），並更新欄位顯示。
   */
  const handleVerifyDeployTriton = async () => {
    if (!params?.id) return;
    // 先正規化：補足使用者未輸入的埠號後再驗證
    const normalized = normalizeTritonUrl(deployTritonServerUrl);
    if (normalized !== deployTritonServerUrl) {
      commitDeployTritonUrl(normalized);
    }
    setTritonVerifyResult(null);
    setTritonVerifyLoading(true);
    try {
      const r = await verifyTritonConnection(api, params.id, normalized);
      setTritonVerifyResult({ level: r.level, text: r.message });
    } finally {
      setTritonVerifyLoading(false);
    }
  };

  /**
   * 刪除指定訓練紀錄（呼叫後端 DELETE API 移除整個 run 目錄）。
   * 刪除成功後樂觀更新本地 history state，不重新請求整份清單。
   * @param {React.MouseEvent} e
   * @param {string} runId
   * @param {string} runName - 顯示用名稱（用於確認對話框）
   */
  const handleDeleteRun = async (e, runId, runName) => {
    e.stopPropagation();
    if (!params?.id || !runId) return;

    const label = runName || runId;
    if (!window.confirm(`確定要刪除訓練紀錄「${label}」嗎？\n此操作將移除所有相關檔案（模型、圖表、紀錄），且不可復原。`)) return;

    setDeletingRunId(runId);
    try {
      const res = await api.callApi("trainingRunDelete", {
        params: { pk: params.id, run_id: runId },
        errorFilter: () => true,
      });
      if (res?.detail && !res?.deleted) {
        window.alert(`刪除失敗：${res.detail}`);
        return;
      }
      // 樂觀移除本地清單，不重送整份 history
      setHistory((prev) => {
        if (!prev) return prev;
        return { ...prev, runs: prev.runs.filter((r) => r.run_id !== runId) };
      });
      if (expandedRunId === runId) setExpandedRunId(null);
    } catch (err) {
      window.alert(`刪除失敗：${err?.message || "未知錯誤"}`);
    } finally {
      setDeletingRunId(null);
    }
  };

  const handleDeployToTriton = async (runId) => {
    if (!params?.id || !runId) return;
    setDeployError(null);
    // 部署前正規化 URL：補足缺少的埠號
    const raw = (deployTritonServerUrl || "").trim();
    const tu = raw ? normalizeTritonUrl(raw) : "";
    if (tu !== raw) {
      commitDeployTritonUrl(tu);
    }
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
    body.instance_kind = instanceKind;
    body.instance_count = instanceCount;
    body.always_in_memory = alwaysInMemory;
    if (autoVersion) {
      // 自動遞增模式：後端查詢現有最新版本 +1，不傳 target_version
      body.auto_version = true;
    } else {
      body.target_version = Math.max(1, targetVersion);
    }
    // 若有自訂 pbtxt 則傳送，後端將直接使用，跳過自動生成
    if (showCustomPbtxt && customPbtxt.trim()) {
      body.custom_pbtxt = customPbtxt.trim();
    }
    if (instanceKind === "GPU") {
      body.gpu_ids = gpuIds
        .split(",")
        .map((s) => s.trim())
        .filter((s) => /^\d+$/.test(s))
        .map(Number);
      if (body.gpu_ids.length === 0) body.gpu_ids = [0];
    }
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
        // 記錄實際寫入的版本號並更新版本輸入框
        const deployedVer = res.deployed_version ?? null;
        if (deployedVer) {
          setLastDeployedVersion(deployedVer);
          if (!autoVersion) setTargetVersion(deployedVer);
        }
        saveTritonPlaygroundState(params.id, res.model_name);
        const verLabel = deployedVer ? `（版本 ${deployedVer}）` : "";
        if (res.model_name) {
          // eslint-disable-next-line no-alert
          window.alert(`已部署至 Triton：${res.model_name} ${verLabel}\n可前往 Playground 直接測試。`);
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
                else if (v === "local18000") commitDeployTritonUrl("http://localhost:18000");
                else setTritonPreset("custom");
              }}
            >
              <option value="default">後端環境預設（不指定 URL）</option>
              <option value="local18000">本機 Triton（localhost:18000）</option>
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
              onBlur={() => commitDeployTritonUrl(deployTritonServerUrl)}
              placeholder="http://主機:18000"
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

          {/* GPU / 推論裝置設定 */}
          <div className={cn("project-models-page").elem("triton-deploy-label").toClassName()}>
            推論裝置（instance_group）
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-row").toClassName()}>
            <select
              className={cn("project-models-page").elem("triton-deploy-select").toClassName()}
              aria-label="推論裝置種類"
              value={instanceKind}
              onChange={(e) => setInstanceKind(e.target.value)}
            >
              <option value="GPU">GPU（KIND_GPU）</option>
              <option value="CPU">CPU（KIND_CPU）</option>
              <option value="AUTO">自動（KIND_AUTO，Triton 決定）</option>
            </select>
            <label
              className={cn("project-models-page").elem("triton-deploy-inline-label").toClassName()}
              htmlFor="triton-instance-count"
            >
              實例數
            </label>
            <input
              id="triton-instance-count"
              type="number"
              min={1}
              max={16}
              className={cn("project-models-page").elem("triton-deploy-count-input").toClassName()}
              value={instanceCount}
              onChange={(e) => setInstanceCount(Math.max(1, parseInt(e.target.value, 10) || 1))}
              aria-label="推論實例數量"
            />
          </div>
          {instanceKind === "GPU" && (
            <div className={cn("project-models-page").elem("triton-deploy-row").toClassName()}>
              <label
                className={cn("project-models-page").elem("triton-deploy-inline-label").toClassName()}
                htmlFor="triton-gpu-ids"
              >
                GPU ID（逗號分隔）
              </label>
              <input
                id="triton-gpu-ids"
                type="text"
                className={cn("project-models-page").elem("triton-deploy-input").toClassName()}
                value={gpuIds}
                onChange={(e) => setGpuIds(e.target.value)}
                placeholder="0 或 0,1"
                aria-label="GPU 裝置 ID，逗號分隔"
              />
            </div>
          )}

          {/* 記憶體常駐模式 */}
          <div className={cn("project-models-page").elem("triton-deploy-label").toClassName()}>
            記憶體載入策略
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-row").toClassName()}>
            <select
              className={cn("project-models-page").elem("triton-deploy-select").toClassName()}
              aria-label="記憶體載入策略"
              value={alwaysInMemory ? "persistent" : "lazy"}
              onChange={(e) => setAlwaysInMemory(e.target.value === "persistent")}
            >
              <option value="persistent">常駐記憶體（啟動時預熱，低延遲）</option>
              <option value="lazy">即時載入（首次推論時初始化，省記憶體）</option>
            </select>
          </div>


          {/* Triton 版本號 */}
          <div className={cn("project-models-page").elem("triton-deploy-label").toClassName()}>
            模型版本號（Triton version）
          </div>
          <div className={cn("project-models-page").elem("triton-deploy-row").toClassName()}>
            {/* 自動遞增開關 */}
            <label
              className={cn("project-models-page").elem("triton-auto-version-label").toClassName()}
              htmlFor="triton-auto-version"
            >
              <input
                id="triton-auto-version"
                type="checkbox"
                checked={autoVersion}
                onChange={(e) => setAutoVersion(e.target.checked)}
                className={cn("project-models-page").elem("triton-auto-version-checkbox").toClassName()}
              />
              自動遞增版本
            </label>
            {!autoVersion && (
              <>
                <label
                  className={cn("project-models-page").elem("triton-deploy-inline-label").toClassName()}
                  htmlFor="triton-target-version"
                >
                  指定版本
                </label>
                <input
                  id="triton-target-version"
                  type="number"
                  min={1}
                  className={cn("project-models-page").elem("triton-deploy-count-input").toClassName()}
                  value={targetVersion}
                  onChange={(e) => setTargetVersion(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  aria-label="部署目標版本號"
                />
              </>
            )}
          </div>
          {lastDeployedVersion && (
            <div className={cn("project-models-page").elem("triton-last-version-hint").toClassName()}>
              上次部署版本：<strong>v{lastDeployedVersion}</strong>
              {autoVersion && (
                <span>，下次將自動寫入 v{lastDeployedVersion + 1}</span>
              )}
            </div>
          )}

          {/* 自訂 config.pbtxt */}
          <div className={cn("project-models-page").elem("triton-deploy-pbtxt-toggle-row").toClassName()}>
            <button
              type="button"
              className={cn("project-models-page").elem("triton-deploy-pbtxt-toggle").toClassName()}
              onClick={() => {
                const next = !showCustomPbtxt;
                setShowCustomPbtxt(next);
                // 展開時若內容為空，自動填入當前設定的預覽
                if (next && !customPbtxt.trim()) {
                  setCustomPbtxt(generatePbtxtPreview());
                }
              }}
            >
              {showCustomPbtxt ? "▼ 隱藏自訂 config.pbtxt" : "▶ 自訂 config.pbtxt（進階）"}
            </button>
            {showCustomPbtxt && (
              <button
                type="button"
                className={cn("project-models-page").elem("triton-deploy-pbtxt-reset").toClassName()}
                onClick={() => setCustomPbtxt(generatePbtxtPreview())}
                title="根據當前設定重新生成預覽"
              >
                ↺ 重設為自動生成
              </button>
            )}
          </div>
          {showCustomPbtxt && (
            <div className={cn("project-models-page").elem("triton-deploy-pbtxt-area-wrap").toClassName()}>
              <p className={cn("project-models-page").elem("triton-deploy-pbtxt-hint").toClassName()}>
                直接編輯以下 config.pbtxt 後再部署；留空則使用上方設定自動生成。
              </p>
              <textarea
                className={cn("project-models-page").elem("triton-deploy-pbtxt-textarea").toClassName()}
                value={customPbtxt}
                onChange={(e) => setCustomPbtxt(e.target.value)}
                rows={20}
                spellCheck={false}
                aria-label="自訂 config.pbtxt 內容"
                placeholder={generatePbtxtPreview()}
              />
            </div>
          )}

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
                      {run.name || run.run_id}
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
                      <div className={cn("project-models-page").elem("run-name-row").toClassName()}>
                        {editingRunId === run.run_id ? (
                          <div className={cn("project-models-page").elem("run-name-edit").toClassName()}>
                            <input
                              className={cn("project-models-page").elem("run-name-input").toClassName()}
                              type="text"
                              value={editingName}
                              placeholder={run.run_id}
                              autoFocus
                              onClick={(e) => e.stopPropagation()}
                              onChange={(e) => setEditingName(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") saveRunName(e, run.run_id);
                                if (e.key === "Escape") cancelEditRunName(e);
                              }}
                            />
                            <button
                              type="button"
                              className={cn("project-models-page").elem("run-name-action").toClassName()}
                              title="儲存名稱"
                              onClick={(e) => saveRunName(e, run.run_id)}
                            >
                              <IconCheck />
                            </button>
                            <button
                              type="button"
                              className={cn("project-models-page").elem("run-name-action").toClassName()}
                              title="取消"
                              onClick={cancelEditRunName}
                            >
                              <IconClose />
                            </button>
                          </div>
                        ) : (
                          <div className={cn("project-models-page").elem("run-name-display").toClassName()}>
                            <div className={cn("project-models-page").elem("run-id").toClassName()}>
                              {run.name || run.run_id}
                            </div>
                            <button
                              type="button"
                              className={cn("project-models-page").elem("run-name-edit-btn").toClassName()}
                              title="編輯名稱"
                              onClick={(e) => startEditRunName(e, run)}
                            >
                              <IconPencil />
                            </button>
                          </div>
                        )}
                        {run.name && (
                          <div className={cn("project-models-page").elem("run-uuid").toClassName()}>
                            {run.run_id}
                          </div>
                        )}
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
                        <Button
                          look="outlined"
                          size="small"
                          variant="negative"
                          icon={<IconTrash size={14} />}
                          onClick={(e) => handleDeleteRun(e, run.run_id, run.name || run.run_id)}
                          disabled={deletingRunId === run.run_id}
                          aria-label="刪除此訓練紀錄"
                          title="刪除此訓練紀錄（不可復原）"
                        >
                          {deletingRunId === run.run_id ? "刪除中…" : "刪除"}
                        </Button>
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

