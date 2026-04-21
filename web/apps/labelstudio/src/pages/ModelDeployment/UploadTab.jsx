import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Spinner, Typography } from "@humansignal/ui";
import { IconRefresh, IconTrash, IconUpload } from "@humansignal/icons";
import { TRITON_PLAYGROUND_STATE_KEY } from "./tritonUrlState";
import "./ModelDeployment.scss";

/**
 * Upload Server 固定埠號（對應 docker-compose.yml `upload` service）。
 * @type {number}
 */
const UPLOAD_PORT = 8003;

/** localStorage 鍵：上傳伺服器主機 IP（合併存於同一 playground state 物件）。 */
const UPLOAD_SERVER_HOST_KEY = "uploadServerHost";

/**
 * 從 localStorage 讀取 Triton Upload Server 主機 IP。
 * @returns {string}
 */
function readUploadServerHost() {
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return typeof parsed?.[UPLOAD_SERVER_HOST_KEY] === "string"
      ? parsed[UPLOAD_SERVER_HOST_KEY]
      : "";
  } catch (_) {
    return "";
  }
}

/**
 * 將 Triton Upload Server 主機 IP 寫入 localStorage（合併，不覆蓋其他欄位）。
 * @param {string} host
 */
function persistUploadServerHost(host) {
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    window.localStorage.setItem(
      TRITON_PLAYGROUND_STATE_KEY,
      JSON.stringify({ ...parsed, [UPLOAD_SERVER_HOST_KEY]: host }),
    );
  } catch (_) {
    // 忽略儲存錯誤
  }
}

/**
 * 根據主機 IP 組成完整的 Upload Server Base URL（固定埠 8003）。
 * @param {string} host - 主機 IP 或 hostname，例如 `10.214.57.66`
 * @returns {string} 完整 URL，例如 `http://10.214.57.66:8003`；host 為空時返回空字串
 */
function buildUploadUrl(host) {
  const h = (host || "").trim();
  if (!h) return "";
  // 若已帶 scheme（http/https）則直接用，否則補 http://
  const withScheme = /^https?:\/\//i.test(h) ? h : `http://${h}`;
  return `${withScheme.replace(/\/+$/, "")}:${UPLOAD_PORT}`;
}

/**
 * 模型倉庫上傳分頁：直接透過 Triton Upload Server REST API
 * 上傳 ONNX / TensorRT / PyTorch 模型，並管理現有模型。
 *
 * Upload Server 端點：
 * - `POST  /upload/model?model_name=xxx&version=1`   — 上傳模型主檔
 * - `POST  /upload/config?model_name=xxx`             — 上傳 config.pbtxt
 * - `GET   /models`                                   — 列出所有模型
 * - `DELETE /models/{model_name}`                     — 刪除整個模型目錄
 */
export function UploadTab() {
  /** @type {[string, Function]} 主機 IP，例如 `10.214.57.66` */
  const [uploadServerHost, setUploadServerHost] = useState(() => readUploadServerHost());

  // ── 模型列表 ──────────────────────────────────────────────────────────────
  const [models, setModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelListError, setModelListError] = useState("");

  // ── 上傳表單 ──────────────────────────────────────────────────────────────
  const [uploadModelName, setUploadModelName] = useState("");
  const [uploadVersion, setUploadVersion] = useState(1);
  /** @type {[File|null, Function]} */
  const [modelFile, setModelFile] = useState(null);
  /** @type {[File|null, Function]} */
  const [configFile, setConfigFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState("");

  const modelFileRef = useRef(null);
  const configFileRef = useRef(null);

  // ── 列出模型 ───────────────────────────────────────────────────────────────

  /**
   * 向 Upload Server 的 `GET /models` 取得模型清單。
   * @param {string} host Upload Server 主機 IP
   */
  const fetchModels = useCallback(async (host) => {
    const base = buildUploadUrl(host);
    if (!base) return;
    setLoadingModels(true);
    setModelListError("");
    try {
      const res = await fetch(`${base}/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setModels(Array.isArray(data.models) ? data.models : []);
    } catch (e) {
      setModelListError(`無法連線至上傳伺服器：${e.message}`);
      setModels([]);
    } finally {
      setLoadingModels(false);
    }
  }, []);

  // 首次載入若有已存主機則自動列出
  useEffect(() => {
    if (uploadServerHost) fetchModels(uploadServerHost);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 主機變更 ───────────────────────────────────────────────────────────────

  /**
   * 更新主機 IP 欄位並寫入 localStorage。
   * @param {React.ChangeEvent<HTMLInputElement>} e
   */
  const handleHostChange = (e) => {
    const host = e.target.value;
    setUploadServerHost(host);
    persistUploadServerHost(host);
  };

  // ── 上傳 ───────────────────────────────────────────────────────────────────

  /**
   * 提交表單：依序上傳模型主檔與（選用的）config.pbtxt。
   * @param {React.FormEvent} e
   */
  const handleUpload = async (e) => {
    e.preventDefault();
    const base = buildUploadUrl(uploadServerHost);

    if (!base) {
      setUploadError("請先填入伺服器 IP（例如 10.214.57.66）");
      return;
    }
    if (!modelFile) {
      setUploadError("請選擇要上傳的模型檔案");
      return;
    }
    if (!uploadModelName.trim()) {
      setUploadError("請輸入模型名稱");
      return;
    }

    setUploading(true);
    setUploadError("");
    setUploadResult(null);

    try {
      // 上傳模型主檔
      const modelFormData = new FormData();
      modelFormData.append("file", modelFile);
      const modelRes = await fetch(
        `${base}/upload/model?model_name=${encodeURIComponent(uploadModelName.trim())}&version=${uploadVersion}`,
        { method: "POST", body: modelFormData },
      );
      if (!modelRes.ok) {
        const errBody = await modelRes.json().catch(() => ({}));
        throw new Error(errBody?.detail || `模型上傳失敗：HTTP ${modelRes.status}`);
      }
      const modelResult = await modelRes.json();

      // 選用：上傳 config.pbtxt
      let configResult = null;
      if (configFile) {
        const configFormData = new FormData();
        configFormData.append("file", configFile);
        const configRes = await fetch(
          `${base}/upload/config?model_name=${encodeURIComponent(uploadModelName.trim())}`,
          { method: "POST", body: configFormData },
        );
        if (!configRes.ok) {
          const errBody = await configRes.json().catch(() => ({}));
          throw new Error(errBody?.detail || `config.pbtxt 上傳失敗：HTTP ${configRes.status}`);
        }
        configResult = await configRes.json();
      }

      setUploadResult({ model: modelResult, config: configResult });

      // 重設表單
      setUploadModelName("");
      setUploadVersion(1);
      setModelFile(null);
      setConfigFile(null);
      if (modelFileRef.current) modelFileRef.current.value = "";
      if (configFileRef.current) configFileRef.current.value = "";

      // 重新整理列表
      await fetchModels(base);
    } catch (err) {
      setUploadError(err.message || "未知錯誤");
    } finally {
      setUploading(false);
    }
  };

  // ── 刪除 ───────────────────────────────────────────────────────────────────

  /**
   * 刪除指定模型（整個目錄，包含所有版本）。
   * @param {string} modelName
   */
  const handleDelete = async (modelName) => {
    const base = buildUploadUrl(uploadServerHost);
    if (!base) return;
    if (!window.confirm(`確定要刪除模型「${modelName}」及其所有版本嗎？`)) return;

    try {
      const res = await fetch(`${base}/models/${encodeURIComponent(modelName)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody?.detail || `HTTP ${res.status}`);
      }
      await fetchModels(base);
    } catch (err) {
      alert(`刪除失敗：${err.message}`);
    }
  };

  // ── 渲染 ───────────────────────────────────────────────────────────────────

  return (
    <section className="model-upload-tab">

      {/* ── 連線設定 ── */}
      <div className="model-upload-tab__settings">
        <Typography variant="headline" size="medium">上傳模型至 Triton</Typography>
        <Typography variant="body" size="small" className="text-neutral-content-subtle">
          透過 Triton Upload Server REST API 直接將模型推送到推論伺服器的模型倉庫。
          Upload Server 埠號固定為 <code>{UPLOAD_PORT}</code>。
        </Typography>

        <div className="model-upload-tab__url-row">
          <div className="model-upload-tab__field model-upload-tab__field--url">
            <label className="model-upload-tab__label" htmlFor="upload-server-host">
              伺服器 IP
            </label>
            <input
              id="upload-server-host"
              className="model-upload-tab__input"
              value={uploadServerHost}
              onChange={handleHostChange}
              placeholder="10.214.57.66"
              spellCheck={false}
            />
            {uploadServerHost.trim() && (
              <span className="model-upload-tab__hint">
                有效 URL：<code>{buildUploadUrl(uploadServerHost)}</code>
              </span>
            )}
          </div>
          <Button
            look="outlined"
            icon={<IconRefresh />}
            onClick={() => fetchModels(uploadServerHost)}
            disabled={!uploadServerHost.trim()}
            style={{ alignSelf: "flex-end" }}
          >
            重新整理
          </Button>
        </div>
      </div>

      {/* ── 主要內容區：左上傳表單 + 右模型列表 ── */}
      <div className="model-upload-tab__grid">

        {/* 左：上傳表單 */}
        <div className="model-upload-tab__card">
          <Typography variant="title" size="medium" className="model-upload-tab__card-title">
            上傳新模型
          </Typography>

          <form onSubmit={handleUpload} className="model-upload-tab__form">

            {/* 模型名稱 */}
            <div className="model-upload-tab__field">
              <label className="model-upload-tab__label" htmlFor="model-name">
                模型名稱 <span className="model-upload-tab__required">*</span>
              </label>
              <input
                id="model-name"
                className="model-upload-tab__input"
                value={uploadModelName}
                onChange={(e) => setUploadModelName(e.target.value)}
                placeholder="例如: yolo_v8_detection"
                required
              />
              <span className="model-upload-tab__hint">
                對應 Triton 模型倉庫中的子目錄名稱
              </span>
            </div>

            {/* 版本號 */}
            <div className="model-upload-tab__field model-upload-tab__field--narrow">
              <label className="model-upload-tab__label" htmlFor="model-version">
                版本號
              </label>
              <input
                id="model-version"
                className="model-upload-tab__input"
                type="number"
                min={1}
                value={uploadVersion}
                onChange={(e) => setUploadVersion(Math.max(1, parseInt(e.target.value, 10) || 1))}
              />
            </div>

            {/* 模型主檔 */}
            <div className="model-upload-tab__field">
              <label className="model-upload-tab__label">
                模型檔案 <span className="model-upload-tab__required">*</span>
              </label>
              <div
                className={`model-upload-tab__dropzone${modelFile ? " model-upload-tab__dropzone--has-file" : ""}`}
                onClick={() => modelFileRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && modelFileRef.current?.click()}
              >
                <IconUpload className="model-upload-tab__drop-icon" />
                {modelFile ? (
                  <span className="model-upload-tab__file-name">{modelFile.name}</span>
                ) : (
                  <span>點擊選擇 .onnx / .pt / .pth / .plan</span>
                )}
                <input
                  ref={modelFileRef}
                  type="file"
                  accept=".onnx,.pt,.pth,.plan,.bin"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files?.[0] ?? null;
                    setModelFile(f);
                    // 若模型名稱尚未填寫，自動取檔名（去副檔名）作為預設
                    if (f && !uploadModelName.trim()) {
                      setUploadModelName(f.name.replace(/\.[^.]+$/, ""));
                    }
                  }}
                />
              </div>
            </div>

            {/* config.pbtxt（選用） */}
            <div className="model-upload-tab__field">
              <label className="model-upload-tab__label">
                config.pbtxt <span className="model-upload-tab__optional">（選用）</span>
              </label>
              <div
                className={`model-upload-tab__dropzone model-upload-tab__dropzone--secondary${configFile ? " model-upload-tab__dropzone--has-file" : ""}`}
                onClick={() => configFileRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && configFileRef.current?.click()}
              >
                <IconUpload className="model-upload-tab__drop-icon model-upload-tab__drop-icon--sm" />
                {configFile ? (
                  <span className="model-upload-tab__file-name">{configFile.name}</span>
                ) : (
                  <span>點擊選擇 config.pbtxt</span>
                )}
                <input
                  ref={configFileRef}
                  type="file"
                  accept=".pbtxt,.txt"
                  style={{ display: "none" }}
                  onChange={(e) => setConfigFile(e.target.files?.[0] ?? null)}
                />
              </div>
            </div>

            {/* 上傳結果 / 錯誤訊息 */}
            {uploadError && (
              <div className="model-upload-tab__alert model-upload-tab__alert--error">
                {uploadError}
              </div>
            )}
            {uploadResult && (
              <div className="model-upload-tab__alert model-upload-tab__alert--success">
                <strong>上傳成功！</strong>
                <br />
                模型路徑：<code>{uploadResult.model?.saved_to}</code>
                {uploadResult.config && (
                  <>
                    <br />
                    Config：<code>{uploadResult.config?.saved_to}</code>
                  </>
                )}
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              icon={<IconUpload />}
              waiting={uploading}
              style={{ alignSelf: "flex-start" }}
            >
              {uploading ? "上傳中…" : "開始上傳"}
            </Button>
          </form>
        </div>

        {/* 右：模型倉庫列表 */}
        <div className="model-upload-tab__card">
          <div className="model-upload-tab__card-header">
            <Typography variant="title" size="medium">模型倉庫</Typography>
            {loadingModels && <Spinner size={16} />}
          </div>

          {modelListError && (
            <div className="model-upload-tab__alert model-upload-tab__alert--error">
              {modelListError}
            </div>
          )}

          {!loadingModels && !modelListError && models.length === 0 && (
            <div className="model-upload-tab__empty">
              {uploadServerHost.trim()
                ? "模型倉庫目前為空，請上傳第一個模型。"
                : "請先填入伺服器 IP 並按「重新整理」。"}
            </div>
          )}

          {models.length > 0 && (
            <ul className="model-upload-tab__model-list">
              {models.map((name) => (
                <li key={name} className="model-upload-tab__model-item">
                  <span className="model-upload-tab__model-name">{name}</span>
                  <Button
                    size="small"
                    look="outlined"
                    variant="negative"
                    icon={<IconTrash />}
                    onClick={() => handleDelete(name)}
                    title={`刪除模型 ${name}`}
                  >
                    刪除
                  </Button>
                </li>
              ))}
            </ul>
          )}

          {/* API 端點速查 */}
          <details className="model-upload-tab__api-hint">
            <summary className="model-upload-tab__api-hint-summary">Upload Server API 端點</summary>
            <table className="model-upload-tab__api-table">
              <tbody>
                <tr>
                  <td><code>POST /upload/model</code></td>
                  <td>上傳模型主檔（含版本）</td>
                </tr>
                <tr>
                  <td><code>POST /upload/config</code></td>
                  <td>上傳 config.pbtxt</td>
                </tr>
                <tr>
                  <td><code>GET /models</code></td>
                  <td>列出所有模型</td>
                </tr>
                <tr>
                  <td><code>DELETE /models/{"{name}"}</code></td>
                  <td>刪除模型目錄</td>
                </tr>
                <tr>
                  <td><code>GET /health</code></td>
                  <td>健康檢查</td>
                </tr>
              </tbody>
            </table>
          </details>
        </div>
      </div>
    </section>
  );
}
