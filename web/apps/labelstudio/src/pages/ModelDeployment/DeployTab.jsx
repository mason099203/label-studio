import { useCallback, useEffect, useState, useRef } from "react";
import { Button, Typography, Spinner } from "@humansignal/ui";
import { IconCopy, IconPlus, IconUpload } from "@humansignal/icons";
import { useAPI } from "../../providers/ApiProvider";
import { cn } from "../../utils/bem";
import { Modal } from "../../components/Modal/Modal";
import { useProject } from "../../providers/ProjectProvider";
import { Toggle } from "../../components/Form";
import "./ModelDeployment.scss";

/**
 * 通用數字步進器，支援自訂最小值、最大值與步長。
 * @param {{
 *   value: number,
 *   onChange: (n: number) => void,
 *   min?: number,
 *   max?: number,
 *   step?: number,
 *   disabled?: boolean,
 *   ariaLabel?: string,
 * }} props
 * @returns {JSX.Element}
 */
function NumericStepper({
  value,
  onChange,
  min = 1,
  max = Number.POSITIVE_INFINITY,
  step = 1,
  disabled = false,
  ariaLabel = "數值",
}) {
  const handleDec = () => onChange(Math.max(min, value - step));
  const handleInc = () => onChange(max === Number.POSITIVE_INFINITY ? value + step : Math.min(max, value + step));

  return (
    <div className={rootClass.elem("instance-stepper").toClassName()}>
      <button
        type="button"
        className={rootClass.elem("stepper-btn").toClassName()}
        onClick={handleDec}
        disabled={disabled || value <= min}
        aria-label={`減少${ariaLabel}`}
      >
        −
      </button>
      <span className={rootClass.elem("stepper-value").toClassName()}>{value}</span>
      <button
        type="button"
        className={rootClass.elem("stepper-btn").toClassName()}
        onClick={handleInc}
        disabled={disabled || value >= max}
        aria-label={`增加${ariaLabel}`}
      >
        +
      </button>
    </div>
  );
}

/**
 * GPU 實例數量步進器（1–10），薄包裝 NumericStepper。
 * @param {{ value: number, onChange: (n: number) => void, disabled?: boolean }} props
 * @returns {JSX.Element}
 */
function InstanceGroupStepper({ value, onChange, disabled = false }) {
  return (
    <NumericStepper value={value} onChange={onChange} min={1} max={10} disabled={disabled} ariaLabel="GPU 實例數" />
  );
}

/**
 * Model Version 步進器（最小 1，無上限），薄包裝 NumericStepper。
 * @param {{ value: number, onChange: (n: number) => void, disabled?: boolean }} props
 * @returns {JSX.Element}
 */
function ModelVersionStepper({ value, onChange, disabled = false }) {
  return <NumericStepper value={value} onChange={onChange} min={1} disabled={disabled} ariaLabel="模型版本" />;
}

const rootClass = cn("model-deploy-tab");

/**
 * Premium 磁貼卡片：顯示模型資訊與操作
 */
function ModelCard({ title, subtitle, status, children, footer, type = "deployed", onToggle, isEnabled, updating }) {
  return (
    <div
      className={rootClass
        .elem("card")
        .mod({ type, disabled: !isEnabled && type === "deployed" })
        .toClassName()}
    >
      <div className={rootClass.elem("card-header").toClassName()}>
        <div>
          <Typography variant="title" size="medium">
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body" size="small" className="text-neutral-content-subtle">
              {subtitle}
            </Typography>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {onToggle && (
            <Toggle checked={isEnabled} onChange={onToggle} disabled={updating} style={{ transform: "scale(0.8)" }} />
          )}
          <span
            className={rootClass
              .elem("status")
              .mod({ [status.toLowerCase()]: true })
              .toClassName()}
          >
            {status}
          </span>
        </div>
      </div>
      <div className={rootClass.elem("card-body").toClassName()}>{children}</div>
      {footer && <div className={rootClass.elem("actions").toClassName()}>{footer}</div>}
    </div>
  );
}

export function DeployTab() {
  const api = useAPI();
  const project = useProject()?.project;
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [updatingId, setUpdatingId] = useState(null);

  /**
   * 每張「可部署模型卡」的 instance_group_count 暫存值（key = ml_backend.id）。
   * 部署前可由使用者調整；不影響已部署的實例。
   * @type {[Record<number, number>, Function]}
   */
  const [pendingCounts, setPendingCounts] = useState({});

  /**
   * 每張「可部署模型卡」的 model_version 暫存值（key = ml_backend.id）。
   * @type {[Record<number, number>, Function]}
   */
  const [pendingVersions, setPendingVersions] = useState({});

  /**
   * 已部署卡片的本地設定暫存（key = deployment.id）。
   * 結構：{ instance_group_count?: number }
   * 使用者調整後不立即 PATCH，改由「套用」按鈕統一送出。
   * @type {[Record<number, { instance_group_count?: number }>, Function]}
   */
  const [deployedPending, setDeployedPending] = useState({});

  /**
   * 正在套用設定的已部署 ID（送出 PATCH 期間）。
   * @type {[number|null, Function]}
   */
  const [applyingId, setApplyingId] = useState(null);

  // Upload form state
  const [uploadModelName, setUploadModelName] = useState("");
  const fileInputRef = useRef(null);

  /**
   * 取得指定 ml_backend 的待部署實例數（預設 1）。
   * @param {number} backendId
   * @returns {number}
   */
  const getPendingCount = (backendId) => pendingCounts[backendId] ?? 1;

  /**
   * 更新指定 ml_backend 的待部署實例數。
   * @param {number} backendId
   * @param {number} count
   */
  const setPendingCount = (backendId, count) => setPendingCounts((prev) => ({ ...prev, [backendId]: count }));

  /**
   * 取得指定 ml_backend 的待部署模型版本號（預設 1）。
   * @param {number} backendId
   * @returns {number}
   */
  const getPendingVersion = (backendId) => pendingVersions[backendId] ?? 1;

  /**
   * 更新指定 ml_backend 的待部署模型版本號。
   * @param {number} backendId
   * @param {number} version
   */
  const setPendingVersion = (backendId, version) => setPendingVersions((prev) => ({ ...prev, [backendId]: version }));

  /**
   * 取得已部署卡片中指定欄位的「本地暫存值」，若無則使用伺服器值。
   * @param {number} deploymentId
   * @param {'instance_group_count'} field
   * @param {number} serverValue
   * @returns {number}
   */
  const getDeployedSetting = (deploymentId, field, serverValue) =>
    deployedPending[deploymentId]?.[field] ?? serverValue;

  /**
   * 更新已部署卡片的本地暫存設定（不立即送出 PATCH）。
   * @param {number} deploymentId
   * @param {'instance_group_count'} field
   * @param {number} value
   */
  const setDeployedSetting = (deploymentId, field, value) =>
    setDeployedPending((prev) => ({
      ...prev,
      [deploymentId]: { ...(prev[deploymentId] ?? {}), [field]: value },
    }));

  /**
   * 判斷已部署卡片是否有尚未套用的變更（與伺服器值比較）。
   * @param {number} deploymentId
   * @param {{ instance_group_count: number, model_version: number }} deployment
   * @returns {boolean}
   */
  const hasDeployedChanges = (deploymentId, deployment) => {
    const local = deployedPending[deploymentId];
    if (!local) return false;
    return (
      (local.instance_group_count !== undefined && local.instance_group_count !== deployment.instance_group_count) ||
      (local.model_version !== undefined && local.model_version !== deployment.model_version)
    );
  };

  /**
   * 清除指定已部署卡片的本地暫存（套用成功或取消後呼叫）。
   * @param {number} deploymentId
   */
  const clearDeployedPending = (deploymentId) =>
    setDeployedPending((prev) => {
      const next = { ...prev };
      delete next[deploymentId];
      return next;
    });

  const fetchList = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.callApi("deploymentsList");
      setList(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error("Failed to fetch deployments", e);
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    fetchList();
  }, [fetchList]);

  const handleToggle = async (deploymentId, currentState) => {
    setUpdatingId(deploymentId);
    try {
      await api.callApi("updateDeployment", {
        params: { pk: deploymentId },
        body: { is_enabled: !currentState },
      });
      fetchList();
    } finally {
      setUpdatingId(null);
    }
  };

  /**
   * 套用已部署模型的暫存設定變更（PATCH）。
   * - 若使用者手動指定 model_version，後端以該值為準。
   * - 若只改了 instance_group_count 而未指定版本，後端自動遞增版本。
   * 套用成功後清除本地暫存並重新取得列表。
   * @param {number} deploymentId
   * @param {{ instance_group_count: number, model_version: number }} deployment - 當前伺服器值
   */
  const handleApplySettings = async (deploymentId, deployment) => {
    const local = deployedPending[deploymentId];
    if (!local) return;

    // 僅送出實際有變更的欄位
    const body = {};
    if (local.instance_group_count !== undefined && local.instance_group_count !== deployment.instance_group_count) {
      body.instance_group_count = local.instance_group_count;
    }
    // 使用者明確調整版本時帶入，後端收到 model_version 就不再自動遞增
    if (local.model_version !== undefined && local.model_version !== deployment.model_version) {
      body.model_version = local.model_version;
    }
    if (Object.keys(body).length === 0) return;

    setApplyingId(deploymentId);
    try {
      await api.callApi("updateDeployment", {
        params: { pk: deploymentId },
        body,
      });
      clearDeployedPending(deploymentId);
      fetchList();
    } finally {
      setApplyingId(null);
    }
  };

  const handleCopyKey = (key) => {
    navigator.clipboard.writeText(key);
    alert("API Key 已複製到剪貼簿");
  };

  const handleUploadModel = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current?.files[0];
    if (!file || !project?.id) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_name", uploadModelName || file.name.split(".")[0]);

      await api.callApi("trainingModelUpload", {
        params: { pk: project.id },
        body: formData,
      });

      setShowUploadModal(false);
      setUploadModelName("");
      fetchList();
    } catch (err) {
      alert("上傳失敗: " + (err.message || "未知錯誤"));
    } finally {
      setUploading(false);
    }
  };

  const deployed = list.filter((item) => item.deployment != null);
  const available = list.filter((item) => item.deployment == null);

  if (loading) return <Spinner size={32} centered />;

  return (
    <section className={rootClass.toClassName()}>
      <div className={rootClass.elem("header-row").toClassName()}>
        <div>
          <Typography variant="headline" size="medium">
            模型部署
          </Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            管理您的模型服務，並獲得推論 API 金鑰。
          </Typography>
        </div>
        <Button variant="primary" icon={<IconPlus />} onClick={() => setShowUploadModal(true)}>
          上傳自定義模型
        </Button>
      </div>

      <div className={rootClass.elem("grid").toClassName()}>
        {deployed.map((item) => {
          const d = item.deployment;
          const depId = d.id;
          const isDirty = hasDeployedChanges(depId, d);
          const isApplying = applyingId === depId;
          const localCount = getDeployedSetting(depId, "instance_group_count", d.instance_group_count ?? 1);
          const localVersion = getDeployedSetting(depId, "model_version", d.model_version ?? 1);
          const countChanged = localCount !== (d.instance_group_count ?? 1);
          const versionChanged = localVersion !== (d.model_version ?? 1);
          // 預覽版本：若使用者手動改版本以手動值為準；若只改了設定則預覽自動遞增後的版本
          const previewVersion = versionChanged
            ? localVersion
            : countChanged
              ? (d.model_version ?? 1) + 1
              : (d.model_version ?? 1);

          return (
            <ModelCard
              key={depId}
              title={item.ml_backend.title}
              subtitle={`ID: ${depId}`}
              status={d.is_enabled ? "RUNNING" : "STOPPED"}
              isEnabled={d.is_enabled}
              onToggle={() => handleToggle(depId, d.is_enabled)}
              updating={updatingId === depId}
              footer={
                <>
                  {/* 有暫存變更時才顯示「套用」與「取消」按鈕 */}
                  {isDirty && (
                    <>
                      <Button
                        size="small"
                        variant="primary"
                        waiting={isApplying}
                        onClick={() => handleApplySettings(depId, d)}
                      >
                        套用（→ v{previewVersion}）
                      </Button>
                      <Button
                        size="small"
                        look="outlined"
                        onClick={() => clearDeployedPending(depId)}
                        disabled={isApplying}
                      >
                        取消
                      </Button>
                    </>
                  )}
                  <Button size="small" look="outlined" icon={<IconCopy />} onClick={() => handleCopyKey(d.api_key)}>
                    複製金鑰
                  </Button>
                  <Button
                    size="small"
                    look="outlined"
                    variant="negative"
                    onClick={async () => {
                      if (window.confirm("確定要移除此部署嗎？")) {
                        await api.callApi("deleteDeployment", { params: { pk: depId } });
                        fetchList();
                      }
                    }}
                  >
                    移除
                  </Button>
                </>
              }
            >
              <div className={rootClass.elem("field").toClassName()}>
                <span className={rootClass.elem("label").toClassName()}>API Key (已遮蔽)</span>
                <code className={rootClass.elem("mono").toClassName()}>●●●●●●●●●●●●●●●●●●●●</code>
              </div>

              {/* GPU 實例數：本地暫存，不即時 PATCH */}
              <div
                className={rootClass
                  .elem("field")
                  .mod({ dirty: localCount !== (d.instance_group_count ?? 1) })
                  .toClassName()}
              >
                <span
                  className={rootClass.elem("label").toClassName()}
                  title="對應 Triton instance_group[].count，套用後自動遞增版本號"
                >
                  GPU 並行實例數
                  {localCount !== (d.instance_group_count ?? 1) && (
                    <span className={rootClass.elem("diff-badge").toClassName()}>
                      {d.instance_group_count ?? 1} → {localCount}
                    </span>
                  )}
                </span>
                <InstanceGroupStepper
                  value={localCount}
                  onChange={(count) => setDeployedSetting(depId, "instance_group_count", count)}
                  disabled={isApplying}
                />
              </div>

              {/* 模型版本：可手動調整；若未手動改版但有其他設定變更，套用後自動遞增 */}
              <div className={rootClass.elem("field").mod({ dirty: versionChanged }).toClassName()}>
                <span
                  className={rootClass.elem("label").toClassName()}
                  title="手動指定則以指定值為準；只改設定未改版本則套用後自動遞增"
                >
                  模型版本（Triton version）
                  {versionChanged && (
                    <span className={rootClass.elem("diff-badge").toClassName()}>
                      v{d.model_version ?? 1} → v{localVersion}
                    </span>
                  )}
                </span>
                <div className={rootClass.elem("version-display").toClassName()}>
                  <ModelVersionStepper
                    value={localVersion}
                    onChange={(v) => setDeployedSetting(depId, "model_version", v)}
                    disabled={isApplying}
                  />
                  {!versionChanged && countChanged && (
                    <span className={rootClass.elem("version-arrow").toClassName()}>
                      → <strong>v{previewVersion}</strong>（設定變更後自動遞增）
                    </span>
                  )}
                </div>
              </div>
            </ModelCard>
          );
        })}

        {available.map((item) => (
          <ModelCard
            key={item.ml_backend.id}
            title={item.ml_backend.title}
            subtitle={`專案: ${item.ml_backend.project_title}`}
            status="READY"
            type="available"
            footer={
              <Button
                variant="primary"
                size="small"
                onClick={async () => {
                  await api.callApi("createDeployment", {
                    body: {
                      ml_backend_id: item.ml_backend.id,
                      instance_group_count: getPendingCount(item.ml_backend.id),
                      model_version: getPendingVersion(item.ml_backend.id),
                    },
                  });
                  fetchList();
                }}
              >
                部署並取得金鑰
              </Button>
            }
          >
            <Typography variant="body" size="small">
              此模型已訓練完成，準備好提供服務。
            </Typography>
            {/* 部署前設定 GPU 並行實例數（不需修改 pbtxt） */}
            <div className={rootClass.elem("field").toClassName()} style={{ marginTop: 8 }}>
              <span
                className={rootClass.elem("label").toClassName()}
                title="對應 Triton instance_group[].count，指定 GPU 並行模型實例數量"
              >
                GPU 並行實例數（1–10）
              </span>
              <InstanceGroupStepper
                value={getPendingCount(item.ml_backend.id)}
                onChange={(count) => setPendingCount(item.ml_backend.id, count)}
              />
            </div>
            {/* 部署前設定 Triton 模型版本號 */}
            <div className={rootClass.elem("field").toClassName()}>
              <span
                className={rootClass.elem("label").toClassName()}
                title="Triton 模型版本號，對應倉庫版本子目錄（1/ 2/ 3/...），預設為 1"
              >
                模型版本（Triton version）
              </span>
              <ModelVersionStepper
                value={getPendingVersion(item.ml_backend.id)}
                onChange={(v) => setPendingVersion(item.ml_backend.id, v)}
              />
            </div>
          </ModelCard>
        ))}
      </div>

      <Modal
        visible={showUploadModal}
        title="上傳自定義模型 (.pt / .pth)"
        onHide={() => setShowUploadModal(false)}
        footer={null}
      >
        <form className="upload-modal__form" onSubmit={handleUploadModel}>
          <div>
            <Typography variant="label">模型顯示名稱</Typography>
            <input
              className="ls-input"
              value={uploadModelName}
              onChange={(e) => setUploadModelName(e.target.value)}
              placeholder="例如: my-custom-resnet"
              required
            />
          </div>

          <div className="upload-modal__dropzone" onClick={() => fileInputRef.current?.click()}>
            <IconUpload size={32} className="mb-tight" />
            <Typography variant="body">點擊或拖放 .pt 或 .pth 檔案至此</Typography>
            <input
              type="file"
              ref={fileInputRef}
              accept=".pt,.pth"
              onChange={(e) => setUploadModelName((v) => v || e.target.files[0]?.name.split(".")[0])}
            />
            {fileInputRef.current?.files[0] && (
              <div className="mt-tight text-primary">已選擇: {fileInputRef.current.files[0].name}</div>
            )}
          </div>

          <div className="flex justify-end gap-2 mt-wide">
            <Button onClick={() => setShowUploadModal(false)}>取消</Button>
            <Button variant="primary" type="submit" waiting={uploading}>
              開始上傳與部署
            </Button>
          </div>
        </form>
      </Modal>
    </section>
  );
}
