import { useCallback, useEffect, useState, useRef } from "react";
import { Button, Typography, SimpleCard, Spinner } from "@humansignal/ui";
import { IconCopy, IconPlus, IconUpload, IconInfoOutline } from "@humansignal/icons";
import { useAPI } from "../../providers/ApiProvider";
import { cn } from "../../utils/bem";
import { Modal, confirm } from "../../components/Modal/Modal";
import { useProject } from "../../providers/ProjectProvider";
import { Toggle } from "../../components/Form";
import "./ModelDeployment.module.scss";

const rootClass = cn("model-deploy-tab");

/**
 * Premium 磁貼卡片：顯示模型資訊與操作
 */
function ModelCard({ title, subtitle, status, children, footer, type = "deployed", onToggle, isEnabled, updating }) {
  return (
    <div className={rootClass.elem("card").mod({ type, disabled: !isEnabled && type === "deployed" }).toClassName()}>
      <div className={rootClass.elem("card-header").toClassName()}>
        <div>
          <Typography variant="title" size="medium">{title}</Typography>
          {subtitle && (
             <Typography variant="body" size="small" className="text-neutral-content-subtle">
               {subtitle}
             </Typography>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {onToggle && (
            <Toggle 
              checked={isEnabled} 
              onChange={onToggle} 
              disabled={updating}
              style={{ transform: 'scale(0.8)' }}
            />
          )}
          <span className={rootClass.elem("status").mod({ [status.toLowerCase()]: true }).toClassName()}>
            {status}
          </span>
        </div>
      </div>
      <div className={rootClass.elem("card-body").toClassName()}>
        {children}
      </div>
      {footer && (
        <div className={rootClass.elem("actions").toClassName()}>
          {footer}
        </div>
      )}
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
  
  // Upload form state
  const [uploadModelName, setUploadModelName] = useState("");
  const fileInputRef = useRef(null);

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
          <Typography variant="headline" size="medium">模型部署</Typography>
          <Typography variant="body" size="small" className="text-neutral-content-subtle">
            管理您的模型服務，並獲得推論 API 金鑰。
          </Typography>
        </div>
        <Button 
          variant="primary" 
          icon={<IconPlus />} 
          onClick={() => setShowUploadModal(true)}
        >
          上傳自定義模型
        </Button>
      </div>

      <div className={rootClass.elem("grid").toClassName()}>
        {deployed.map((item) => (
          <ModelCard 
            key={item.deployment.id}
            title={item.ml_backend.title}
            subtitle={`ID: ${item.deployment.id}`}
            status={item.deployment.is_enabled ? "RUNNING" : "STOPPED"}
            isEnabled={item.deployment.is_enabled}
            onToggle={() => handleToggle(item.deployment.id, item.deployment.is_enabled)}
            updating={updatingId === item.deployment.id}
            footer={(
              <>
                <Button 
                  size="small" 
                  look="outlined" 
                  icon={<IconCopy />}
                  onClick={() => handleCopyKey(item.deployment.api_key)}
                >
                  複製金鑰
                </Button>
                <Button 
                  size="small" 
                  look="outlined" 
                  variant="negative"
                  onClick={async () => {
                    if (window.confirm("確定要移除此部署嗎？")) {
                       await api.callApi("deleteDeployment", { params: { pk: item.deployment.id } });
                       fetchList();
                    }
                  }}
                >
                  移除
                </Button>
              </>
            )}
          >
             <div className={rootClass.elem("field").toClassName()}>
               <span className={rootClass.elem("label").toClassName()}>API Key (已遮蔽)</span>
               <code className={rootClass.elem("mono").toClassName()}>
                 ●●●●●●●●●●●●●●●●●●●●
               </code>
             </div>
          </ModelCard>
        ))}
        
        {available.map((item) => (
          <ModelCard 
            key={item.ml_backend.id}
            title={item.ml_backend.title}
            subtitle={`專案: ${item.ml_backend.project_title}`}
            status="READY"
            type="available"
            footer={(
              <Button 
                variant="primary" 
                size="small"
                onClick={async () => {
                  await api.callApi("createDeployment", { body: { ml_backend_id: item.ml_backend.id } });
                  fetchList();
                }}
              >
                部署並取得金鑰
              </Button>
            )}
          >
            <Typography variant="body" size="small">
              此模型已訓練完成，準備好提供服務。
            </Typography>
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
              onChange={e => setUploadModelName(e.target.value)}
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
              onChange={e => setUploadModelName(v => v || e.target.files[0]?.name.split('.')[0])}
            />
            {fileInputRef.current?.files[0] && (
               <div className="mt-tight text-primary">
                 已選擇: {fileInputRef.current.files[0].name}
               </div>
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
