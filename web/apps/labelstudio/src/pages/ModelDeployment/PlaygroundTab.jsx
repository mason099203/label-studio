import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Typography } from "@humansignal/ui";
import { useAPI } from "../../providers/ApiProvider";
import { cn } from "../../utils/bem";
import "./ModelDeployment.module.scss";

const rootClass = cn("playground-tab");
const PLAYGROUND_STATE_KEY = "labelstudio.triton.playground";

/**
 * 讀取最近一次部署後寫入的 Playground 預設值。
 * @returns {{projectId: string, modelName: string, apiKey?: string}}
 */
function readSavedPlaygroundState() {
  try {
    const raw = window.localStorage.getItem(PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};

    return {
      projectId: parsed?.projectId ? String(parsed.projectId) : "",
      modelName: parsed?.modelName ?? "",
      apiKey: parsed?.apiKey ?? "",
    };
  } catch (_) {
    return { projectId: "", modelName: "", apiKey: "" };
  }
}

/**
 * 儲存 Playground 最近使用的專案、模型與設定，讓部署後可直接測試。
 * @param {{projectId: string, modelName: string, apiKey: string}} nextState
 */
function persistPlaygroundState(nextState) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(PLAYGROUND_STATE_KEY, JSON.stringify(nextState));
}

/**
 * 測試 Triton Playground：選擇專案與已部署模型，上傳圖片後透過代理送出 infer request。
 */
export function PlaygroundTab() {
  const api = useAPI();
  const savedState = useMemo(() => readSavedPlaygroundState(), []);
  const [projectId, setProjectId] = useState(savedState.projectId);
  const [modelName, setModelName] = useState(savedState.modelName);
  const [apiKey, setApiKey] = useState(savedState.apiKey);
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState({ status: null, summary: null, error: null });

  // 預覽與 Tensor 處理使用 ref
  const canvasRef = useRef(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processingImage, setProcessingImage] = useState(false);
  
  // 不要將 Tensor 資料存進 useState，避免渲染卡頓
  const tensorPayloadRef = useRef(null);

  useEffect(() => {
    persistPlaygroundState({ projectId, modelName, apiKey });
  }, [projectId, modelName, apiKey]);

  useEffect(() => {
    if (!projectId) {
      setModels([]);
      setModelsError(null);
      return;
    }

    let cancelled = false;
    setModelsLoading(true);
    setModelsError(null);

    api
      .callApi("trainingTritonModels", {
        params: { pk: projectId },
        errorFilter: () => true,
      })
      .then((res) => {
        if (cancelled) return;
        const nextModels = Array.isArray(res?.models) ? res.models : [];
        setModels(nextModels);
        
        // 若之前儲存的 model 不存在於剛拉下來的清單中，重設為第一個
        const modelExists = nextModels.some((m) => m.model_name === modelName);
        if (!modelExists && nextModels.length > 0) {
          setModelName(nextModels[0].model_name);
        } else if (!modelExists) {
          setModelName("");
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setModels([]);
        setModelsError(err?.message || "無法載入 Triton 模型列表");
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [api, projectId]);

  // 處理圖片上傳與預先調整大小轉換為 Tensor
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setProcessingImage(true);
    const imageUrl = URL.createObjectURL(file);
    setImagePreview(imageUrl);
    tensorPayloadRef.current = null;
    setResponse({ status: null, summary: null, error: null });

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        setProcessingImage(false);
        return;
      }
      
      const TARGET_SIZE = 640;
      canvas.width = TARGET_SIZE;
      canvas.height = TARGET_SIZE;
      const ctx = canvas.getContext("2d");
      
      // 計算保持比例的縮放與 padding (YOLO common resize approach: letterbox)
      const scale = Math.min(TARGET_SIZE / img.width, TARGET_SIZE / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const offsetX = (TARGET_SIZE - scaledWidth) / 2;
      const offsetY = (TARGET_SIZE - scaledHeight) / 2;

      // 填滿背景色 (YOLO 建議通常使用 114)
      ctx.fillStyle = "rgb(114, 114, 114)";
      ctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
      ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

      // 取得影像像素，並轉為 [1, 3, 640, 640] 的一維陣列 (FP32, normalized to 0-1)
      const imageData = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
      const numPixels = TARGET_SIZE * TARGET_SIZE;
      
      // CHW 順序: R channel, G channel, B channel
      const rChannel = new Float32Array(numPixels);
      const gChannel = new Float32Array(numPixels);
      const bChannel = new Float32Array(numPixels);

      for (let i = 0; i < numPixels; i++) {
        // Normalize 到 0~1 之間 (YOLOv8/v11 標準預設前處理)
        rChannel[i] = imageData[i * 4] / 255.0;
        gChannel[i] = imageData[i * 4 + 1] / 255.0;
        bChannel[i] = imageData[i * 4 + 2] / 255.0;
      }

      // 串接成一維: [R..., G..., B...]
      const tensorData = new Float32Array(numPixels * 3);
      tensorData.set(rChannel, 0);
      tensorData.set(gChannel, numPixels);
      tensorData.set(bChannel, numPixels * 2);

      const flatDataArray = Array.from(tensorData);

      tensorPayloadRef.current = {
        inputs: [
          {
            name: "images",
            shape: [1, 3, TARGET_SIZE, TARGET_SIZE],
            datatype: "FP32",
            data: flatDataArray,
          },
        ],
        outputs: [{ name: "output0" }],
      };

      setProcessingImage(false);
    };
    img.src = imageUrl;
  };

  const sendRequest = useCallback(async () => {
    setLoading(true);
    setResponse({ status: null, summary: null, error: null });
    try {
      if (!projectId.trim()) {
        setResponse({ status: null, summary: null, error: "請先輸入專案 ID" });
        return;
      }
      if (!modelName.trim()) {
        setResponse({ status: null, summary: null, error: "請先選擇模型" });
        return;
      }
      if (!tensorPayloadRef.current) {
        setResponse({ status: null, summary: null, error: "請先上傳圖片" });
        return;
      }

      const res = await api.callApi("trainingTritonInfer", {
        params: { pk: projectId },
        body: { model_name: modelName, api_key: apiKey, ...tensorPayloadRef.current },
        errorFilter: () => true,
      });
      
      let summaryStr = "";
      if (res?.body?.outputs) {
        summaryStr = res.body.outputs.map(o => `輸出節點: ${o.name}, 形狀: [${(o.shape || []).join(", ")}]`).join("\n");
      }

      setResponse({
        status: res?.status_code ?? 200,
        statusText: res?.ok ? "OK" : "ERROR",
        summary: summaryStr || JSON.stringify(res?.body ?? res).substring(0, 200) + "...",
        ok: Boolean(res?.ok),
      });
      
      // 如果預覽且為 YOLO 的 output0，嘗試加上 BBox (簡化版示意繪圖)
      drawBBoxesIfPossible(res?.body);
    } catch (err) {
      setResponse({
        status: null,
        summary: null,
        error: err.message || "請求失敗",
      });
    } finally {
      setLoading(false);
    }
  }, [api, modelName, projectId, apiKey]);

  // 超越極限簡化版的 BBox 繪製 (假設輸出形式為 YOLOv8 的 [1, 84, 8400])
  const drawBBoxesIfPossible = (responseBody) => {
     if (!canvasRef.current || !responseBody?.outputs || !imagePreview) return;
     const output = responseBody.outputs.find(o => o.name === "output0");
     if (!output || !output.shape || output.shape.length !== 3) return; // shape: [1, num_classes+4, boxes]
     
     // Note: 這只是非常非常粗略地取出前幾高機率的框示意，並未包含完整的 NMS 演算法。
     const data = output.data;
     const shape = output.shape;
     const numRows = shape[1]; 
     const numCols = shape[2];
     
     if (numRows < 4) return;
     
     const ctx = canvasRef.current.getContext("2d");
     // 不要清空背景，我們要蓋在原有圖片上
     ctx.strokeStyle = "#4CAF50";
     ctx.lineWidth = 3;
     ctx.font = "16px Arial";
     ctx.fillStyle = "#4CAF50";
     
     // 找最高的 confidences
     const threshold = 0.25;
     let foundBoxes = 0;
     for (let col = 0; col < numCols; col++) {
         let maxConf = 0;
         let bestClassIndex = -1;
         
         // column-major array
         for (let row = 4; row < numRows; row++) {
             const val = data[row * numCols + col];
             if (val > maxConf) {
                 maxConf = val;
                 bestClassIndex = row - 4;
             }
         }
         
         if (maxConf > threshold) {
             const cx = data[0 * numCols + col];
             const cy = data[1 * numCols + col];
             const w  = data[2 * numCols + col];
             const h  = data[3 * numCols + col];
             
             const x = cx - w / 2;
             const y = cy - h / 2;
             
             ctx.strokeRect(x, y, w, h);
             const text = `Class ${bestClassIndex} (${maxConf.toFixed(2)})`;
             // 背景以使文字更易讀
             const textWidth = ctx.measureText(text).width;
             ctx.fillStyle = "rgba(76, 175, 80, 0.8)";
             ctx.fillRect(x, y > 20 ? y - 20 : y, textWidth + 8, 20);
             ctx.fillStyle = "white";
             ctx.fillText(text, x + 4, y > 20 ? y - 4 : y + 16);
             ctx.fillStyle = "#4CAF50";
             foundBoxes++;
         }
     }
     
     if (foundBoxes === 0) {
        ctx.fillStyle = "red";
        ctx.fillText("未偵測到超過門檻的物體", 10, 20);
     }
  };

  return (
    <section className={rootClass.toClassName()}>
      <Typography variant="headline" size="medium" className="mb-wide">
        Playground
      </Typography>
      <Typography variant="body" size="small" className="text-neutral-content-subtle mb-wide">
        選擇 Triton 已部署之模型，並上傳圖片來測試其推理效能與視覺化結果。
      </Typography>

      <div className={rootClass.elem("form").toClassName()}>
        <div className={rootClass.elem("form-row").toClassName()} style={{ display: "flex", gap: "16px", marginBottom: "16px" }}>
           <div className={rootClass.elem("field").toClassName()} style={{ flex: 1 }}>
             <label className="text-label-small text-neutral-content mb-tightest block">資源專案 ID</label>
             <input
               type="text"
               className={rootClass.elem("input").toClassName()}
               value={projectId}
               onChange={(e) => setProjectId(e.target.value)}
               placeholder="例如：10"
             />
           </div>
           <div className={rootClass.elem("field").toClassName()} style={{ flex: 1 }}>
             <label className="text-label-small text-neutral-content mb-tightest block">Triton 安全金鑰 (API Key)</label>
             <input
               type="password"
               className={rootClass.elem("input").toClassName()}
               value={apiKey}
               onChange={(e) => setApiKey(e.target.value)}
               placeholder="輸入專案對應的 API Key (選填)"
             />
           </div>
        </div>

        <div className={rootClass.elem("field").toClassName()} style={{ marginBottom: "16px" }}>
          <label className="text-label-small text-neutral-content mb-tightest block">選擇模型</label>
          <select
            className={rootClass.elem("input").toClassName()}
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            disabled={modelsLoading || models.length === 0}
            style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc", height: "36px" }}
          >
            {modelsLoading && <option value="">載入中...</option>}
            {!modelsLoading && models.length === 0 && <option value="">未找到已部署的模型</option>}
            {!modelsLoading && models.length > 0 && models.map((model) => (
              <option key={model.model_name} value={model.model_name}>
                {model.model_name} (部署於: {new Date(model.deployed_at || Date.now()).toLocaleString()})
              </option>
            ))}
          </select>
          {modelsError && (
            <div className={rootClass.elem("error").toClassName()}>{modelsError}</div>
          )}
        </div>

        <div className={rootClass.elem("field").toClassName()} style={{ marginBottom: "24px", border: "1px solid #e0e0e0", padding: "16px", borderRadius: "8px", backgroundColor: "#fff" }}>
          <label className="text-label-small text-neutral-content block" style={{ marginBottom: "8px", fontWeight: "bold" }}>測試圖片上傳</label>
          <input type="file" accept="image/*" onChange={handleImageUpload} disabled={processingImage} />
          {processingImage && <span className="text-neutral-content-subtle ml-tight">前處理運算中...</span>}
          
          <div style={{ marginTop: "16px", display: imagePreview ? "block" : "none", textAlign: "center" }}>
              <canvas 
                 ref={canvasRef} 
                 style={{ maxWidth: "100%", maxHeight: "500px", border: "1px solid #ddd", borderRadius: "4px", backgroundColor: "#f9f9f9" }}
              />
          </div>
        </div>

        <Button
          variant="primary"
          look="filled"
          onClick={sendRequest}
          disabled={loading || !projectId.trim() || !modelName.trim() || !imagePreview || processingImage}
          aria-label="發送 Triton 請求"
        >
          {loading ? "推理執行中…" : "發送影像至模型測試"}
        </Button>
      </div>

      {(response.status != null || response.error) && (
        <div className={rootClass.elem("response").toClassName()} style={{ marginTop: "24px" }}>
          <Typography variant="title" size="medium" className="mb-tight mt-wide block">
            推理結果
          </Typography>
          {response.error && (
            <div className={rootClass.elem("error").toClassName()} style={{ padding: "12px", background: "#fee2e2", color: "#b91c1c", borderRadius: "6px" }}>
              {response.error}
            </div>
          )}
          {response.status != null && (
            <div style={{ padding: "12px", background: response.ok ? "#f0fdf4" : "#fef2f2", borderRadius: "6px" }}>
              <div className={rootClass.elem("status").toClassName()} style={{ fontWeight: "bold", marginBottom: "8px", color: response.ok ? "#166534" : "#991b1b" }}>
                HTTP 狀態: {response.status} {response.statusText}
              </div>
              <pre className={rootClass.elem("pre").toClassName()} style={{ margin: 0, whiteSpace: "pre-wrap", color: "#374151", fontSize: "14px", fontFamily: "monospace" }}>
                {response.summary}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
