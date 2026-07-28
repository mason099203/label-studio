import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Typography } from "@humansignal/ui";
import { useAPI } from "../../providers/ApiProvider";
import { cn } from "../../utils/bem";
import {
  TRITON_PLAYGROUND_STATE_KEY,
  verifyTritonConnection,
  normalizeTritonUrl,
  validateRemoteServiceUrl,
  persistTritonUrlFields,
} from "./tritonUrlState";
import {
  inferPlaygroundTaskTypeFromLabelConfig,
  isSpatialYoloTaskType,
  normalizePlaygroundTaskType,
  playgroundInputSizeForTask,
  playgroundTaskTypeLabel,
  PLAYGROUND_TASK_TYPES,
} from "./trainingTaskTypes";
import "./ModelDeployment.scss";

const rootClass = cn("playground-tab");

/**
 * 讀取最近一次部署後寫入的 Playground 預設值。
 * @returns {{
 *   projectId: string,
 *   modelName: string,
 *   apiKey?: string,
 *   taskType?: string,
 *   tritonServerUrl?: string,
 *   tritonMetricsUrl?: string,
 * }}
 */
function readSavedPlaygroundState() {
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};

    return {
      projectId: parsed?.projectId ? String(parsed.projectId) : "",
      modelName: parsed?.modelName ?? "",
      apiKey: parsed?.apiKey ?? "",
      taskType: normalizePlaygroundTaskType(parsed?.taskType),
      tritonServerUrl: typeof parsed?.tritonServerUrl === "string" ? parsed.tritonServerUrl : "",
      tritonMetricsUrl: typeof parsed?.tritonMetricsUrl === "string" ? parsed.tritonMetricsUrl : "",
    };
  } catch (_) {
    return {
      projectId: "",
      modelName: "",
      apiKey: "",
      taskType: "detect",
      tritonServerUrl: "",
      tritonMetricsUrl: "",
    };
  }
}

/**
 * 儲存 Playground 最近使用的專案、模型與設定，讓部署後可直接測試。
 * @param {{
 *   projectId: string,
 *   modelName: string,
 *   apiKey: string,
 *   taskType?: string,
 *   tritonServerUrl?: string,
 *   tritonMetricsUrl?: string,
 * }} nextState
 */
function persistPlaygroundState(nextState) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TRITON_PLAYGROUND_STATE_KEY, JSON.stringify(nextState));
}

/** Playground 輸入解析度（與送進 Triton 的 tensor 一致）。 */
const PLAYGROUND_INPUT_SIZE = 640;

/**
 * 將後端 ephemeral 生命週期（load → infer → unload）格式化成可讀摘要。
 * @param {Record<string, unknown> | undefined} lifecycle
 * @returns {string}
 */
function formatTritonLifecycleSummary(lifecycle) {
  if (!lifecycle) return "";
  const lines = [];
  if (lifecycle.was_ready) {
    lines.push("模型狀態：已載入（直接使用）");
  } else if (lifecycle.load?.ok) {
    lines.push("模型狀態：已自動載入");
  } else if (lifecycle.load) {
    lines.push(`模型載入：失敗 — ${lifecycle.load.detail || "未知錯誤"}`);
  }
  if (lifecycle.unload?.ok) {
    lines.push("模型狀態：測試完成後已釋放記憶體");
  } else if (lifecycle.unload) {
    lines.push(`模型釋放：${lifecycle.unload.detail || "失敗或未執行"}`);
  }
  return lines.length ? `${lines.join("\n")}\n\n` : "";
}

/**
 * 組合 Label Studio 代理至 Triton 的推論 API URL（對應 `trainingTritonInfer`）。
 * @param {string} hostname 例如 `window.APP_SETTINGS.hostname`，不含結尾斜線
 * @param {string} projectId 專案主鍵
 * @returns {string}
 */
function buildTritonInferProxyUrl(hostname, projectId) {
  const base = String(hostname || "").replace(/\/+$/, "");
  const pk = String(projectId || "").trim();
  if (!base || !pk) return "";
  return `${base}/api/projects/${encodeURIComponent(pk)}/training/triton/infer/`;
}

/**
 * 從瀏覽器 Cookie 中讀取 `sessionid` 值。
 * 僅用於填充程式碼範例，方便複製後直接執行。
 * @returns {string} 找不到時回傳空字串
 */
function readSessionIdFromCookie() {
  if (typeof document === "undefined") return "";
  const seg = document.cookie.split(";").find((c) => c.trim().startsWith("sessionid="));
  return seg ? seg.trim().slice("sessionid=".length) : "";
}

/**
 * 產生模型測試頁「程式呼叫 API」區塊的範例字串（供 pre 區塊顯示）。
 * 當有 sessionId 時直接填入；Python／JS 均含完整圖片前處理程式碼，
 * 使用者只需改圖片路徑或 <input> 來源即可直接執行。
 *
 * Python 範例包含兩種呼叫方式：
 *   1. 直接呼叫 Triton REST API（`/v2/models/{model}/infer`）
 *   2. 透過 Label Studio 代理（`/api/projects/{pk}/training/triton/infer/`，需 sessionid Cookie）
 *
 * @param {{
 *   inferUrl: string,
 *   modelName: string,
 *   apiKey: string,
 *   sessionId?: string,
 *   tritonServerUrl?: string,
 *   inputSize?: number,
 * }} opts
 * @returns {{ js: string, curl: string, py: string }}
 */
function buildPlaygroundApiExampleSnippets(opts) {
  const { inferUrl, modelName, apiKey, sessionId, tritonServerUrl, inputSize } = opts;
  const SIZE = inputSize ?? 640;
  const modelJson = JSON.stringify(modelName || "YOUR_MODEL_NAME");
  const apiKeyJson = JSON.stringify(apiKey && String(apiKey).trim() ? String(apiKey).trim() : "");
  const sid = sessionId && String(sessionId).trim() ? String(sessionId).trim() : "YOUR_SESSION";
  const urlStr = inferUrl || "https://YOUR_HOST/api/projects/YOUR_PROJECT_ID/training/triton/infer/";

  /* ── 從 tritonServerUrl 解析出 host / port（供 Python 直接呼叫範例） ─────── */
  let tritonHost = "YOUR_TRITON_HOST";
  let tritonPort = 18000;
  if (tritonServerUrl) {
    try {
      const parsed = new URL(tritonServerUrl);
      tritonHost = parsed.hostname || tritonHost;
      tritonPort = parsed.port ? Number(parsed.port) : parsed.protocol === "https:" ? 443 : 18000;
    } catch (_) {
      // URL 格式不合法，保留佔位符
    }
  }

  /* ── JavaScript（canvas letterbox + fetch） ─────────────────────────────── */
  const js = `/**
 * 與本頁 Playground 完全一致的前處理：
 *   letterbox 縮放至 ${SIZE}×${SIZE}，灰邊 rgb(114,114,114)，
 *   像素轉 CHW Float32（0.0 ~ 1.0）後攤平為一維陣列。
 *
 * 使用方式：將 <input type="file"> 選取的 File 傳入 imageFileToTensor()，
 * 取得 data 陣列後填入 payload.inputs[0].data。
 */
const SIZE = ${SIZE};

/**
 * @param {File} file
 * @returns {Promise<number[]>}
 */
async function imageFileToTensor(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = SIZE;
      const ctx = canvas.getContext("2d");
      const scale = Math.min(SIZE / img.width, SIZE / img.height);
      const nw = img.width * scale, nh = img.height * scale;
      const ox = (SIZE - nw) / 2, oy = (SIZE - nh) / 2;
      ctx.fillStyle = "rgb(114,114,114)";
      ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.drawImage(img, ox, oy, nw, nh);
      const px = ctx.getImageData(0, 0, SIZE, SIZE).data; // RGBA
      const n = SIZE * SIZE;
      const t = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        t[i]         = px[i * 4]     / 255; // R channel
        t[n + i]     = px[i * 4 + 1] / 255; // G channel
        t[n * 2 + i] = px[i * 4 + 2] / 255; // B channel
      }
      resolve(Array.from(t)); // length = ${1 * 3 * SIZE * SIZE}
    };
    img.src = URL.createObjectURL(file);
  });
}

// ── 送出推論 ──────────────────────────────────────────────────────────────
const url = ${JSON.stringify(urlStr)};

// 取得圖片（瀏覽器環境：已登入，credentials: "include" 自動帶 Cookie）
const fileInput = document.querySelector('input[type="file"]');
const tensorData = await imageFileToTensor(fileInput.files[0]);

const body = {
  model_name: ${modelJson},
  api_key: ${apiKeyJson},
  inputs: [{
    name: "images",
    shape: [1, 3, ${SIZE}, ${SIZE}],
    datatype: "FP32",
    data: tensorData,
  }],
  outputs: [{ name: "output0" }],
};

const res = await fetch(url, {
  method: "POST",
  credentials: "include",           // 自動帶入 sessionid Cookie
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const json = await res.json();
console.log(json);`;

  /* ── Python（PIL letterbox + requests） ─────────────────────────────────── */
  const py = `"""
與本頁 Playground 完全一致的前處理：
  letterbox 縮放至 ${SIZE}×${SIZE}，灰邊 (114,114,114)，CHW Float32 (0.0~1.0)。

依賴：pip install requests Pillow numpy
使用：修改 IMAGE_PATH 為你的圖片路徑，即可直接執行。
"""
import requests
import numpy as np
from PIL import Image

IMAGE_PATH = "your_image.jpg"   # ← 改成你的圖片路徑
SIZE = ${SIZE}

def letterbox_to_tensor(path, size=SIZE):
    """讀取圖片並執行 letterbox 前處理，回傳一維 FP32 list（CHW，0~1）。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    arr = np.array(canvas, dtype=np.float32) / 255.0   # HWC, 0~1
    arr = arr.transpose(2, 0, 1).ravel().tolist()       # CHW → 一維
    return arr  # length = ${1 * 3 * SIZE * SIZE}

url = ${JSON.stringify(urlStr)}
tensor = letterbox_to_tensor(IMAGE_PATH)

payload = {
    "model_name": ${modelJson},
    "api_key": ${apiKeyJson},
    "inputs": [{
        "name": "images",
        "shape": [1, 3, ${SIZE}, ${SIZE}],
        "datatype": "FP32",
        "data": tensor,
    }],
    "outputs": [{"name": "output0"}],
}

# sessionid 取自瀏覽器 Cookie（F12 → Application → Cookies → sessionid）
r = requests.post(url, json=payload, cookies={"sessionid": "${sid}"})
print(r.status_code)
print(r.json())`;

  /* ── cURL（由 Python 產生 payload 檔再呼叫） ──────────────────────────── */
  const curl = `# Tensor 資料龐大（${1 * 3 * SIZE * SIZE} 個 float），不適合直接嵌入 cURL 指令。
# 建議：先用 Python 將 payload 存成 JSON 檔，再以 cURL 傳送。

# 步驟 1 — 產生 payload.json（執行一次即可）
python3 - <<'EOF'
import json
import numpy as np
from PIL import Image

SIZE = ${SIZE}
def letterbox_to_tensor(path, size=SIZE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(size/w, size/h)
    nw, nh = int(w*scale), int(h*scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    c = Image.new("RGB", (size, size), (114,114,114))
    c.paste(img, ((size-nw)//2, (size-nh)//2))
    arr = np.array(c, dtype=np.float32)/255.0
    return arr.transpose(2,0,1).ravel().tolist()

payload = {
    "model_name": ${modelJson},
    "api_key": ${apiKeyJson},
    "inputs": [{"name":"images","shape":[1,3,${SIZE},${SIZE}],"datatype":"FP32","data":letterbox_to_tensor("your_image.jpg")}],
    "outputs": [{"name":"output0"}],
}
with open("payload.json","w") as f:
    json.dump(payload, f)
print("payload.json 已產生")
EOF

# 步驟 2 — 送出推論（sessionid 取自瀏覽器 Cookie）
curl -X POST ${JSON.stringify(urlStr)} \\
  -H "Content-Type: application/json" \\
  -H "Cookie: sessionid=${sid}" \\
  --data @payload.json`;

  return { js, curl, py };
}

/**
 * 一鍵複製程式碼按鈕。點擊後顯示「已複製 ✓」2 秒，並回復原狀。
 * 使用 Clipboard API，降級至 execCommand。
 * @param {{ text: string, className?: string }} props
 */
function CopyCodeButton({ text, className }) {
  const [copied, setCopied] = useState(false);

  /** @param {React.MouseEvent} e */
  const handleCopy = (e) => {
    // 防止觸發 <details> 的展開/收合
    e.preventDefault();
    e.stopPropagation();

    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    if (navigator?.clipboard?.writeText) {
      navigator.clipboard
        .writeText(text)
        .then(done)
        .catch(() => {
          execCommandCopy(text);
          done();
        });
    } else {
      execCommandCopy(text);
      done();
    }
  };

  return (
    <button type="button" className={className} onClick={handleCopy} aria-label="複製程式碼">
      {copied ? "已複製 ✓" : "複製"}
    </button>
  );
}

/**
 * 降級複製方案：使用 document.execCommand('copy')。
 * @param {string} text
 */
function execCommandCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;opacity:0;pointer-events:none;";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
  } catch (_) {
    // 無法複製時靜默失敗
  }
  document.body.removeChild(ta);
}

/**
 * 依專案 Labeling Interface（label_config XML）推斷影像模型測試應使用偵測或分類流程。
 * 優先辨識空間標註標籤；否則在含 Image 且含 Choices 時視為影像分類。
 * @param {string | null | undefined} labelConfig
 * @returns {{ taskType: "detect" | "classification", detail: string } | null} 無法判斷時回傳 null
 */
/**
 * 依專案列表建立選項顯示文字：以標題為主；重複標題時附帶 ID 以區分。
 * @param {{ id: number | string, title?: string | null }[]} projectList
 * @returns {Map<string, string>} 專案 id（字串）→ 顯示字串
 */
function buildProjectSelectLabelsById(projectList) {
  const list = Array.isArray(projectList) ? projectList : [];
  const titleCounts = {};
  for (const p of list) {
    const t = (p?.title != null ? String(p.title) : "").trim() || "未命名專案";
    titleCounts[t] = (titleCounts[t] || 0) + 1;
  }
  /** @type {Map<string, string>} */
  const map = new Map();
  for (const p of list) {
    const id = p?.id;
    if (id === undefined || id === null) continue;
    const idKey = String(id);
    const t = (p?.title != null ? String(p.title) : "").trim() || "未命名專案";
    const label = titleCounts[t] > 1 ? `${t} (#${idKey})` : t;
    map.set(idKey, label);
  }
  return map;
}

/**
 * 自 label_config 中第一個 Choices 區塊依序取得 Choice 的 value（對應分類輸出之類別索引）。
 * @param {string | null | undefined} labelConfig
 * @returns {string[]}
 */
function extractClassificationChoiceLabels(labelConfig) {
  if (typeof labelConfig !== "string" || !labelConfig.trim()) return [];
  const xml = labelConfig.replace(/<!--[\s\S]*?-->/g, " ");
  const blockMatch = xml.match(/<Choices\b[^>]*>[\s\S]*?<\/Choices>/i);
  if (!blockMatch) return [];
  const block = blockMatch[0];
  const values = [];
  const re = /<Choice\b[^>]*\bvalue\s*=\s*(["'])([\s\S]*?)\1/gi;
  let m;
  while ((m = re.exec(block)) !== null) {
    const v = m[2].trim();
    if (v) values.push(v);
  }
  return values;
}

/**
 * 自 RectangleLabels 等區塊依序取得 Label 的 value（對應偵測類別索引）。
 * @param {string | null | undefined} labelConfig
 * @returns {string[]}
 */
function extractSpatialControlLabelNames(labelConfig) {
  if (typeof labelConfig !== "string" || !labelConfig.trim()) return [];
  const xml = labelConfig.replace(/<!--[\s\S]*?-->/g, " ");
  const tagNames = ["RectangleLabels", "PolygonLabels", "KeyPointLabels", "EllipseLabels", "BrushLabels", "MaskLabels"];
  for (const tag of tagNames) {
    const reBlock = new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?<\\/${tag}>`, "i");
    const bm = xml.match(reBlock);
    if (!bm) continue;
    const block = bm[0];
    const values = [];
    const re = /<Label\b[^>]*\bvalue\s*=\s*(["'])([\s\S]*?)\1/gi;
    let m;
    while ((m = re.exec(block)) !== null) {
      const v = m[2].trim();
      if (v) values.push(v);
    }
    if (values.length) return values;
  }
  return [];
}

/**
 * 將類別索引對應到 Labeling Interface 中的顯示名稱（若無對應則回傳空字串）。
 * @param {string[] | undefined} names
 * @param {number} classIndex
 * @returns {string}
 */
function resolveInterfaceLabelName(names, classIndex) {
  const v = names?.[classIndex];
  if (v == null || String(v).trim() === "") return "";
  return String(v).trim();
}

/**
 * 將圖片以 letterbox 繪製至 canvas（灰邊 114），與 YOLO 常見前處理一致。
 * @param {HTMLCanvasElement} canvas
 * @param {CanvasImageSource} img
 * @param {number} [targetSize]
 * @returns {{ scale: number, offsetX: number, offsetY: number, scaledWidth: number, scaledHeight: number, targetSize: number }}
 */
function paintLetterboxToCanvas(canvas, img, targetSize = PLAYGROUND_INPUT_SIZE) {
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return { scale: 1, offsetX: 0, offsetY: 0, scaledWidth: targetSize, scaledHeight: targetSize, targetSize };
  }
  const scale = Math.min(targetSize / img.width, targetSize / img.height);
  const scaledWidth = img.width * scale;
  const scaledHeight = img.height * scale;
  const offsetX = (targetSize - scaledWidth) / 2;
  const offsetY = (targetSize - scaledHeight) / 2;
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);
  return { scale, offsetX, offsetY, scaledWidth, scaledHeight, targetSize };
}

/**
 * 由預覽 URL 重繪 letterbox（推論後先呼叫，避免多次疊加框線）。
 * @param {string | null} imagePreviewUrl
 * @param {{ current: HTMLCanvasElement | null }} canvasRef
 * @returns {Promise<void>}
 */
function redrawLetterboxFromPreviewUrl(imagePreviewUrl, canvasRef) {
  return new Promise((resolve) => {
    if (!imagePreviewUrl || !canvasRef?.current) {
      resolve();
      return;
    }
    const img = new Image();
    img.onload = () => {
      paintLetterboxToCanvas(canvasRef.current, img);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = imagePreviewUrl;
  });
}

/**
 * 解析類 YOLOv8「單輸出」張量（無完整 NMS，僅供 Playground 示意）。
 * 支援 [1, 屬性數, 錨點數] 或 [1, 錨點數, 屬性數]（屬性 = 4 + 類別數）。
 * @param {{ shape?: number[], data?: number[] }} output
 * @param {{ threshold?: number, maxOut?: number }} [opts]
 * @returns {{ detections: Array<{ anchorIndex: number, classIndex: number, confidence: number, cx: number, cy: number, w: number, h: number }>, layout: string }}
 */
function parseYoloLikeOutput0(output, opts = {}) {
  const threshold = opts.threshold ?? 0.25;
  const maxOut = opts.maxOut ?? 48;
  const detections = [];
  if (!output?.shape || output.shape.length !== 3 || !Array.isArray(output.data)) {
    return { detections, layout: "invalid" };
  }
  const data = output.data;
  const [, d1, d2] = output.shape;

  /** 每個候選：前 4 維為框，其餘為各類別分數。 */
  const collectFromColumns = (numAttrs, numCols) => {
    for (let col = 0; col < numCols; col++) {
      let maxConf = 0;
      let bestClassIndex = -1;
      for (let row = 4; row < numAttrs; row++) {
        const val = data[row * numCols + col];
        if (val > maxConf) {
          maxConf = val;
          bestClassIndex = row - 4;
        }
      }
      if (maxConf <= threshold) continue;
      const cx = data[0 * numCols + col];
      const cy = data[1 * numCols + col];
      const w = data[2 * numCols + col];
      const h = data[3 * numCols + col];
      detections.push({
        anchorIndex: col,
        classIndex: bestClassIndex,
        confidence: maxConf,
        cx,
        cy,
        w,
        h,
      });
    }
  };

  /** 每列一個候選：[cx, cy, w, h, cls0, cls1, ...] */
  const collectFromRows = (numRows, numAttrs) => {
    for (let row = 0; row < numRows; row++) {
      const base = row * numAttrs;
      let maxConf = 0;
      let bestClassIndex = -1;
      for (let c = 4; c < numAttrs; c++) {
        const val = data[base + c];
        if (val > maxConf) {
          maxConf = val;
          bestClassIndex = c - 4;
        }
      }
      if (maxConf <= threshold) continue;
      detections.push({
        anchorIndex: row,
        classIndex: bestClassIndex,
        confidence: maxConf,
        cx: data[base + 0],
        cy: data[base + 1],
        w: data[base + 2],
        h: data[base + 3],
      });
    }
  };

  if (d1 >= 4 && d1 <= 4096 && d2 > d1) {
    collectFromColumns(d1, d2);
    detections.sort((a, b) => b.confidence - a.confidence);
    return { detections: detections.slice(0, maxOut), layout: "[1, attrs, anchors]" };
  }
  if (d2 >= 4 && d2 <= 4096 && d1 > d2) {
    collectFromRows(d1, d2);
    detections.sort((a, b) => b.confidence - a.confidence);
    return { detections: detections.slice(0, maxOut), layout: "[1, anchors, attrs]" };
  }
  return { detections, layout: "unsupported_shape" };
}

/**
 * 在 letterbox 座標（與 640 輸入對齊）上繪製偵測框與標籤。
 * @param {CanvasRenderingContext2D} ctx
 * @param {ReturnType<typeof parseYoloLikeOutput0>["detections"]} detections
 * @param {string[]} [interfaceClassNames] 與專案 Labeling Interface 中 Label 順序一致之名稱
 */
function drawDetectionsOnCtx(ctx, detections, interfaceClassNames = []) {
  ctx.strokeStyle = "#4CAF50";
  ctx.lineWidth = 3;
  ctx.font = "16px Arial";
  for (const d of detections) {
    const x = d.cx - d.w / 2;
    const y = d.cy - d.h / 2;
    ctx.strokeRect(x, y, d.w, d.h);
    const labelName = resolveInterfaceLabelName(interfaceClassNames, d.classIndex);
    const labelPart = labelName ? `${labelName} ` : "";
    const text = `${labelPart}#${d.classIndex} 信心 ${d.confidence.toFixed(3)} (${x.toFixed(0)},${y.toFixed(0)}) ${d.w.toFixed(0)}×${d.h.toFixed(0)}`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(76, 175, 80, 0.85)";
    ctx.fillRect(x, y > 20 ? y - 20 : y, textWidth + 8, 20);
    ctx.fillStyle = "white";
    ctx.fillText(text, x + 4, y > 20 ? y - 4 : y + 16);
  }
}

const SEMANTIC_OVERLAY_COLORS = [
  [255, 99, 71, 120],
  [65, 105, 225, 120],
  [50, 205, 50, 120],
  [255, 165, 0, 120],
  [186, 85, 211, 120],
  [64, 224, 208, 120],
  [255, 105, 180, 120],
  [154, 205, 50, 120],
];

/**
 * 是否像語意分割的空間網格（排除偵測張量 [1, attrs, anchors]）。
 * @param {number} h
 * @param {number} w
 * @returns {boolean}
 */
function isLikelySemanticSpatialGrid(h, w) {
  if (!Number.isFinite(h) || !Number.isFinite(w) || h < 8 || w < 8) return false;
  const ratio = Math.max(h, w) / Math.min(h, w);
  return ratio <= 4;
}

/**
 * 對 NCHW logits 做 per-pixel argmax。
 * @param {number[]} data
 * @param {number} numClasses
 * @param {number} height
 * @param {number} width
 * @returns {Int32Array}
 */
function argmaxClassMapNCHW(data, numClasses, height, width) {
  const plane = height * width;
  const classMap = new Int32Array(plane);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = y * width + x;
      let bestClass = 0;
      let bestVal = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const val = Number(data[c * plane + offset]);
        if (val > bestVal) {
          bestVal = val;
          bestClass = c;
        }
      }
      classMap[offset] = bestClass;
    }
  }
  return classMap;
}

/**
 * 對 NHWC logits 做 per-pixel argmax。
 * @param {number[]} data
 * @param {number} height
 * @param {number} width
 * @param {number} numClasses
 * @returns {Int32Array}
 */
function argmaxClassMapNHWC(data, height, width, numClasses) {
  const classMap = new Int32Array(height * width);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * numClasses;
      let bestClass = 0;
      let bestVal = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const val = Number(data[base + c]);
        if (val > bestVal) {
          bestVal = val;
          bestClass = c;
        }
      }
      classMap[y * width + x] = bestClass;
    }
  }
  return classMap;
}

/**
 * 將整數或機率值轉成 class id。
 * @param {number[]} data
 * @returns {Int32Array}
 */
function toIntClassMap(data) {
  return Int32Array.from(data.map((v) => Math.round(Number(v))));
}

/**
 * 解析語意分割輸出：class map 或 per-class logits。
 * 支援 [H,W]、[1,H,W]、[C,H,W]、[1,C,H,W]、[1,H,W,C]。
 * @param {{ shape?: number[], data?: number[] }} output
 * @returns {{ classMap: Int32Array, width: number, height: number, layout: string } | null}
 */
function parseSemanticClassMapOutput(output) {
  if (!output?.shape?.length || !Array.isArray(output.data)) return null;
  const { shape, data } = output;
  const expected = shape.reduce((a, b) => a * b, 1);
  if (expected !== data.length || expected === 0) return null;

  if (shape.length === 2) {
    const [height, width] = shape;
    if (!isLikelySemanticSpatialGrid(height, width)) return null;
    return {
      classMap: toIntClassMap(data),
      width,
      height,
      layout: `[${height}, ${width}]`,
    };
  }

  if (shape.length === 3) {
    const [d0, d1, d2] = shape;
    if (d0 === 1 && isLikelySemanticSpatialGrid(d1, d2)) {
      return {
        classMap: toIntClassMap(data),
        width: d2,
        height: d1,
        layout: `[1, ${d1}, ${d2}]`,
      };
    }
    if (d0 <= 512 && isLikelySemanticSpatialGrid(d1, d2)) {
      return {
        classMap: argmaxClassMapNCHW(data, d0, d1, d2),
        width: d2,
        height: d1,
        layout: `[${d0}, ${d1}, ${d2}] logits→argmax`,
      };
    }
    if (isLikelySemanticSpatialGrid(d0, d1) && d2 <= 512) {
      return {
        classMap: argmaxClassMapNHWC(data, d0, d1, d2),
        width: d1,
        height: d0,
        layout: `[${d0}, ${d1}, ${d2}] NHWC logits→argmax`,
      };
    }
    return null;
  }

  if (shape.length === 4) {
    const [n, d1, d2, d3] = shape;
    if (n !== 1) return null;
    if (d1 <= 512 && isLikelySemanticSpatialGrid(d2, d3)) {
      return {
        classMap: argmaxClassMapNCHW(data, d1, d2, d3),
        width: d3,
        height: d2,
        layout: `[1, ${d1}, ${d2}, ${d3}] logits→argmax`,
      };
    }
    if (isLikelySemanticSpatialGrid(d1, d2) && d3 <= 512) {
      return {
        classMap: argmaxClassMapNHWC(data, d1, d2, d3),
        width: d2,
        height: d1,
        layout: `[1, ${d1}, ${d2}, ${d3}] NHWC logits→argmax`,
      };
    }
  }

  return null;
}

/**
 * 從 Triton 多個輸出中解析語意分割 class map。
 * @param {Array<{ name?: string, shape?: number[], data?: number[] }>} outputs
 * @returns {{ parsed: ReturnType<typeof parseSemanticClassMapOutput>, outputName: string | null }}
 */
function parseSemanticFromOutputs(outputs) {
  if (!Array.isArray(outputs)) return { parsed: null, outputName: null };
  const preferred = outputs.find((o) => o.name === "output0") ?? outputs[0];
  const ordered = preferred ? [preferred, ...outputs.filter((o) => o !== preferred)] : outputs;
  for (const out of ordered) {
    const parsed = parseSemanticClassMapOutput(out);
    if (parsed) return { parsed, outputName: out.name ?? null };
  }
  return { parsed: null, outputName: preferred?.name ?? null };
}

/**
 * 在 canvas 上疊加語意分割色塊（letterbox 640×640 空間）。
 * @param {CanvasRenderingContext2D} ctx
 * @param {Int32Array} classMap
 * @param {number} width
 * @param {number} height
 * @param {string[]} [interfaceClassNames]
 */
function drawSemanticOverlayOnCtx(ctx, classMap, width, height, interfaceClassNames = []) {
  const target = ctx.canvas.width;
  const scale = Math.min(target / width, target / height);
  const scaledW = width * scale;
  const scaledH = height * scale;
  const offsetX = (target - scaledW) / 2;
  const offsetY = (target - scaledH) / 2;

  const overlay = document.createElement("canvas");
  overlay.width = target;
  overlay.height = target;
  const octx = overlay.getContext("2d");
  if (!octx) return;

  const imgData = octx.createImageData(target, target);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const cls = classMap[y * width + x];
      if (!cls || cls === 255) continue;
      const color = SEMANTIC_OVERLAY_COLORS[(cls - 1) % SEMANTIC_OVERLAY_COLORS.length];
      const tx0 = Math.floor(offsetX + x * scale);
      const ty0 = Math.floor(offsetY + y * scale);
      const tx1 = Math.min(target, Math.ceil(offsetX + (x + 1) * scale));
      const ty1 = Math.min(target, Math.ceil(offsetY + (y + 1) * scale));
      for (let ty = ty0; ty < ty1; ty++) {
        for (let tx = tx0; tx < tx1; tx++) {
          if (tx < 0 || ty < 0 || tx >= target || ty >= target) continue;
          const idx = (ty * target + tx) * 4;
          imgData.data[idx] = color[0];
          imgData.data[idx + 1] = color[1];
          imgData.data[idx + 2] = color[2];
          imgData.data[idx + 3] = color[3];
        }
      }
    }
  }
  octx.putImageData(imgData, 0, 0);
  ctx.drawImage(overlay, 0, 0);

  const present = new Set(Array.from(classMap).filter((v) => v > 0 && v !== 255));
  const legendY = target - 8 - present.size * 18;
  ctx.font = "13px Arial";
  let row = 0;
  for (const cls of present) {
    const color = SEMANTIC_OVERLAY_COLORS[(cls - 1) % SEMANTIC_OVERLAY_COLORS.length];
    const name = resolveInterfaceLabelName(interfaceClassNames, cls - 1) || `class ${cls}`;
    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.95)`;
    ctx.fillRect(8, legendY + row * 18, 12, 12);
    ctx.fillStyle = "white";
    ctx.fillText(`${name} (#${cls})`, 24, legendY + row * 18 + 11);
    row += 1;
  }
}

/**
 * 對 logits 做數值穩定的 softmax。
 * @param {number[]} values
 * @returns {number[]}
 */
function softmaxValues(values) {
  if (!values.length) return [];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * 判斷是否已近似機率分佈（非負且總和約為 1）。
 * @param {number[]} values
 * @returns {boolean}
 */
function looksLikeProbabilityVector(values) {
  if (!values.length) return false;
  if (values.some((v) => v < -1e-6 || v > 1 + 1e-6)) return false;
  const s = values.reduce((a, v) => a + v, 0);
  return Math.abs(s - 1) < 0.02;
}

/**
 * 從單一 Triton 輸出張量取出「每類一個分數」的一維向量（batch=1）。
 * @param {{ shape?: number[], data?: number[] }} output
 * @returns {{ values: number[], layout: string } | null}
 */
function extractClassificationVector(output) {
  if (!output?.shape?.length || !Array.isArray(output.data)) return null;
  const { shape, data } = output;
  const n = data.length;
  const expected = shape.reduce((a, b) => a * b, 1);
  if (n !== expected || n === 0) return null;

  if (shape.length === 1) {
    return { values: [...data], layout: `[${shape[0]}]` };
  }
  if (shape.length === 2 && shape[0] === 1) {
    return { values: [...data], layout: `[1, ${shape[1]}]` };
  }
  if (shape.length === 2 && shape[1] === 1) {
    return { values: [...data], layout: `[${shape[0]}, 1]` };
  }
  if (shape.length === 3 && shape[0] === 1 && shape[2] === 1) {
    return { values: [...data], layout: `[1, ${shape[1]}, 1]` };
  }
  if (shape.length === 4 && shape[0] === 1 && shape[2] === 1 && shape[3] === 1) {
    return { values: [...data], layout: `[1, ${shape[1]}, 1, 1]` };
  }
  return null;
}

/**
 * 建立分類結果列：原始分數 + 機率（若非常態 logits 則經 softmax）。
 * @param {number[]} values
 * @returns {Array<{ classIndex: number, raw: number, probability: number }>}
 */
function buildClassificationRows(values) {
  const asProb = looksLikeProbabilityVector(values);
  const probs = asProb ? values : softmaxValues(values);
  return values.map((raw, classIndex) => ({
    classIndex,
    raw,
    probability: probs[classIndex],
  }));
}

/**
 * 在預覽圖左上角疊加分類 Top-K 文字（補足表格之外的快速視覺提示）。
 * @param {CanvasRenderingContext2D} ctx
 * @param {ReturnType<typeof buildClassificationRows>} rows
 * @param {number} [topK]
 * @param {string[]} [interfaceClassNames] 與 Choices 順序一致之標籤名稱
 */
function drawClassificationSummaryOnCtx(ctx, rows, topK = 8, interfaceClassNames = []) {
  if (!rows.length) return;
  const sorted = [...rows].sort((a, b) => b.probability - a.probability);
  const pick = sorted.slice(0, topK);
  const lineH = 20;
  const pad = 8;
  const w = Math.min(560, ctx.canvas.width - 16);
  const h = pad * 2 + pick.length * lineH;
  ctx.fillStyle = "rgba(17, 24, 39, 0.82)";
  ctx.fillRect(pad, pad, w, h);
  ctx.font = "14px Arial";
  ctx.fillStyle = "#f9fafb";
  pick.forEach((r, i) => {
    const name = resolveInterfaceLabelName(interfaceClassNames, r.classIndex);
    const namePart = name ? `${name} ` : "";
    const line = `#${i + 1} ${namePart}[#${r.classIndex}] p=${r.probability.toFixed(4)} raw=${Number(r.raw).toFixed(4)}`;
    ctx.fillText(line, pad + 8, pad + 18 + i * lineH);
  });
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
  /** Triton HTTP 完整基底 URL（含埠號）；空值表示使用伺服器 TRITON_SERVER_URL。 */
  const [tritonServerUrl, setTritonServerUrl] = useState(() => readSavedPlaygroundState().tritonServerUrl);
  /** Triton 連線驗證結果（按「驗證連線」後更新）。 */
  const [tritonVerifyLoading, setTritonVerifyLoading] = useState(false);
  const [tritonVerifyResult, setTritonVerifyResult] = useState(null);
  /** 與後端模型任務對齊：偵測畫框；分類顯示每類分數。 */
  const [taskType, setTaskType] = useState(normalizePlaygroundTaskType(savedState.taskType));
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState({ status: null, summary: null, error: null });
  /** 最近一次推論解析出的偵測列（模型 letterbox 640 座標空間）。 */
  const [detectionRows, setDetectionRows] = useState([]);
  /** 分類模式：每個類別的原始輸出與機率。 */
  const [classificationRows, setClassificationRows] = useState([]);
  const [outputParseHint, setOutputParseHint] = useState(null);
  /** 由專案 label_config 推斷結果之說明（Settings → Labeling Interface）。 */
  const [labelInterfaceHint, setLabelInterfaceHint] = useState(null);
  /** 是否手動選過任務類型（僅影響 UI 提示與是否接受介面自動覆寫）。 */
  const [taskTypeUserOverridden, setTaskTypeUserOverridden] = useState(false);
  /** Labeling Interface 中 Choices 的 value 順序，對應分類模型輸出索引。 */
  const [interfaceClassificationLabels, setInterfaceClassificationLabels] = useState([]);
  /** Labeling Interface 中 RectangleLabels 等之 Label value 順序，對應偵測類別索引。 */
  const [interfaceDetectionLabels, setInterfaceDetectionLabels] = useState([]);
  /** 供專案下拉選單使用（顯示 title，value 為 id）。 */
  const [projects, setProjects] = useState([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState(null);

  // 預覽與 Tensor 處理使用 ref
  const canvasRef = useRef(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processingImage, setProcessingImage] = useState(false);

  // 不要將 Tensor 資料存進 useState，避免渲染卡頓
  const tensorPayloadRef = useRef(null);

  /**
   * 是否曾手動變更任務類型；為 true 時不再依 label_config 覆寫。
   * @type {React.MutableRefObject<boolean>}
   */
  const taskTypeManualRef = useRef(false);
  /** 上次已套用介面推斷的專案 pk，換專案時重設手動旗標。 */
  const lastProjectPkForInferenceRef = useRef(null);
  /** 與 `projects` 同步，供載入模型列表 effect 讀取專案標題而不增加該 effect 依賴。 */
  const projectsRef = useRef([]);

  const projectSelectLabelsById = useMemo(() => buildProjectSelectLabelsById(projects), [projects]);

  /** 目前選定專案之標題（用於預設模型名稱等）。 */
  const selectedProjectTitle = useMemo(() => {
    const row = projects.find((p) => String(p.id) === String(projectId));
    const t = row?.title != null ? String(row.title).trim() : "";
    return t;
  }, [projects, projectId]);

  /** 與目前輸入之 model_name 對應的已部署模型後設（若有）。 */
  const selectedDeployedModelMeta = useMemo(
    () => models.find((m) => m.model_name === modelName) ?? null,
    [models, modelName],
  );

  const playgroundInputSize = useMemo(
    () => playgroundInputSizeForTask(taskType, selectedDeployedModelMeta?.imgsz),
    [taskType, selectedDeployedModelMeta],
  );

  useEffect(() => {
    projectsRef.current = projects;
  }, [projects]);

  /** 進入頁面時載入專案清單（顯示名稱用）。 */
  useEffect(() => {
    let cancelled = false;
    setProjectsLoading(true);
    setProjectsError(null);
    api
      .callApi("projects", {
        params: {
          page: 1,
          page_size: 500,
          include: ["id", "title"].join(","),
        },
        errorFilter: () => true,
      })
      .then((data) => {
        if (cancelled) return;
        setProjects(Array.isArray(data?.results) ? data.results : []);
      })
      .catch((err) => {
        if (!cancelled) {
          setProjects([]);
          setProjectsError(err?.message || "無法載入專案列表");
        }
      })
      .finally(() => {
        if (!cancelled) setProjectsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [api]);

  /**
   * 可選查詢參數：轉發至指定 Triton 基底（須與後端可連線）。
   */
  const tritonProxyQueryParams = useMemo(() => {
    const u = tritonServerUrl.trim();
    return u ? { triton_url: u } : {};
  }, [tritonServerUrl]);

  /**
   * 請後端向 Triton `/v2/health` 探測連線（空位址則檢查伺服器預設）。
   */
  const handleVerifyTriton = useCallback(async () => {
    setTritonVerifyResult(null);
    if (!projectId.trim()) {
      setTritonVerifyResult({ level: "error", text: "請先選擇專案。" });
      return;
    }
    setTritonVerifyLoading(true);
    try {
      const r = await verifyTritonConnection(api, projectId, tritonServerUrl);
      setTritonVerifyResult({ level: r.level, text: r.message });
    } finally {
      setTritonVerifyLoading(false);
    }
  }, [api, projectId, tritonServerUrl]);

  useEffect(() => {
    setTritonVerifyResult(null);
  }, [tritonServerUrl]);

  /** Label Studio 代理 Triton 推論的 REST URL（隨專案 ID 更新）。 */
  const inferProxyUrl = useMemo(() => {
    const hostname = typeof window !== "undefined" ? (window.APP_SETTINGS?.hostname ?? "") : "";
    return buildTritonInferProxyUrl(hostname, projectId);
  }, [projectId]);

  /** 供「程式呼叫」區塊顯示的 fetch／curl／Python 範例。 */
  const apiExampleSnippets = useMemo(
    () =>
      buildPlaygroundApiExampleSnippets({
        inferUrl: inferProxyUrl,
        modelName,
        apiKey,
        sessionId: readSessionIdFromCookie(),
        tritonServerUrl,
        inputSize: playgroundInputSize,
      }),
    [inferProxyUrl, modelName, apiKey, tritonServerUrl, playgroundInputSize],
  );

  useEffect(() => {
    const saved = readSavedPlaygroundState();
    persistPlaygroundState({
      projectId,
      modelName,
      apiKey,
      taskType,
      tritonServerUrl,
      tritonMetricsUrl: saved.tritonMetricsUrl,
    });
    if (tritonServerUrl.trim()) {
      persistTritonUrlFields({ tritonServerUrl: normalizeTritonUrl(tritonServerUrl) });
    }
  }, [projectId, modelName, apiKey, taskType, tritonServerUrl]);

  useEffect(() => {
    const pk = Number.parseInt(projectId.trim(), 10);
    if (!Number.isFinite(pk) || pk < 1) {
      setLabelInterfaceHint(null);
      setInterfaceClassificationLabels([]);
      setInterfaceDetectionLabels([]);
      lastProjectPkForInferenceRef.current = null;
      return;
    }
    if (lastProjectPkForInferenceRef.current !== pk) {
      lastProjectPkForInferenceRef.current = pk;
      taskTypeManualRef.current = false;
      setTaskTypeUserOverridden(false);
    }

    let cancelled = false;
    api
      .callApi("project", {
        params: { pk },
        errorFilter: () => true,
      })
      .then((proj) => {
        if (cancelled || !proj) return;
        setInterfaceClassificationLabels(extractClassificationChoiceLabels(proj.label_config));
        setInterfaceDetectionLabels(extractSpatialControlLabelNames(proj.label_config));
        const inferred = inferPlaygroundTaskTypeFromLabelConfig(proj.label_config);
        if (inferred) {
          setLabelInterfaceHint(`依 Labeling Interface：${inferred.detail}`);
          if (!taskTypeManualRef.current) {
            setTaskType(inferred.taskType);
          }
        } else {
          setLabelInterfaceHint("無法由 Labeling Interface 自動判斷，請手動選擇任務類型");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLabelInterfaceHint("無法讀取專案設定，故無法依介面自動判斷任務類型");
          setInterfaceClassificationLabels([]);
          setInterfaceDetectionLabels([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [api, projectId]);

  useEffect(() => {
    if (taskTypeManualRef.current) return;
    const fromModel = selectedDeployedModelMeta?.task_type;
    if (fromModel) {
      setTaskType(normalizePlaygroundTaskType(fromModel));
    }
  }, [selectedDeployedModelMeta]);

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
        params: { pk: projectId, ...tritonProxyQueryParams },
        errorFilter: () => true,
      })
      .then((res) => {
        if (cancelled) return;
        const nextModels = Array.isArray(res?.models) ? res.models : [];
        setModels(nextModels);

        setModelName((prev) => {
          const inList = nextModels.some((m) => m.model_name === prev);
          if (inList) return prev;
          if (nextModels.length === 0) return "";
          const list = projectsRef.current;
          const row = list.find((p) => String(p.id) === String(projectId));
          const title = row?.title != null ? String(row.title).trim() : "";
          const byTitle = title ? nextModels.find((m) => m.model_name === title) : undefined;
          if (byTitle) return byTitle.model_name;
          return nextModels[0].model_name;
        });
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
  }, [api, projectId, tritonProxyQueryParams]);

  // 處理圖片上傳與預先調整大小轉換為 Tensor
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setProcessingImage(true);
    const imageUrl = URL.createObjectURL(file);
    setImagePreview(imageUrl);
    tensorPayloadRef.current = null;
    setResponse({ status: null, summary: null, error: null });
    setDetectionRows([]);
    setClassificationRows([]);
    setOutputParseHint(null);

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        setProcessingImage(false);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setProcessingImage(false);
        return;
      }
      paintLetterboxToCanvas(canvas, img);

      // 取得影像像素，並轉為 [1, 3, 640, 640] 的一維陣列 (FP32, normalized to 0-1)
      const TARGET_SIZE = playgroundInputSize;
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
    setDetectionRows([]);
    setClassificationRows([]);
    setOutputParseHint(null);
    try {
      if (!projectId.trim()) {
        setResponse({ status: null, summary: null, error: "請先選擇專案" });
        return;
      }
      if (!modelName.trim()) {
        setResponse({ status: null, summary: null, error: "請填寫模型名稱（可從建議清單選取或自行輸入）" });
        return;
      }
      if (!tensorPayloadRef.current) {
        setResponse({ status: null, summary: null, error: "請先上傳圖片" });
        return;
      }

      const res = await api.callApi("trainingTritonInfer", {
        params: { pk: projectId, ephemeral: true, ...tritonProxyQueryParams },
        body: { model_name: modelName, api_key: apiKey, ...tensorPayloadRef.current },
        errorFilter: () => true,
      });

      const lifecyclePrefix = formatTritonLifecycleSummary(res?.lifecycle);
      let summaryStr = lifecyclePrefix;
      if (res?.body?.outputs) {
        summaryStr = res.body.outputs
          .map(
            (o) =>
              `輸出節點: ${o.name}, 資料型別: ${o.datatype ?? "?"}, 形狀: [${(o.shape || []).join(", ")}], 元素數: ${
                Array.isArray(o.data) ? o.data.length : "?"
              }`,
          )
          .join("\n");
      }

      // res.status_code：Triton 回傳的 HTTP status（由 Django proxy 放入 JSON body）
      // res.status：api.callApi 錯誤包裹層的 HTTP status（Django 層級的非 2xx 回應）
      // 優先以 status_code（Triton 實際狀態）顯示，回退到 HTTP 包裹層 status，再回退 200
      const displayStatus = res?.status_code ?? res?.status ?? 200;
      const isOk = Boolean(res?.ok);
      if (!isOk && res?.detail) {
        summaryStr += `${res.detail}\n`;
      }
      setResponse({
        status: displayStatus,
        statusText: isOk ? "OK" : "ERROR",
        summary: summaryStr || `${JSON.stringify(res?.body ?? res).substring(0, 200)}...`,
        ok: isOk,
      });

      if (res?.ok && imagePreview && canvasRef.current && res?.body?.outputs?.length) {
        await redrawLetterboxFromPreviewUrl(imagePreview, canvasRef);
        const ctx = canvasRef.current.getContext("2d");
        if (ctx && taskType === "classification") {
          const primary = res.body.outputs.find((o) => o.name === "output0") ?? res.body.outputs[0];
          const extracted = extractClassificationVector(primary);
          if (extracted) {
            const rows = buildClassificationRows(extracted.values);
            setClassificationRows(rows);
            setOutputParseHint(`分類輸出 ${extracted.layout}（${rows.length} 類）`);
            drawClassificationSummaryOnCtx(ctx, rows, 8, interfaceClassificationLabels);
          } else {
            setOutputParseHint("分類：無法從主要輸出解析一維類別向量，請確認 shape（如 [1,C]）與輸出名稱");
          }
        } else if (ctx && taskType === "semantic_segmentation") {
          const { parsed, outputName } = parseSemanticFromOutputs(res.body.outputs);
          if (parsed) {
            const unique = new Set(Array.from(parsed.classMap).filter((v) => v > 0 && v !== 255));
            const outLabel = outputName ? ` · ${outputName}` : "";
            setOutputParseHint(`語意分割 ${parsed.layout}${outLabel} · ${unique.size} 個類別`);
            setDetectionRows([]);
            drawSemanticOverlayOnCtx(ctx, parsed.classMap, parsed.width, parsed.height, interfaceDetectionLabels);
          } else {
            const primary = res.body.outputs.find((o) => o.name === "output0") ?? res.body.outputs[0];
            const shapeStr = primary?.shape?.length ? `[${primary.shape.join(", ")}]` : "未知";
            setOutputParseHint(
              `語意分割：無法解析 class map（實際 shape ${shapeStr}；支援 [H,W]、[1,H,W]、[1,C,H,W] logits 等）`,
            );
          }
        } else if (ctx && isSpatialYoloTaskType(taskType)) {
          const output0 = res.body.outputs.find((o) => o.name === "output0");
          if (output0) {
            const { detections, layout } = parseYoloLikeOutput0(output0);
            const hintPrefix =
              taskType === "segmentation"
                ? "實例分割（以偵測張量示意解析）"
                : taskType === "pose"
                  ? "姿態（以偵測張量示意解析，完整關鍵點需專用後處理）"
                  : taskType === "obb"
                    ? "OBB（以偵測張量示意解析，旋轉角需專用後處理）"
                    : playgroundTaskTypeLabel(taskType);
            setOutputParseHint(`${hintPrefix}：${layout}`);
            setDetectionRows(detections);
            if (detections.length > 0) {
              drawDetectionsOnCtx(ctx, detections, interfaceDetectionLabels);
            } else if (layout !== "invalid" && layout !== "unsupported_shape") {
              ctx.fillStyle = "rgba(220, 38, 38, 0.95)";
              ctx.font = "16px Arial";
              ctx.fillText("未偵測到超過門檻的物體（仍可依下方表格／輸出摘要檢查張量）", 10, 24);
            }
          } else {
            setOutputParseHint(`${playgroundTaskTypeLabel(taskType)}：找不到名為 output0 的輸出`);
          }
        }
      }
    } catch (err) {
      setResponse({
        status: null,
        summary: null,
        error: err.message || "請求失敗",
      });
    } finally {
      setLoading(false);
    }
  }, [
    api,
    modelName,
    projectId,
    apiKey,
    imagePreview,
    taskType,
    interfaceClassificationLabels,
    interfaceDetectionLabels,
    tritonProxyQueryParams,
    playgroundInputSize,
  ]);

  return (
    <section className={rootClass.toClassName()}>
      <p className={rootClass.elem("intro").toClassName()}>
        選擇已部署之模型，並上傳圖片來測試其效能與結果。測試時會自動檢查 Triton 是否已載入模型，必要時先載入，完成後釋放記憶體。
      </p>

      <div className={rootClass.elem("api-docs").toClassName()}>
        <div className={rootClass.elem("section-title").toClassName()}>以程式引用此模型（REST API）</div>
        <Typography variant="body" size="small" className="text-neutral-content-subtle mb-tight block">
          後端會校驗該 <code>model_name</code> 是否為此專案已部署之模型，並轉送 Triton{" "}
          <code>{`/v2/models/<name>/infer`}</code>。請求方法為 <code>POST</code>，內容類型 <code>application/json</code>
          ；需具專案檢視權限（與網頁相同之登入狀態或對應 Cookie／Token）。若專案設定了 Triton 安全金鑰，JSON 內須含{" "}
          <code>api_key</code>。可選查詢參數 <code>?timeout=60</code>（秒）調整逾時；<code>?triton_url=...</code>{" "}
          指定後端轉發之 Triton HTTP 基底（與下方「Triton 服務位址」一致）。
        </Typography>
        {inferProxyUrl ? (
          <Typography variant="body" size="small" className="mb-tight block">
            <span className="text-neutral-content-subtle">端點：</span>
            <code className={rootClass.elem("api-docs-url").toClassName()}>{inferProxyUrl}</code>
          </Typography>
        ) : (
          <Typography variant="body" size="small" className="text-neutral-content-subtle mb-tight block">
            {/* 請先填寫專案名稱以顯示完整端點 URL。 */}
          </Typography>
        )}
        <details className={rootClass.elem("api-docs-details").toClassName()}>
          <summary className={rootClass.elem("api-docs-summary").toClassName()}>
            <span>JavaScript（fetch）</span>
            <CopyCodeButton
              text={apiExampleSnippets.js}
              className={rootClass.elem("api-docs-copy-btn").toClassName()}
            />
          </summary>
          <pre className={rootClass.elem("api-docs-pre").toClassName()}>{apiExampleSnippets.js}</pre>
        </details>
        <details className={rootClass.elem("api-docs-details").toClassName()}>
          <summary className={rootClass.elem("api-docs-summary").toClassName()}>
            <span>cURL</span>
            <CopyCodeButton
              text={apiExampleSnippets.curl}
              className={rootClass.elem("api-docs-copy-btn").toClassName()}
            />
          </summary>
          <pre className={rootClass.elem("api-docs-pre").toClassName()}>{apiExampleSnippets.curl}</pre>
        </details>
        <details className={rootClass.elem("api-docs-details").toClassName()}>
          <summary className={rootClass.elem("api-docs-summary").toClassName()}>
            <span>Python（requests）</span>
            <CopyCodeButton
              text={apiExampleSnippets.py}
              className={rootClass.elem("api-docs-copy-btn").toClassName()}
            />
          </summary>
          <pre className={rootClass.elem("api-docs-pre").toClassName()}>{apiExampleSnippets.py}</pre>
        </details>
      </div>

      <div className={rootClass.elem("form").toClassName()}>
        <div className={rootClass.elem("form-row").toClassName()}>
          <div className={rootClass.elem("field").toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()} htmlFor="playground-project-select">
              專案名稱
            </label>
            <select
              id="playground-project-select"
              className={rootClass.elem("select").toClassName()}
              value={projectId}
              onChange={(e) => setProjectId(e.target.value)}
              disabled={projectsLoading}
            >
              <option value="">{projectsLoading ? "載入專案列表中…" : "請選擇專案"}</option>
              {!projectsLoading &&
                projects.map((p) => (
                  <option key={p.id} value={String(p.id)}>
                    {projectSelectLabelsById.get(String(p.id)) ?? `專案 #${p.id}`}
                  </option>
                ))}
            </select>
            {projectId.trim() ? (
              <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
                API 路徑使用專案 ID：<code>{projectId.trim()}</code>
              </Typography>
            ) : null}
            {projectsError && <div className={rootClass.elem("error").toClassName()}>{projectsError}</div>}
          </div>
          <div className={rootClass.elem("field").toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()} htmlFor="playground-api-key">
              Triton 安全金鑰（API Key，選填）
            </label>
            <input
              id="playground-api-key"
              type="password"
              className={rootClass.elem("input").toClassName()}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="專案有設定 Triton 金鑰時請填寫"
              autoComplete="off"
            />
          </div>
        </div>

        <div className={rootClass.elem("form-row").toClassName()}>
          <div className={rootClass.elem("field").toClassName()}>
            <label className={rootClass.elem("field-label").toClassName()} htmlFor="playground-triton-server-url">
              Triton 伺服器 URL（選填）
            </label>
            <input
              id="playground-triton-server-url"
              type="url"
              className={rootClass.elem("input").toClassName()}
              value={tritonServerUrl}
              onChange={(e) => setTritonServerUrl(e.target.value)}
              onBlur={() => {
                const next = normalizeTritonUrl(tritonServerUrl);
                if (next !== tritonServerUrl) setTritonServerUrl(next);
              }}
              placeholder="http://192.168.1.10:18000"
              autoComplete="off"
            />
            <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
              {tritonServerUrl.trim()
                ? `將使用：${normalizeTritonUrl(tritonServerUrl)}`
                : "留空則由伺服器 TRITON_SERVER_URL 決定。自訂時請輸入完整 URL（含埠號），例如 Docker 對外 http://主機:18000、本機 start-triton http://主機:8000。"}
            </Typography>
          </div>
        </div>

        <div className={rootClass.elem("form-row").toClassName()}>
          <Button
            type="button"
            variant="neutral"
            look="outlined"
            size="small"
            disabled={tritonVerifyLoading || !projectId.trim()}
            onClick={handleVerifyTriton}
          >
            {tritonVerifyLoading ? "驗證中…" : "驗證 Triton 連線"}
          </Button>
          {tritonVerifyResult ? (
            <Typography
              variant="body"
              size="small"
              style={{
                color:
                  tritonVerifyResult.level === "success"
                    ? "var(--color-positive-content, #166534)"
                    : tritonVerifyResult.level === "warning"
                      ? "var(--color-warning-content, #92400e)"
                      : "var(--color-negative-content, #991b1b)",
              }}
            >
              {tritonVerifyResult.text}
            </Typography>
          ) : (
            <Typography variant="body" size="small" className="text-neutral-content-subtle">
              送出推論前可先驗證；未填 Triton 位址時會檢查伺服器預設。
            </Typography>
          )}
        </div>

        <div className={rootClass.elem("field").toClassName()}>
          <label className={rootClass.elem("field-label").toClassName()} htmlFor="playground-model-select">
            已部署模型
          </label>
          <select
            id="playground-model-select"
            className={rootClass.elem("select").toClassName()}
            value={models.some((m) => m.model_name === modelName) ? modelName : ""}
            onChange={(e) => {
              const v = e.target.value;
              if (v) setModelName(v);
            }}
            disabled={modelsLoading || !projectId.trim() || models.length === 0}
          >
            <option value="">
              {modelsLoading ? "載入中…" : models.length === 0 ? "無已部署模型" : "自訂（請在下方輸入 model_name）"}
            </option>
            {!modelsLoading &&
              models.map((model) => (
                <option key={model.model_name} value={model.model_name}>
                  {model.run_name ? model.run_name : model.model_name}
                  {model.run_name ? `（${model.model_name}）` : ""}
                  {model.deployed_at ? `　部署於 ${new Date(model.deployed_at).toLocaleString()}` : ""}
                </option>
              ))}
          </select>
          <label className={rootClass.elem("field-label").toClassName()} htmlFor="playground-model-name-input">
            模型名稱（送請求用）
          </label>
          <input
            id="playground-model-name-input"
            type="text"
            className={rootClass.elem("input").toClassName()}
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            disabled={modelsLoading || !projectId.trim()}
            placeholder={
              selectedProjectTitle
                ? `可與專案標題相同（例如：${selectedProjectTitle}），或從上方清單選取`
                : "請先選擇專案；與 Triton model_name 一致"
            }
            autoComplete="off"
          />
          <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
            下拉會列出此專案<strong>全部</strong>已部署模型；實際呼叫 API 以上方「模型名稱」欄為準，可自行修改。
          </Typography>
          {selectedDeployedModelMeta?.deployed_at ? (
            <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
              {selectedDeployedModelMeta.run_name
                ? `「${selectedDeployedModelMeta.run_name}」（${selectedDeployedModelMeta.model_name}），部署時間：${new Date(selectedDeployedModelMeta.deployed_at).toLocaleString()}`
                : `此名稱對應已部署模型，部署時間：${new Date(selectedDeployedModelMeta.deployed_at).toLocaleString()}`}
            </Typography>
          ) : null}
          {!modelsLoading && projectId.trim() && models.length === 0 ? (
            <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
              此專案尚無已部署模型於清單中；若已知 Triton 名稱仍可手動輸入後測試。
            </Typography>
          ) : null}
          {modelsError && <div className={rootClass.elem("error").toClassName()}>{modelsError}</div>}
        </div>

        <div className={rootClass.elem("field").toClassName()}>
          <label className={rootClass.elem("field-label").toClassName()}>任務類型</label>
          <select
            className={rootClass.elem("select").toClassName()}
            value={taskType}
            onChange={(e) => {
              taskTypeManualRef.current = true;
              setTaskTypeUserOverridden(true);
              setTaskType(normalizePlaygroundTaskType(e.target.value));
              setDetectionRows([]);
              setClassificationRows([]);
              setOutputParseHint(null);
              tensorPayloadRef.current = null;
            }}
          >
            {Object.entries(PLAYGROUND_TASK_TYPES).map(([key, meta]) => (
              <option key={key} value={key}>
                {meta.label}
              </option>
            ))}
          </select>
          {selectedDeployedModelMeta?.task_type && (
            <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
              已部署模型任務：{playgroundTaskTypeLabel(selectedDeployedModelMeta.task_type)}
              {selectedDeployedModelMeta.imgsz ? ` · 輸入 ${selectedDeployedModelMeta.imgsz}px` : ""}
            </Typography>
          )}
          {labelInterfaceHint && (
            <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
              {labelInterfaceHint}
              {taskTypeUserOverridden ? "（目前已手動選擇任務類型）" : ""}
            </Typography>
          )}
          <Typography variant="body" size="small" className="text-neutral-content-subtle mt-tightest block">
            預設依專案 Settings → Labeling Interface 的設定自動選擇；若與實際模型不符請手動調整。分類模式優先讀取{" "}
            <code>output0</code>，若無則使用第一個輸出。
          </Typography>
        </div>

        <div className={rootClass.elem("upload-panel").toClassName()}>
          <label className={rootClass.elem("upload-label").toClassName()}>上傳圖片</label>
          <input type="file" accept="image/*" onChange={handleImageUpload} disabled={processingImage} />
          {processingImage && <span className="text-neutral-content-subtle ml-tight">運算中...</span>}

          <div className={rootClass.elem("canvas-wrap").toClassName()} style={{ display: imagePreview ? "block" : "none" }}>
            <canvas ref={canvasRef} className={rootClass.elem("canvas").toClassName()} />
            {outputParseHint && (
              <Typography
                variant="body"
                size="small"
                className="text-neutral-content-subtle mt-tight block"
                style={{ textAlign: "left" }}
              >
                輸出解析：{outputParseHint}
                {taskType === "semantic_segmentation"
                  ? `（class map 依 letterbox ${playgroundInputSize}×${playgroundInputSize} 疊加；解析度可低於輸入尺寸）`
                  : isSpatialYoloTaskType(taskType)
                    ? `（座標為 letterbox ${playgroundInputSize}×${playgroundInputSize} 模型空間；無 NMS，框可能重疊）`
                    : "（機率欄：若輸出已近似機率分佈則沿用；否則以 softmax(logits) 計算）"}
              </Typography>
            )}
            {isSpatialYoloTaskType(taskType) && detectionRows.length > 0 && (
              <div style={{ marginTop: "16px", overflowX: "auto", textAlign: "left" }}>
                <div className={rootClass.elem("section-title").toClassName()}>
                  {playgroundTaskTypeLabel(taskType)}數值（前 {detectionRows.length} 筆；標籤來自 Labeling Interface）
                </div>
                <table className={rootClass.elem("data-table").toClassName()}>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>錨點</th>
                      <th>索引</th>
                      <th>標籤（介面）</th>
                      <th>信心度</th>
                      <th>cx</th>
                      <th>cy</th>
                      <th>w</th>
                      <th>h</th>
                      <th>x1</th>
                      <th>y1</th>
                      <th>x2</th>
                      <th>y2</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detectionRows.map((row, i) => {
                      const x1 = row.cx - row.w / 2;
                      const y1 = row.cy - row.h / 2;
                      const x2 = row.cx + row.w / 2;
                      const y2 = row.cy + row.h / 2;
                      return (
                        <tr key={`${row.anchorIndex}-${row.classIndex}-${i}`}>
                          <td>{i + 1}</td>
                          <td>{row.anchorIndex}</td>
                          <td>{row.classIndex}</td>
                          <td>
                            {resolveInterfaceLabelName(interfaceDetectionLabels, row.classIndex) || (
                              <span className="text-neutral-content-subtle">—</span>
                            )}
                          </td>
                          <td>{row.confidence.toFixed(4)}</td>
                          <td>{Number(row.cx).toFixed(2)}</td>
                          <td>{Number(row.cy).toFixed(2)}</td>
                          <td>{Number(row.w).toFixed(2)}</td>
                          <td>{Number(row.h).toFixed(2)}</td>
                          <td>{x1.toFixed(2)}</td>
                          <td>{y1.toFixed(2)}</td>
                          <td>{x2.toFixed(2)}</td>
                          <td>{y2.toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {taskType === "classification" && classificationRows.length > 0 && (
              <div style={{ marginTop: "16px", textAlign: "left" }}>
                <div className={rootClass.elem("section-title").toClassName()}>
                  分類數值（共 {classificationRows.length} 類；標籤來自專案 Labeling Interface 之 Choices 順序）
                </div>
                <div style={{ maxHeight: "360px", overflow: "auto", border: "1px solid var(--color-neutral-border-subtler)", borderRadius: "6px" }}>
                  <table className={rootClass.elem("data-table").toClassName()}>
                    <thead style={{ position: "sticky", top: 0, background: "var(--color-neutral-emphasis-subtle)", zIndex: 1 }}>
                      <tr>
                        <th>標籤（介面）</th>
                        <th>索引</th>
                        <th>原始輸出</th>
                        <th>機率</th>
                      </tr>
                    </thead>
                    <tbody>
                      {classificationRows.map((row) => (
                        <tr key={row.classIndex}>
                          <td>
                            {resolveInterfaceLabelName(interfaceClassificationLabels, row.classIndex) || (
                              <span className="text-neutral-content-subtle">—</span>
                            )}
                          </td>
                          <td>{row.classIndex}</td>
                          <td>{Number(row.raw).toFixed(6)}</td>
                          <td>{row.probability.toFixed(6)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>

        <Button
          variant="primary"
          look="filled"
          onClick={sendRequest}
          disabled={loading || !projectId.trim() || !modelName.trim() || !imagePreview || processingImage}
          aria-label="發送請求"
        >
          {loading ? "推理執行中…" : "模型測試"}
        </Button>
      </div>

      {(response.status != null || response.error) && (
        <div className={rootClass.elem("response").toClassName()}>
          <div className={rootClass.elem("section-title").toClassName()}>結果</div>
          {response.error && <div className={rootClass.elem("error").toClassName()}>{response.error}</div>}
          {response.status != null && (
            <div
              className={rootClass
                .elem("response-box")
                .mod({ ok: response.ok, error: !response.ok })
                .toClassName()}
            >
              <div
                className={rootClass
                  .elem("status-line")
                  .mod({ ok: response.ok, error: !response.ok })
                  .toClassName()}
              >
                HTTP 狀態: {response.status} {response.statusText}
              </div>
              <pre className={rootClass.elem("pre").toClassName()}>{response.summary}</pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
