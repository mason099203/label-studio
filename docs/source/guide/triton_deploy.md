# 將 Training 訓練完成的模型部署至 Triton

本指南說明如何**不使用 ML Backend**，直接將 Label Studio 內建 Training（YOLO）訓練產出的模型部署到 **NVIDIA Triton Inference Server**，並如何部署 Triton 本身。

---

## 一、流程概覽

1. 在 Label Studio 專案中準備資料集並**執行 Training**（YOLO detect / classify）。
2. 訓練完成後，在 **Training History** 取得 `run_id`，呼叫 **部署至 Triton** API，將 `best.pt` 轉成 ONNX 並寫入 Triton 模型目錄。
3. 啟動 **Triton**（Docker），掛載同一模型目錄，即可對外提供推論 API。

---

## 二、如何部署 Triton

### 2.1 模型目錄（Model Repository）

Triton 從一個「模型倉庫」目錄載入模型，目錄結構為：

```
<model_repository>/
└── <model_name>/
    ├── config.pbtxt      # 模型設定（本專案由「部署至 Triton」API 自動產生）
    └── 1/                # 版本號
        └── model.onnx    # ONNX 模型檔（由 best.pt 匯出）
```

- 本專案預設將匯出結果寫入：**`<Label Studio 專案根目錄>/data/triton_models`**。
- 若希望改用其他路徑，可設定環境變數 **`TRITON_MODEL_REPOSITORY`**（例如 `/path/to/triton_models`）。

### 2.2 使用 Docker 啟動 Triton

確保上述模型目錄已存在（可先跑一次「部署至 Triton」或手動建立），然後啟動 Triton 並掛載該目錄：

```bash
# 假設模型目錄為 ./data/triton_models（與 Label Studio 同機時）
export TRITON_REPO=/path/to/label-studio/data/triton_models
mkdir -p "$TRITON_REPO"

docker run -d --name triton \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$TRITON_REPO":/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

- **8000**：HTTP API  
- **8001**：gRPC  
- **8002**：metrics  

若無 GPU，可改用 CPU（較慢）：

```bash
docker run -d --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$TRITON_REPO":/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 2.3 使用 Docker Compose（建議）

在專案根目錄建立或使用既有 `docker-compose.yml`：

```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: triton
    # 有 GPU 時取消下一行註解
    # runtime: nvidia
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./data/triton_models:/models
    command: tritonserver --model-repository=/models
    restart: unless-stopped
```

啟動：

```bash
docker compose up -d triton
```

### 2.4 檢查 Triton 是否正常

```bash
# 健康檢查
curl http://localhost:8000/v2/health/ready

# 列出已載入的模型
curl http://localhost:8000/v2/models
```

若回應中出現你部署的 `model_name`，即表示 Triton 已載入該模型。

---

## 三、從 Training 部署到 Triton（API）

訓練完成後，可呼叫 API 將該 run 的 **best.pt** 匯出為 ONNX 並寫入 Triton 模型目錄。

**Endpoint：**

```
POST /api/projects/<project_id>/training/runs/<run_id>/deploy-to-triton
```

**Headers：** `Authorization: Token <your_token>`

**Body（JSON，可選）：**

| 欄位         | 說明 |
|--------------|------|
| `model_name` | Triton 模型名稱（目錄名）。未填則使用 `ls_project_{project_id}_{run_id}`。僅允許英文、數字、底線。 |
| `imgsz`      | 輸入尺寸（預設 640），需與訓練時一致。 |

**範例：**

```bash
curl -X POST "http://localhost:8080/api/projects/1/training/runs/<run_id>/deploy-to-triton" \
  -H "Authorization: Token YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "my_yolo", "imgsz": 640}'
```

**成功時回應範例：**

```json
{
  "message": "Model exported to Triton repository.",
  "triton_repo_root": "/path/to/data/triton_models",
  "model_name": "my_yolo",
  "repo_path": "/path/to/data/triton_models",
  "version_dir": "/path/to/data/triton_models/my_yolo/1",
  "onnx_path": "/path/to/data/triton_models/my_yolo/1/model.onnx",
  "config_path": "/path/to/data/triton_models/my_yolo/config.pbtxt"
}
```

之後**重啟 Triton**（或若已設定 `--model-control-mode=poll` 則等待輪詢），即可在 `http://localhost:8000/v2/models/my_yolo` 使用該模型。

---

## 四、使用 Triton 推論

部署完成且 Triton 已載入模型後，可依 [Triton 文件](https://github.com/triton-inference-server/server) 呼叫推論，例如：

```bash
curl -X POST "http://localhost:8000/v2/models/<model_name>/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"images","shape":[1,3,640,640],"datatype":"FP32","data":[...]}],"outputs":[{"name":"output0"}]}'
```

輸入 shape 需與訓練／匯出時一致（例如 `[batch, 3, imgsz, imgsz]`）。

---

## 五、總結

| 步驟 | 說明 |
|------|------|
| 1. 訓練 | 在 Label Studio 專案中執行 Training，取得 `run_id`。 |
| 2. 部署至 Triton | 呼叫 `POST .../runs/<run_id>/deploy-to-triton`，將 best.pt 匯出到 Triton 模型目錄。 |
| 3. 啟動 Triton | 用 Docker/Compose 掛載同一模型目錄（預設 `./data/triton_models`），啟動 Triton。 |
| 4. 推論 | 透過 Triton HTTP/gRPC API 對已載入的模型進行推論。 |

不需另外啟用或設定 ML Backend；訓練產物直接匯出到 Triton 使用。
