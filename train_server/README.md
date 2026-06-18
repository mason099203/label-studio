# Label Studio Train Server

獨立 YOLO 訓練服務，與 Label Studio 主應用分離部署。參考 [Ultralytics 文件](https://docs.ultralytics.com/zh)。

## 功能

- 支援 YOLO 任務：`detect`、`segment`、`classify`、`pose`、`obb`
- 預設模型：YOLO11 / YOLO26 各尺寸（n/s/m/l/x）
- 模型不存在時透過 Ultralytics **自動下載**
- 各任務預設訓練參數（epochs、imgsz、batch 等）對齊 Ultralytics 預設值
- REST API 供 Label Studio 後端轉發訓練任務

## 快速啟動

### Docker Compose（與 Label Studio 一起）

```bash
docker compose up -d train-server
```

主應用設定：

```env
TRAIN_SERVER_URL=http://train-server:8011
TRAIN_SERVER_API_KEY=your-secret   # 可選
```

### 本機（Windows）

Train Server 使用**獨立虛擬環境** `.venv-train`，與 Label Studio 的 `.venv` 分離（避免 NumPy 2.x / torch / Anaconda 衝突）：

```powershell
# 首次設定（約需數分鐘，下載 PyTorch CPU）
.\scripts\setup-venv-train.ps1

# 啟動（預設 http://localhost:8011）
.\start-train-server.ps1
```

若 torch 載入失敗（如 `c10.dll`），可重裝：

```powershell
.\scripts\install-torch-cpu-windows.ps1
```

Label Studio 本機開發：

```env
TRAIN_SERVER_URL=http://localhost:8011
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/tasks` | 各任務預設參數 |
| GET | `/models?task=detect` | 模型目錄 |
| POST | `/models/ensure` | 確保權重存在（自動下載） |
| POST | `/jobs` | 建立訓練任務（multipart：dataset_zip + 表單欄位） |
| GET | `/jobs/{id}` | 任務狀態 |
| GET | `/jobs/{id}/artifacts` | 產物清單 |
| GET | `/jobs/{id}/download?file=best.pt` | 下載權重 |

若設定 `TRAIN_SERVER_API_KEY`，請求需帶 header：`X-API-Key: <key>`。

## 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `TRAIN_SERVER_HOST` | `0.0.0.0` | 監聽位址 |
| `TRAIN_SERVER_PORT` | `8011` | 監聽埠 |
| `TRAIN_SERVER_OUTPUT_ROOT` | `/data/training/models/trained` | 訓練產物根目錄 |
| `TRAIN_SERVER_MODELS_DIR` | `/data/training/models/original` | 預訓練權重快取 |
| `TRAIN_SERVER_MAX_WORKERS` | `1` | 同時訓練任務數 |
| `TRAIN_SERVER_API_KEY` | （空） | API 金鑰（可選） |

## 共享資料集路徑

若 Label Studio 與 Train Server 掛載同一 `data/` 卷，可在建立 job 時傳 `dataset_path` 略過 zip 上傳。
