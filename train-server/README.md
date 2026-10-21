# Train Server（獨立 YOLO 訓練服務）

獨立 YOLO 訓練服務，與 Label Studio 主應用分離部署。參考 [Ultralytics 文件](https://docs.ultralytics.com/zh)。

**完整指南：** [`docs/source/guide/train_server_deploy.md`](../docs/source/guide/train_server_deploy.md)

## 目錄結構

```
train-server/
├── docker-compose.yml
├── Dockerfile
└── train_server/        # Python 應用（底線為 Python 模組慣例）
    ├── main.py
    └── requirements-train.txt
```

## 啟動（Docker）

```bash
cd train-server
docker compose up -d --build
```

從專案根目錄：

```bash
docker compose -f train-server/docker-compose.yml up -d --build
```

健康檢查：

```bash
curl http://localhost:8011/health
```

預設掛載 `../data`（專案根目錄 `data/`），與 Label Studio 共用資料集與訓練產物。遠端獨立主機若只用本目錄資料，可設：

```bash
TRAIN_SERVER_DATA_DIR=./data docker compose up -d --build
```

## 與 Label Studio 的關係

| 部署方式 | 啟動 | Train Server URL |
|----------|------|------------------|
| **本目錄 Docker** | `cd train-server && docker compose up -d --build` | `http://localhost:8011` |
| 全堆疊 Docker | 專案根目錄 `docker compose up -d` | 容器內 `http://train-server:8011` |
| 本機 venv（開發） | 根目錄 `.\start-train-server.ps1` | `http://localhost:8011` |

Label Studio `.env`：

```env
TRAIN_SERVER_URL=http://192.168.1.10:8011
TRAIN_SERVER_API_KEY=your-secret   # 可選
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/tasks` | 各任務預設參數 |
| GET | `/models?task=detect` | 模型目錄 |
| POST | `/models/ensure` | 確保權重存在 |
| POST | `/jobs` | 建立訓練任務 |
| GET | `/jobs/{id}` | 任務狀態 |
| GET | `/jobs/{id}/artifacts` | 產物清單 |
| GET | `/jobs/{id}/download?file=best.pt` | 下載權重 |

## 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `TRAIN_SERVER_DATA_DIR` | `../data` | 主機資料目錄（compose volume） |
| `TRAIN_SERVER_HOST` | `0.0.0.0` | 監聽位址 |
| `TRAIN_SERVER_PORT` | `8011` | 監聽埠 |
| `TRAIN_SERVER_API_KEY` | （空） | API 金鑰（可選） |

無 GPU 時可註解 `docker-compose.yml` 的 `deploy.resources`，並加上 `TRAIN_SERVER_DEVICE=cpu`。

## NVIDIA DGX Spark（GB10）

僅 Train Server：

```bash
cd train-server
docker compose build --no-cache
docker compose up -d
```

同機全堆疊（標註 + Train + Triton + GPU）在專案根目錄：

```bash
cp .env.spark.example .env
docker compose -f docker-compose.full.yml -f docker-compose.spark.yml up --build -d
```

`Dockerfile` 預設 **PyTorch cu130**（CUDA 13），須在 Spark 本機建置（`aarch64`）。詳見根目錄 `測試以及部署.md`。
