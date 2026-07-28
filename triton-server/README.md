# Triton Server（獨立 GPU 主機）

本目錄提供 **獨立 Triton 推論主機** 的 Docker 部署，供 Label Studio 在**遠端機器**上部署模型時使用。

## 與 Label Studio 的關係

| 部署方式 | 啟動指令 | Triton URL | 模型寫入 |
|----------|----------|------------|----------|
| 本機開發 | `.\start-triton.ps1` | `http://localhost:8000` | Django 直接寫 `data/triton_models` |
| 全堆疊 Docker | 根目錄 `docker compose up` | 主機 `http://localhost:18000`；容器內 `http://triton:8000` | 共用 volume `./data/triton_models` |
| **獨立 Triton 主機** | 本目錄 `docker compose up -d` | `http://<IP>:18000` | Upload Server `:18003` + Triton load API |

Label Studio 呼叫 `deploy-to-triton` 時：

1. **本機**（`localhost` / 共享磁碟）：直接寫入 `TRITON_MODEL_REPOSITORY`，Triton 自動偵測新模型。
2. **遠端**（區網 IP）：透過 Upload Server 上傳 `model.pt` / `config.pbtxt`，再呼叫 Triton `POST /v2/repository/models/{name}/load`（`model-control-mode=explicit`）。

## 埠號

| 服務 | 容器內 | 主機對外 |
|------|--------|----------|
| Triton HTTP | 8000 | **18000** |
| Triton gRPC | 8001 | 18001 |
| Triton Metrics | 8002 | 18002 |
| Upload Server | 8003 | **18003** |

Label Studio `.env` 範例（連此獨立主機）：

```env
TRITON_SERVER_URL=http://192.168.1.10:18000
# 可省略；由 18000 自動推導為 18003
# TRITON_UPLOAD_SERVER_PORT=18003
```

## 啟動

```powershell
# 專案根目錄
mkdir -Force data\triton_models
docker compose -f triton-server/docker-compose.yml up -d --build
```

需 **NVIDIA Container Toolkit** 與 GPU 驅動。

## 健康檢查

```powershell
Invoke-WebRequest -Uri http://localhost:18000/v2/health/ready -UseBasicParsing
Invoke-WebRequest -Uri http://localhost:18003/health -UseBasicParsing
```

## 目錄說明

| 檔案 | 用途 |
|------|------|
| `docker-compose.yml` | Triton + Upload Server |
| `upload_server.py` | 遠端模型上傳 REST API |
| `Dockerfile.upload` | Upload Server 映像 |
| `inspect_model.py` | 開發用：檢視 TorchScript / ONNX 模型結構 |

詳見專案文件：[docs/source/guide/triton_deploy.md](../docs/source/guide/triton_deploy.md)
