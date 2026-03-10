# Label Studio 本機開發指南（非 Docker）

本說明描述如何在本機執行專案並啟用**前端即時更新（HMR）**，無需使用 Docker。

---

## 快速參考：開啟與關閉

### 一鍵啟動（推薦）

在**專案根目錄**執行下列任一方式，會自動開啟兩個視窗（後端 + 前端）：

```powershell
# 方式一：直接執行腳本
.\start-dev.ps1

# 方式二：若執行原則不允許，先放行再執行
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\start-dev.ps1
```

完成後在瀏覽器開啟 **http://localhost:8010**。

### 開啟（手動依序執行）

| 步驟 | 終端 | 指令 | 說明 |
|------|------|------|------|
| 1 | **終端 A**（專案根目錄） | 見下方「終端 A – 後端」 | 先啟動 Django |
| 2 | **終端 B**（`web/` 目錄） | `yarn dev:win` | 再啟動前端 HMR |
| 3 | 瀏覽器 | 開啟 **http://localhost:8010** | 使用此網址才有即時更新 |

### 關閉

| 方式 | 作法 |
|------|------|
| **建議** | 在 **終端 A** 按 `Ctrl+C` 關閉後端；在 **終端 B** 按 `Ctrl+C` 關閉前端。 |
| 用指令關閉前端（port 8010） | `netstat -ano \| findstr :8010` 取得 PID，再執行 `taskkill /PID <PID> /F` |
| 用指令關閉後端（port 8080） | `netstat -ano \| findstr :8080` 取得 PID，再執行 `taskkill /PID <PID> /F`（注意：可能有多個 process 使用 8080，請確認是 Label Studio 的 Django 再關閉） |

---

## 前置需求

- **Python ≥ 3.10**（後端）
- **Node.js / Yarn**（前端，在 `web/` 目錄）
- **Poetry**（Python 依賴管理，`pip install poetry`）

---

## 一、後端（Django）

在**專案根目錄**執行：

```bash
# 1. 安裝 Python 依賴
poetry install

# 2. 資料庫遷移（首次或 schema 變更時）
poetry run python label_studio/manage.py migrate

# 3. 靜態檔案（非 HMR 模式才需要；使用 HMR 可略過）
# poetry run python label_studio/manage.py collectstatic
```

---

## 二、啟用前端即時更新（HMR）

要讓**修改前端程式即時反映**，需啟用 HMR：Django 不提供前端靜態檔，改由 Webpack Dev Server 提供並熱重載。

### 1. 建立環境變數檔

在**專案根目錄**（與 `label_studio`、`web` 同層）建立 `.env`：

```env
# 啟用前端 HMR（必填）
FRONTEND_HMR=true

# 選填：預設即可
# FRONTEND_HOSTNAME=http://localhost:8010
# DJANGO_HOSTNAME=http://localhost:8080
```

- `FRONTEND_HOSTNAME`：前端 dev server 位址（預設 `http://localhost:8010`）
- `DJANGO_HOSTNAME`：Django 位址（預設 `http://localhost:8080`），前端會把 API 轉發到這裡

### 2. 安裝前端依賴

```bash
cd web
yarn install --frozen-lockfile
```

### 3. 同時啟動後端與前端

需要**兩個終端**：

**終端 A – 後端（專案根目錄）：**

```bash
# Windows PowerShell（有 Poetry 時）
$env:DJANGO_DB="sqlite"; $env:LOG_DIR="tmp"; $env:DEBUG="true"; $env:LOG_LEVEL="DEBUG"; $env:DJANGO_SETTINGS_MODULE="core.settings.label_studio"; $env:FRONTEND_HMR="true"; poetry run python label_studio/manage.py runserver

# Windows PowerShell（僅用 pip 時，先 pip install -e .）
$env:DJANGO_DB="sqlite"; $env:LOG_DIR="tmp"; $env:DEBUG="true"; $env:LOG_LEVEL="DEBUG"; $env:DJANGO_SETTINGS_MODULE="core.settings.label_studio"; $env:FRONTEND_HMR="true"; python label_studio/manage.py runserver

# Linux / macOS / Git Bash
make run-dev
# 或：
# DJANGO_DB=sqlite LOG_DIR=tmp DEBUG=true LOG_LEVEL=DEBUG DJANGO_SETTINGS_MODULE=core.settings.label_studio FRONTEND_HMR=true poetry run python label_studio/manage.py runserver
```

**終端 B – 前端（HMR，在 `web/` 目錄）：**

```bash
cd web

# Windows PowerShell（若直接執行 yarn dev 有問題，可先設環境變數）
$env:NODE_ENV="development"; $env:BUILD_NO_SERVER="true"; yarn ls:dev

# 或使用專案提供的 Windows 腳本（若已加入）
yarn dev:win

# Linux / macOS / Git Bash
yarn dev
# 或從專案根目錄：make frontend-dev
```

### 4. 開啟瀏覽器

- **使用 HMR 時**：請開啟 **http://localhost:8010**  
  - 頁面與 API 都會經由 8010，前端改動會即時熱重載。
- 若未啟用 HMR、只跑後端：則開啟 http://localhost:8080（使用已 build 的靜態檔）。

---

## 指令對照

| 用途           | 指令（在 `web/`）     | 說明                         |
|----------------|------------------------|------------------------------|
| 前端 HMR 開發  | `yarn dev`             | 啟動 dev server + 即時更新   |
| 僅監聽建置     | `yarn watch`           | 檔案變更時建置，不帶 dev server |
| 主應用建置     | `yarn ls:build`        | 正式環境建置                 |
| Editor 監聽    | `yarn lsf:watch`       | 僅 Label Studio Frontend     |
| Data Manager   | `yarn dm:watch`        | 僅 Data Manager 監聽建置     |

---

## 僅建置、不即時更新

若不需要 HMR，只要本機跑起來：

1. 不設 `FRONTEND_HMR=true`（或設為 `false`）。
2. 在 `web/` 執行一次：`yarn ls:build`（或 `yarn build`）。
3. 在專案根目錄執行：`poetry run python label_studio/manage.py collectstatic`。
4. 只啟動後端：`make run-dev`（或上述 `runserver` 指令）。
5. 瀏覽 **http://localhost:8080**。  
此時修改前端需重新執行 `yarn ls:build` 與 `collectstatic` 才會看到變更。

---

## 常見問題

- **前端改動沒反應**  
  - 確認 `.env` 在專案根目錄且含 `FRONTEND_HMR=true`。  
  - 確認是開 **http://localhost:8010**（不是 8080）。  
  - 確認終端 B 跑的是 `yarn dev`（或 `yarn ls:dev`），且無報錯。

- **Windows 下 `yarn dev` 報錯**  
  - 使用 `yarn dev:win`（若已加入），或在 PowerShell 先設定環境變數再執行 `yarn ls:dev`（見上方「終端 B」）。

- **API 404 或連線錯誤**  
  - 先確認終端 A 的 Django `runserver` 已啟動（預設 8080）。  
  - 確認 `.env` 中 `DJANGO_HOSTNAME` 與實際 Django 位址一致（預設 `http://localhost:8080`）。
