# 匯出 Docker 部署所需檔案（不含本機 data / node_modules）
# 用法：.\scripts\export-docker-deploy.ps1
# 產物：dist\label-studio-docker-deploy-<日期>.zip

param(
    [switch]$IncludeGit = $false,
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutputDir) {
    $OutputDir = Join-Path $root 'dist'
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$stageName = "label-studio-docker-deploy-$stamp"
$stage = Join-Path $OutputDir $stageName
$zipPath = Join-Path $OutputDir "$stageName.zip"

New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Copy-Tree {
    param([string]$RelativePath, [string[]]$ExcludeDirs = @())
    $src = Join-Path $root $RelativePath
    if (-not (Test-Path $src)) { return }
    $dst = Join-Path $stage $RelativePath
    $xd = @('node_modules', 'dist', '.nx', '__pycache__', '.pytest_cache', '.venv', '.venv-train', 'bundles') + $ExcludeDirs
    $argList = @(
        $src, $dst,
        '/E', '/SL', '/NFL', '/NDL', '/NJH', '/NJS', '/NC', '/NS', '/NP'
    )
    foreach ($d in $xd) { $argList += "/XD"; $argList += $d }
    & robocopy @argList | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $RelativePath (exit $LASTEXITCODE)" }
}

# 根目錄檔案
$rootFiles = @(
    'Dockerfile',
    '.dockerignore',
    'docker-compose.yml',
    'docker-compose.full.yml',
    'docker-compose.app-only.yml',
    'docker-compose.minio.yml',
    'docker-compose.override.example.yml',
    'pyproject.toml',
    'poetry.lock',
    'README.md',
    'LICENSE',
    '.env.example'
)
foreach ($f in $rootFiles) {
    $src = Join-Path $root $f
    if (Test-Path $src) { Copy-Item $src (Join-Path $stage $f) -Force }
}

# 目錄（Docker build context 需要）
foreach ($dir in @('deploy', 'label_studio', 'web', 'train-server', 'licenses', 'scripts')) {
    Copy-Tree $dir
}

if ($IncludeGit) {
    $gitSrc = Join-Path $root '.git'
    if (Test-Path $gitSrc) {
        Write-Host "Copying .git (required for docker build version step)..."
        Copy-Tree '.git'
    } else {
        Write-Warning '.git not found; target server build may fail unless you git clone instead.'
    }
}

# 部署說明（隨包附上）
@'
# Label Studio Docker 部署包

## 前置
- Docker Engine + Docker Compose v2
- 磁碟建議 20GB+（首次 build 會下載依賴）

## 快速啟動

```bash
cp .env.example .env
# 編輯 .env：LABEL_STUDIO_HOST、CSRF_TRUSTED_ORIGINS（與瀏覽器 URL 一致）
# 若要部署模型至 Triton：INCLUDE_YOLO_TRAINING=true

# 若壓縮包不含 .git（預設），先建立最小 git（Docker build 需要）：
# Windows: .\scripts\init-minimal-git.ps1
# Linux:   bash scripts/init-minimal-git.sh

docker compose build
docker compose up -d
```

瀏覽器：http://<主機IP>:8080

## 變體
| 需求 | 指令 |
|------|------|
| 僅標註（無 PyTorch） | `.env` 設 `INCLUDE_YOLO_TRAINING=false`，`docker compose up --build -d` |
| 部署至 Triton | `.env` 設 `INCLUDE_YOLO_TRAINING=true`，UI 填遠端 Triton URL |
| 完整堆疊（本機 Triton+Train+PG） | `docker compose -f docker-compose.full.yml up --build -d` |

## 資料目錄（首次啟動自動建立）
- `./data` — SQLite、上傳檔
- 若要遷移舊資料，複製本機 `data/` 到同一路徑

## 遠端 Triton（例如 10.214.57.20）
- UI 部署時填：`http://10.214.57.20:18000`
- Upload Server 預設埠：18003（Triton 用 18000 時自動推導）
- 不必在 compose 內起 Triton 服務

詳見專案 `測試以及部署.md`。
'@ | Set-Content -Path (Join-Path $stage 'DEPLOY-DOCKER.md') -Encoding utf8

# 壓縮（tar 比 Compress-Archive 快且穩定）
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Push-Location $stage
try {
    tar -a -cf $zipPath *
} finally {
    Pop-Location
}

# 清理 staging（保留 zip）
Remove-Item $stage -Recurse -Force

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Done: $zipPath ($sizeMb MB)"
Write-Host "Copy this zip to target server, unzip, configure .env, then: docker compose build && docker compose up -d"
