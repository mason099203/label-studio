# 啟動 YOLO Train Server（使用獨立 .venv-train，不與 Label Studio .venv 共用）
# 首次請先：.\scripts\setup-venv-train.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$TrainVenv = Join-Path $Root ".venv-train"
$TrainPy = Join-Path $TrainVenv "Scripts\python.exe"
$TrainScripts = Join-Path $TrainVenv "Scripts"

. (Join-Path $Root "scripts\Get-LanIp.ps1")

$env:TRAIN_SERVER_HOST = if ($env:TRAIN_SERVER_HOST) { $env:TRAIN_SERVER_HOST } else { "0.0.0.0" }
$env:TRAIN_SERVER_PORT = if ($env:TRAIN_SERVER_PORT) { $env:TRAIN_SERVER_PORT } else { "8011" }
$env:TRAIN_SERVER_OUTPUT_ROOT = if ($env:TRAIN_SERVER_OUTPUT_ROOT) {
    $env:TRAIN_SERVER_OUTPUT_ROOT
} else {
    Join-Path $Root "data\training\models\trained"
}
$env:TRAIN_SERVER_MODELS_DIR = if ($env:TRAIN_SERVER_MODELS_DIR) {
    $env:TRAIN_SERVER_MODELS_DIR
} else {
    Join-Path $Root "data\training\models\original"
}

function Use-IsolatedTrainEnv {
    if (Test-Path -LiteralPath $TrainScripts) {
        $env:PATH = "$TrainScripts;" + ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch 'anaconda|conda' }) -join ';'
    }
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONPATH = ""
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
    Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
}

function Test-TrainStack {
    & $TrainPy -c "import numpy, torch; from ultralytics import YOLO; import fastapi" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $TrainPy)) {
    Write-Host ".venv-train not found. Running setup..."
    & (Join-Path $Root "scripts\setup-venv-train.ps1")
}

if (-not (Test-TrainStack)) {
    Write-Host "Train stack incomplete. Repairing..."
    & (Join-Path $Root "scripts\install-torch-windows.ps1")
}

$lanIp = Get-LanIpAddress
Write-Host "Train Server -> http://localhost:$($env:TRAIN_SERVER_PORT)  (LAN: http://${lanIp}:$($env:TRAIN_SERVER_PORT))"
Write-Host "Python       -> $TrainPy"
Write-Host "Models dir   -> $($env:TRAIN_SERVER_MODELS_DIR)"
Write-Host "Output root  -> $($env:TRAIN_SERVER_OUTPUT_ROOT)"

Use-IsolatedTrainEnv
& $TrainPy -m train_server.main
