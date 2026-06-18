# 在 .venv-train 內重裝 PyTorch CPU + NumPy 1.x（修復 c10.dll / NumPy 2 衝突）
# 用法：.\scripts\install-torch-cpu-windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$TrainPy = Join-Path $Root ".venv-train\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $TrainPy)) {
    Write-Host ".venv-train not found. Running full setup..."
    & (Join-Path $Root "scripts\setup-venv-train.ps1")
    exit $LASTEXITCODE
}

Write-Host "Reinstalling PyTorch CPU stack in .venv-train..."

& $TrainPy -m pip install --upgrade pip
& $TrainPy -m pip install "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" `
    --index-url https://download.pytorch.org/whl/cpu --force-reinstall
& $TrainPy -m pip install "numpy==1.26.4" --force-reinstall

& $TrainPy -c "import numpy, torch; from ultralytics import YOLO; print('numpy', numpy.__version__, 'torch', torch.__version__)"
Write-Host "Done. Restart: .\start-train-server.ps1"
