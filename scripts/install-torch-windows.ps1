# 在 .venv-train 內安裝 PyTorch：有 NVIDIA GPU 用 CUDA，否則 CPU
# 用法：.\scripts\install-torch-windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$TrainPy = Join-Path $Root ".venv-train\Scripts\python.exe"

function Test-NvidiaGpuAvailable {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $false
    }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & nvidia-smi -L 2>$null | Out-Null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    return $ok
}

if (-not (Test-Path -LiteralPath $TrainPy)) {
    Write-Host ".venv-train not found. Running full setup..."
    & (Join-Path $Root "scripts\setup-venv-train.ps1")
    exit $LASTEXITCODE
}

Write-Host "Upgrading pip..."
& $TrainPy -m pip install --upgrade pip

if (Test-NvidiaGpuAvailable) {
    Write-Host "NVIDIA GPU detected. Installing PyTorch with CUDA (cu124)..."
    & $TrainPy -m pip install "numpy==1.26.4" "torch>=2.0.1" "torchvision>=0.15.2" `
        --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
} else {
    Write-Host "No NVIDIA GPU detected. Installing PyTorch CPU..."
    & $TrainPy -m pip install "numpy==1.26.4" "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" `
        --index-url https://download.pytorch.org/whl/cpu --force-reinstall
}

Write-Host "Re-pinning NumPy 1.26.4 if upgraded by dependencies..."
& $TrainPy -m pip install "numpy==1.26.4" --force-reinstall

$verify = @"
import numpy, torch
from ultralytics import YOLO
cuda = torch.cuda.is_available()
device = '0' if cuda else 'cpu'
print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', cuda, 'default_device', device)
"@
& $TrainPy -c $verify
Write-Host "Done. Restart: .\start-train-server.ps1"
