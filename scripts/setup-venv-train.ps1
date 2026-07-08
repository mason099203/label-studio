# 建立 Train Server 專用虛擬環境 .venv-train（與 Label Studio .venv 分離）
# 用法（專案根目錄）：.\scripts\setup-venv-train.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$TrainVenv = Join-Path $Root ".venv-train"
$TrainPy = Join-Path $TrainVenv "Scripts\python.exe"
$ReqFile = Join-Path $Root "train_server\requirements-train.txt"

function Get-BootstrapPython {
    # 1) 本專案 Poetry .venv（Windows 開發最常見）
    $poetryVenvPy = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $poetryVenvPy) {
        return @{ exe = $poetryVenvPy; args = @() }
    }
    # 2) py launcher
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("-3.12", "-3.11", "-3.10", "-3")) {
            $prev = $ErrorActionPreference
            $ErrorActionPreference = "SilentlyContinue"
            & py $ver -c "import sys" 2>$null | Out-Null
            $ok = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $prev
            if ($ok) {
                return @{ exe = "py"; args = @($ver) }
            }
        }
    }
    # 3) PATH 上非 Anaconda 的 python
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "anaconda|conda") {
        return @{ exe = $cmd.Source; args = @() }
    }
    if ($cmd) {
        Write-Warning "Only Anaconda python found. Using it to create venv; train env will still be isolated."
        return @{ exe = $cmd.Source; args = @() }
    }
    throw "No Python found. Run 'poetry install' first or install Python 3.10+."
}

function Invoke-BootstrapPy {
    param([string[]]$ScriptArgs)
    $boot = Get-BootstrapPython
    if ($boot.args.Count -gt 0) {
        & $boot.exe @($boot.args + $ScriptArgs)
    } else {
        & $boot.exe @ScriptArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw ("Python command failed: {0} {1}" -f $boot.exe, ($ScriptArgs -join " "))
    }
}

Write-Host "=== Train Server venv: .venv-train ==="
Write-Host "Root: $Root"

if (-not (Test-Path -LiteralPath $TrainPy)) {
    Write-Host "Creating virtualenv at $TrainVenv ..."
    Invoke-BootstrapPy @("-m", "venv", $TrainVenv)
}

Write-Host "Upgrading pip..."
& $TrainPy -m pip install --upgrade pip

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

if (Test-NvidiaGpuAvailable) {
    Write-Host "NVIDIA GPU detected. Installing PyTorch with CUDA (cu124)..."
    & $TrainPy -m pip install "numpy==1.26.4" "torch>=2.0.1" "torchvision>=0.15.2" `
        --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "No NVIDIA GPU detected. Installing PyTorch CPU..."
    & $TrainPy -m pip install "numpy==1.26.4" "torch==2.0.1+cpu" "torchvision==0.15.2+cpu" `
        --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Installing train_server requirements..."
& $TrainPy -m pip install -r $ReqFile

Write-Host "Re-pinning NumPy 1.26.4 if upgraded by dependencies..."
& $TrainPy -m pip install "numpy==1.26.4" --force-reinstall

Write-Host "Verifying stack..."
$verify = "import numpy, torch; from ultralytics import YOLO; import fastapi, uvicorn; assert numpy.__version__.startswith('1.'); cuda=torch.cuda.is_available(); print('OK numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', cuda, 'default_device', ('0' if cuda else 'cpu'))"
& $TrainPy -c $verify

Write-Host ""
Write-Host "Setup complete. Start Train Server:"
Write-Host "  .\start-train-server.ps1"
Write-Host "Python: $TrainPy"
