# 一鍵啟動本機開發環境（可選 Redis、RQ worker、Triton、Train Server、區網 IP）
# 用法：
#   .\start-all.ps1                              # Django + 前端 HMR（localhost）
#   .\start-all.ps1 -WithQueue                   # 含 Redis + RQ worker
#   .\start-all.ps1 -WithQueue -WithTrainServer  # 含 YOLO Train Server
#   .\start-all.ps1 -WithQueue -WithTrainServer -LanAccess   # 支援本機 IP / 區網
#   .\start-full.ps1                             # 上述完整組合（推薦）
#   .\start-all.ps1 -Migrate                     # 啟動前 migrate

param(
    [switch]$WithQueue,
    [switch]$WithTriton,
    [switch]$WithTrainServer,
    [switch]$LanAccess,
    [switch]$Migrate
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

. (Join-Path $root 'scripts\Import-DotEnv.ps1') -Root $root | Out-Null

if ($Migrate) {
    Write-Host 'Running database migrations...'
    $env:DJANGO_DB = 'sqlite'
    $env:LOG_DIR = 'tmp'
    $env:DJANGO_SETTINGS_MODULE = 'core.settings.label_studio'
    . (Join-Path $root 'scripts\Resolve-LabelStudioPython.ps1')
    $py = Resolve-LabelStudioPython -Root $root
    if ($py.kind -eq 'poetry') {
        & poetry run python label_studio/manage.py migrate
    } elseif ($py.kind -eq 'venv') {
        & $py.path label_studio/manage.py migrate
    } else {
        Write-Error 'Python not found. Run: poetry install'
    }
}

if ($WithTriton) {
    & (Join-Path $root 'start-triton.ps1')
}

if ($WithQueue) {
    & (Join-Path $root 'start-redis.ps1')
    Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$root\start-rqworker.ps1'"
    Write-Host 'RQ worker starting in new window...'
}

if ($WithTrainServer) {
    function Test-PortListening { param([int]$Port)
        try { return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1) }
        catch { return $false }
    }
    if (-not (Test-PortListening -Port 8011)) {
        Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$root\start-train-server.ps1'"
        Write-Host 'Train Server starting in new window...'
    } else {
        Write-Host 'Train Server already running on port 8011'
    }
}

& (Join-Path $root 'start-dev.ps1') @PSBoundParameters
