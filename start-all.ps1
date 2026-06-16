# 一鍵啟動本機開發環境（可選 Redis、RQ worker、Triton）
# 用法：
#   .\start-all.ps1                    # 僅 Django + 前端 HMR
#   .\start-all.ps1 -WithQueue         # 含 Redis + RQ worker
#   .\start-all.ps1 -WithQueue -WithTriton
#   .\start-all.ps1 -Migrate           # 啟動前執行 migrate

param(
    [switch]$WithQueue,
    [switch]$WithTriton,
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
    & poetry run python label_studio/manage.py migrate
}

if ($WithTriton) {
    & (Join-Path $root 'start-triton.ps1')
}

if ($WithQueue) {
    & (Join-Path $root 'start-redis.ps1')
    Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$root\start-rqworker.ps1'"
    Write-Host 'RQ worker starting in new window...'
}

& (Join-Path $root 'start-dev.ps1')
