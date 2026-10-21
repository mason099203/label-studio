# 僅啟動 Windows 開發用 Docker 基礎設施（Redis + Triton）
# 不啟動 Label Studio / 前端 HMR / Train Server —— 改碼請另跑 start-dev.ps1 或 start-full.ps1
#
# 用法：
#   .\start-infra.ps1              # Redis + Triton
#   .\start-infra.ps1 -RedisOnly   # 僅 Redis
#   .\start-infra.ps1 -TritonOnly  # 僅 Triton
#
# 之後啟動應用（熱更新）：
#   .\start-dev.ps1
#   或 .\start-all.ps1 -WithQueue -WithTrainServer

param(
    [switch]$RedisOnly,
    [switch]$TritonOnly
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$startRedis = -not $TritonOnly
$startTriton = -not $RedisOnly

if ($RedisOnly -and $TritonOnly) {
    Write-Error '請勿同時指定 -RedisOnly 與 -TritonOnly'
}

Write-Host ''
Write-Host '=== Dev infra (Docker only) ==='
Write-Host 'App / HMR is NOT started. Use start-dev.ps1 or start-full.ps1 for code changes.'
Write-Host ''

if ($startRedis) {
    & (Join-Path $root 'start-redis.ps1')
}

if ($startTriton) {
    & (Join-Path $root 'start-triton.ps1')
}

Write-Host ''
Write-Host 'Infra ready. Next:'
if ($startRedis) {
    Write-Host '  .\start-rqworker.ps1          # if you need the training queue'
}
Write-Host '  .\start-dev.ps1               # Django + frontend HMR (:8010)'
Write-Host '  .\start-train-server.ps1      # optional YOLO train server'
Write-Host '  Browser: http://localhost:8010'
Write-Host ''
