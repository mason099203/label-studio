# Train Server 健康檢查與 API 煙霧測試
# 用法：.\scripts\test-train-server-health.ps1
#       .\scripts\test-train-server-health.ps1 -BaseUrl http://192.168.1.10:8011 -ApiKey your-secret

param(
    [string]$BaseUrl = "http://localhost:8011",
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"
$BaseUrl = $BaseUrl.TrimEnd("/")

function Invoke-TrainApi {
    param([string]$Path)
    $headers = @{}
    if ($ApiKey) { $headers["X-API-Key"] = $ApiKey }
    return Invoke-RestMethod -Uri "$BaseUrl$Path" -Headers $headers -Method Get
}

Write-Host "Train Server smoke test -> $BaseUrl"
Write-Host ""

try {
    $health = Invoke-TrainApi "/health"
    Write-Host "[OK] /health -> $($health | ConvertTo-Json -Compress)"
} catch {
    Write-Host "[FAIL] /health -> $_" -ForegroundColor Red
    exit 1
}

try {
    $tasks = Invoke-TrainApi "/tasks"
    $count = ($tasks.tasks | Measure-Object).Count
    Write-Host "[OK] /tasks -> $count task types"
} catch {
    Write-Host "[FAIL] /tasks -> $_" -ForegroundColor Red
    exit 1
}

try {
    $models = Invoke-TrainApi "/models?task=detect"
    $modelCount = ($models.models | Measure-Object).Count
    Write-Host "[OK] /models?task=detect -> $modelCount models"
    Write-Host "     models dir: $($models.root)"
    Write-Host "     output root: $($models.output_root)"
} catch {
    Write-Host "[FAIL] /models -> $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green
