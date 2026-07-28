param(
    [switch]$WithQueue,
    [switch]$WithTriton,
    [switch]$WithTrainServer,
    [switch]$LanAccess,
    [switch]$Migrate
)

$root = $PSScriptRoot

. (Join-Path $root 'scripts\Import-DotEnv.ps1') -Root $root | Out-Null
. (Join-Path $root 'scripts\Get-LanIp.ps1')
. (Join-Path $root 'scripts\Resolve-LabelStudioPython.ps1')

$tritonModelRepo = if ($env:TRITON_MODEL_REPOSITORY) {
    [System.IO.Path]::GetFullPath($env:TRITON_MODEL_REPOSITORY)
} else {
    Join-Path $root 'data\triton_models'
}
$tritonServerUrl = if ($env:TRITON_SERVER_URL) { $env:TRITON_SERVER_URL } else { 'http://localhost:8000' }
$frontendHmr = if ($env:FRONTEND_HMR) { $env:FRONTEND_HMR } else { 'true' }
$redisUrl = if ($env:REDIS_URL) { $env:REDIS_URL } else { 'redis://localhost:6379/0' }

$hostName = if ($LanAccess) { Get-LanIpAddress } else { 'localhost' }
$djangoUrl = "http://${hostName}:8080"
$frontendUrl = "http://${hostName}:8010"
$trainServerUrl = "http://${hostName}:8011"
$djangoBind = if ($LanAccess) { '0.0.0.0:8080' } else { '8080' }

$resolvedPy = Resolve-LabelStudioPython -Root $root
if ($resolvedPy.kind -eq 'poetry') {
    $runBackendCmd = 'poetry run python label_studio/manage.py runserver ' + $djangoBind
} elseif ($resolvedPy.kind -eq 'venv') {
    $venvPy = $resolvedPy.path -replace '\\', '/'
    $runBackendCmd = "& '$($resolvedPy.path)' label_studio/manage.py runserver $djangoBind"
} else {
    Write-Error '找不到 Python。請先執行 poetry install 或建立 .venv'
}

function Test-PortListening { param([int]$Port)
    try { return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1) }
    catch { return $false }
}

Write-Host ''
Write-Host '=== Label Studio Dev ==='
if ($LanAccess) {
    Write-Host "LAN IP       : $hostName"
    Write-Host "Frontend     : $frontendUrl"
    Write-Host "Backend API  : $djangoUrl"
    Write-Host "Train Server : $trainServerUrl  (Training 頁請填完整 URL，例如 $trainServerUrl)"
} else {
    Write-Host 'Frontend     : http://localhost:8010'
    Write-Host 'Backend API  : http://localhost:8080'
}
Write-Host ''

if (-not (Test-PortListening -Port 8080)) {
    $csrfOrigins = if ($LanAccess) { "http://${hostName}:8010,http://${hostName}:8080" } else { '' }
    Start-Process powershell -ArgumentList '-NoExit', '-Command', (
        "cd '$root'; " +
        "`$env:DJANGO_DB='sqlite'; `$env:LOG_DIR='tmp'; `$env:DEBUG='true'; `$env:LOG_LEVEL='DEBUG'; " +
        "`$env:COLLECT_ANALYTICS='false'; " +
        "`$env:DJANGO_SETTINGS_MODULE='core.settings.label_studio'; `$env:FRONTEND_HMR='$frontendHmr'; " +
        "`$env:REDIS_URL='$redisUrl'; " +
        "`$env:TRITON_MODEL_REPOSITORY='$tritonModelRepo'; `$env:TRITON_SERVER_URL='$tritonServerUrl'; " +
        $(if ($LanAccess) {
            "`$env:FRONTEND_HOSTNAME='$frontendUrl'; `$env:TRAIN_SERVER_URL='$trainServerUrl'; `$env:CSRF_TRUSTED_ORIGINS='$csrfOrigins'; "
        } else { '' }) +
        $(if ($env:TRAIN_SERVER_SHARED_DATA_ROOT) {
            "`$env:TRAIN_SERVER_SHARED_DATA_ROOT='$($env:TRAIN_SERVER_SHARED_DATA_ROOT)'; "
        } elseif (-not $LanAccess -and (docker ps --format '{{.Names}}' 2>$null | Select-String -Pattern 'train-server' -Quiet)) {
            "`$env:TRAIN_SERVER_SHARED_DATA_ROOT='/data'; "
        } else { '' }) +
        $runBackendCmd
    )
    Write-Host "Backend starting on $djangoBind ..."
} else {
    Write-Host "Backend already running on $djangoUrl"
}

if (-not (Test-PortListening -Port 8010)) {
    Start-Sleep -Seconds 2
    Start-Process powershell -ArgumentList '-NoExit', '-Command', (
        "cd '$root\web'; " +
        $(if ($LanAccess) {
            "`$env:FRONTEND_HOSTNAME='$frontendUrl'; `$env:DJANGO_HOSTNAME='$djangoUrl'; `$env:WEBPACK_DEV_HOST='0.0.0.0'; "
        } else { '' }) +
        "`$env:NODE_ENV='development'; `$env:BUILD_NO_SERVER='true'; " +
        "try { yarn nx reset } catch { Write-Host 'nx reset skipped' }; " +
        'yarn dev:win'
    )
    Write-Host "Frontend HMR starting... $frontendUrl"
} else {
    Write-Host "Frontend HMR already running on $frontendUrl"
}

Write-Host ''
Write-Host "Open Label Studio: $frontendUrl"
