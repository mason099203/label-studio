$root = $PSScriptRoot

. (Join-Path $root 'scripts\Import-DotEnv.ps1') -Root $root | Out-Null

$tritonModelRepo = if ($env:TRITON_MODEL_REPOSITORY) {
    [System.IO.Path]::GetFullPath($env:TRITON_MODEL_REPOSITORY)
} else {
    Join-Path $root 'data\triton_models'
}
$tritonServerUrl = if ($env:TRITON_SERVER_URL) { $env:TRITON_SERVER_URL } else { 'http://localhost:8000' }
$frontendHmr = if ($env:FRONTEND_HMR) { $env:FRONTEND_HMR } else { 'true' }
$redisUrl = if ($env:REDIS_URL) { $env:REDIS_URL } else { 'redis://localhost:6379/0' }

function Test-PortListening { param([int]$Port)
    try { return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1) }
    catch { return $false }
}

if (-not (Test-PortListening -Port 8080)) {
    Start-Process powershell -ArgumentList '-NoExit', '-Command', (
        "cd '$root'; " +
        "`$env:DJANGO_DB='sqlite'; `$env:LOG_DIR='tmp'; `$env:DEBUG='true'; `$env:LOG_LEVEL='DEBUG'; " +
        "`$env:DJANGO_SETTINGS_MODULE='core.settings.label_studio'; `$env:FRONTEND_HMR='$frontendHmr'; " +
        "`$env:REDIS_URL='$redisUrl'; " +
        "`$env:TRITON_MODEL_REPOSITORY='$tritonModelRepo'; `$env:TRITON_SERVER_URL='$tritonServerUrl'; " +
        'poetry run python label_studio/manage.py runserver 8080'
    )
    Write-Host "Backend starting... Triton repo: $tritonModelRepo | URL: $tritonServerUrl"
} else { Write-Host 'Backend already running on http://localhost:8080' }

if (-not (Test-PortListening -Port 8010)) {
    Start-Sleep -Seconds 2
    Start-Process powershell -ArgumentList '-NoExit', '-Command', (
        "cd '$root\web'; " +
        "try { yarn nx reset } catch { Write-Host 'nx reset skipped' }; " +
        'yarn dev:win'
    )
} else { Write-Host 'Frontend HMR already running on http://localhost:8010' }

Write-Host 'Open Label Studio: http://localhost:8010'
