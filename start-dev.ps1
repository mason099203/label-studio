$root = $PSScriptRoot

$envFile = Join-Path $root '.env'
$envVars = @{}
if (Test-Path $envFile) {
    foreach ($line in (Get-Content $envFile)) {
        if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
        if ($line -match '^([^=]+)=(.*)$') { $envVars[$Matches[1].Trim()] = $Matches[2].Trim() }
    }
}

$tritonModelRepo = if ($envVars['TRITON_MODEL_REPOSITORY']) { $envVars['TRITON_MODEL_REPOSITORY'] } else { "$root\data\triton_models" }
$tritonServerUrl = if ($envVars['TRITON_SERVER_URL']) { $envVars['TRITON_SERVER_URL'] } else { 'http://localhost:8000' }

function Test-PortListening { param([int]$Port)
    try { return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1) }
    catch { return $false }
}

if (-not (Test-PortListening -Port 8080)) {
    Start-Process powershell -ArgumentList '-NoExit', '-Command', (
        "cd '$root'; " +
        "`$env:DJANGO_DB='sqlite'; `$env:LOG_DIR='tmp'; `$env:DEBUG='true'; `$env:LOG_LEVEL='DEBUG'; " +
        "`$env:DJANGO_SETTINGS_MODULE='core.settings.label_studio'; `$env:FRONTEND_HMR='true'; " +
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