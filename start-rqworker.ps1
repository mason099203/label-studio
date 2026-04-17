$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$envFile = Join-Path $root '.env'
$envVars = @{}
if (Test-Path $envFile) {
    foreach ($line in (Get-Content $envFile)) {
        if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
        if ($line -match '^([^=]+)=(.*)$') {
            $envVars[$Matches[1].Trim()] = $Matches[2].Trim()
        }
    }
}

$tritonModelRepo = if ($envVars['TRITON_MODEL_REPOSITORY']) { $envVars['TRITON_MODEL_REPOSITORY'] } else { "$root\data\triton_models" }
$tritonServerUrl = if ($envVars['TRITON_SERVER_URL']) { $envVars['TRITON_SERVER_URL'] } else { 'http://localhost:8000' }

Write-Host "Starting RQ worker (queue: low, SimpleWorker)..."
Write-Host "Triton model repo : $tritonModelRepo"
Write-Host "Triton URL        : $tritonServerUrl"

Set-Location $root
$env:DJANGO_SETTINGS_MODULE  = 'core.settings.label_studio'
$env:TRITON_MODEL_REPOSITORY = $tritonModelRepo
$env:TRITON_SERVER_URL       = $tritonServerUrl

function Resolve-LabelStudioPython {
    if (Get-Command poetry -ErrorAction SilentlyContinue) { return @{ kind = 'poetry' } }
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPy) { return @{ kind = 'venv'; path = $venvPy } }
    return @{ kind = 'path' }
}

$resolved = Resolve-LabelStudioPython
switch ($resolved.kind) {
    'poetry' { & poetry run python label_studio/manage.py rqworker low --worker-class rq.worker.SimpleWorker }
    'venv'   { & $resolved.path label_studio/manage.py rqworker low --worker-class rq.worker.SimpleWorker }
    default  { Write-Error 'Poetry not found. Run: poetry install'; exit 1 }
}