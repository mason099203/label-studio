$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

. (Join-Path $root 'scripts\Import-DotEnv.ps1') -Root $root | Out-Null

$tritonModelRepo = if ($env:TRITON_MODEL_REPOSITORY) {
    [System.IO.Path]::GetFullPath($env:TRITON_MODEL_REPOSITORY)
} else {
    Join-Path $root 'data\triton_models'
}
$tritonServerUrl = if ($env:TRITON_SERVER_URL) { $env:TRITON_SERVER_URL } else { 'http://localhost:8000' }
$redisUrl = if ($env:REDIS_URL) { $env:REDIS_URL } else { 'redis://localhost:6379/0' }

Write-Host 'Starting RQ worker (queue: low, SimpleWorker)...'
Write-Host "Triton model repo : $tritonModelRepo"
Write-Host "Triton URL        : $tritonServerUrl"
Write-Host "Redis URL         : $redisUrl"

Set-Location $root
$env:DJANGO_DB = 'sqlite'
$env:LOG_DIR = 'tmp'
$env:DJANGO_SETTINGS_MODULE = 'core.settings.label_studio'
$env:TRITON_MODEL_REPOSITORY = $tritonModelRepo
$env:TRITON_SERVER_URL = $tritonServerUrl
$env:REDIS_URL = $redisUrl

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
