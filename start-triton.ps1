# 以 Docker 啟動本機 Triton Inference Server
# 需求：Docker Desktop；模型目錄由 .env 的 TRITON_MODEL_REPOSITORY 決定

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

. (Join-Path $root 'scripts\Import-DotEnv.ps1') -Root $root | Out-Null

$tritonModelRepo = if ($env:TRITON_MODEL_REPOSITORY) { $env:TRITON_MODEL_REPOSITORY } else { Join-Path $root 'data\triton_models' }
$tritonModelRepo = [System.IO.Path]::GetFullPath($tritonModelRepo)

if (-not (Test-Path -LiteralPath $tritonModelRepo)) {
    New-Item -ItemType Directory -Path $tritonModelRepo -Force | Out-Null
    Write-Host "Created model repository: $tritonModelRepo"
}

Write-Host "Starting Triton (container: ls-triton, ports: 8000/8001/8002)..."
Write-Host "Model repository: $tritonModelRepo"

$dockerArgs = @(
    'run', '-d', '--name', 'ls-triton',
    '-p', '8000:8000', '-p', '8001:8001', '-p', '8002:8002',
    '-v', "${tritonModelRepo}:/models",
    'nvcr.io/nvidia/tritonserver:24.01-py3',
    'tritonserver', '--model-repository=/models'
)

try {
    docker inspect ls-triton *> $null
    Write-Host 'Container exists. Removing old container to refresh mount...'
    docker rm -f ls-triton | Out-Null
} catch {
    # container does not exist
}

docker @dockerArgs | Out-Null
Write-Host 'Triton is running.'
Write-Host 'Health: curl http://localhost:8000/v2/health/ready'
Write-Host 'Models: curl http://localhost:8000/v2/models'
