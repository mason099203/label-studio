# Start local Redis via Docker
# Requirements: Docker Desktop installed and running (unless Redis already listens on :6379)

$ErrorActionPreference = "Stop"

function Test-DockerEngine {
    try {
        docker info 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-PortListening {
    param([int]$Port)
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1)
    } catch {
        return $false
    }
}

if (Test-PortListening -Port 6379) {
    Write-Host "Redis already listening on localhost:6379 (skipping Docker)."
    return
}

if (-not (Test-DockerEngine)) {
    Write-Error @"
Docker Desktop is not running (required to start Redis).

Fix:
  1. Start Docker Desktop, wait until it is ready, then run this script again.
  2. Or run UI-only dev (no background training queue): .\start-dev.ps1
  3. Or run: .\start-all.ps1 -WithTrainServer -LanAccess   (without -WithQueue)

Full stack with queue: .\start-full.ps1   (needs Docker for Redis)
"@
}

Write-Host "Starting Redis (container: ls-redis, port: 6379)..."

$containerExists = $false
try {
    docker inspect ls-redis 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $containerExists = $true }
} catch {
    $containerExists = $false
}

if ($containerExists) {
    Write-Host "Container exists. Starting..."
    docker start ls-redis | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start Redis container ls-redis. Try: docker rm -f ls-redis"
    }
} else {
    docker run -d --name ls-redis -p 6379:6379 redis:7 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create Redis container ls-redis."
    }
}

Start-Sleep -Seconds 1
if (-not (Test-PortListening -Port 6379)) {
    Write-Error "Redis container started but localhost:6379 is not ready. Check: docker logs ls-redis"
}

Write-Host "Redis is running on localhost:6379"
