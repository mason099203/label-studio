# Start local Redis via Docker
# Requirements: Docker Desktop installed

$ErrorActionPreference = "Stop"

Write-Host "Starting Redis (container: ls-redis, port: 6379)..."

try {
  docker inspect ls-redis *> $null
  Write-Host "Container exists. Starting..."
  docker start ls-redis | Out-Null
} catch {
  docker run -d --name ls-redis -p 6379:6379 redis:7 | Out-Null
}

Write-Host "Redis is running."

