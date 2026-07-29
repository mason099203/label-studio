# 在目標伺服器解壓後、docker compose build 前執行（建立 Docker build 所需的最小 .git）
# 用法：.\scripts\init-minimal-git.ps1  或  bash scripts/init-minimal-git.sh

$ErrorActionPreference = 'Stop'
$root = if ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent } else { Get-Location }

Set-Location $root
if (Test-Path '.git') {
    Write-Host '.git already exists, skip.'
    exit 0
}

git init -q
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "docker deploy snapshot" --allow-empty
Write-Host 'Minimal .git created for Docker build.'
