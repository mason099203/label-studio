#!/usr/bin/env bash
# 在目標伺服器解壓後、docker compose build 前執行
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .git ]; then
  echo ".git already exists, skip."
  exit 0
fi
git init -q
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -q -m "docker deploy snapshot" --allow-empty
echo "Minimal .git created for Docker build."
