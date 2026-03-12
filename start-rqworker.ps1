# 啟動 RQ worker（Windows 需要使用 SimpleWorker，避免 os.fork）
#
# 注意：
# - 需要 Redis 已在 localhost:6379 啟動
# - 本專案 enqueue 的訓練任務使用 low queue

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "啟動 RQ worker（queue: low, worker: SimpleWorker）..."
Write-Host "如果看到 *** Listening on low... 代表 worker 已就緒。"

cd $root

$env:DJANGO_SETTINGS_MODULE = "core.settings.label_studio"

python label_studio/manage.py rqworker low --worker-class rq.worker.SimpleWorker

