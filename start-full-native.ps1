# 本機完整開發（無 Docker）：Django + HMR + Train Server + 區網存取
# 不含 Redis / RQ worker — 背景訓練佇列需另裝 Redis 或改用遠端 Train Server
#
# 用法：.\start-full-native.ps1
# 登入：http://localhost:8010 或 http://<LAN IP>:8010
#
# 與 start-full.ps1 差異：
#   start-full.ps1         → 含 Redis(RQ)，需要 Docker Desktop
#   start-full-native.ps1  → 不需 Docker；YOLO 訓練走遠端 Train Server（本腳本會啟動 :8011）

& (Join-Path $PSScriptRoot 'start-all.ps1') -WithTrainServer -LanAccess @args
