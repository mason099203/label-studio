# 一鍵啟動完整開發環境（Label Studio + Redis/RQ + Train Server，支援本機 IP / 區網存取）
# 用法：
#   .\start-full.ps1              # 含 Redis/RQ（需 Docker Desktop）
#   .\start-full-native.ps1       # 無 Docker（UI + Train Server，無 RQ 佇列）
# 登入：本機 http://localhost:8010 ；區網 http://<腳本輸出的 LAN IP>:8010
# 詳見：測試以及部署.md §2.1

& (Join-Path $PSScriptRoot 'start-all.ps1') -WithQueue -WithTrainServer -LanAccess @args
