# 一鍵啟動完整開發環境（Label Studio + Redis/RQ + Train Server，支援本機 IP / 區網存取）
# 用法：.\start-full.ps1
# 等同：.\start-all.ps1 -WithQueue -WithTrainServer -LanAccess

& (Join-Path $PSScriptRoot 'start-all.ps1') -WithQueue -WithTrainServer -LanAccess @args
