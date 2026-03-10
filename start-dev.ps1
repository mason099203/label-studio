# Label Studio 本機開發一鍵啟動
# 會開啟兩個 PowerShell 視窗：後端 (Django) + 前端 (HMR)
# 關閉時請在各自視窗按 Ctrl+C

$root = $PSScriptRoot

# 視窗 1：後端 Django
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root'; " +
    "`$env:DJANGO_DB='sqlite'; `$env:LOG_DIR='tmp'; `$env:DEBUG='true'; `$env:LOG_LEVEL='DEBUG'; " +
    "`$env:DJANGO_SETTINGS_MODULE='core.settings.label_studio'; `$env:FRONTEND_HMR='true'; " +
    "python label_studio/manage.py runserver"
)

# 視窗 2：前端 HMR（稍延遲讓後端先啟動）
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root\web'; yarn dev:win"
)

Write-Host "已開啟兩個視窗：後端 (Django) 與前端 (HMR)。"
Write-Host "瀏覽器請開啟: http://localhost:8010"
Write-Host "關閉時在各自視窗按 Ctrl+C。"
