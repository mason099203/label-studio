# Label Studio 本機開發一鍵啟動
# 會開啟兩個 PowerShell 視窗：後端 (Django) + 前端 (HMR)
# 關閉時請在各自視窗按 Ctrl+C
#
# 若要使用「Training（YOLO 訓練）」功能（本機 server 訓練）：
# - 先執行：.\start-redis.ps1（啟動 Redis）
# - 再執行：.\start-rqworker.ps1（啟動 RQ worker，Windows 必須用 SimpleWorker）
#   否則會遇到：AttributeError: module 'os' has no attribute 'fork'

$root = $PSScriptRoot

function Test-PortListening {
    param(
        [int]$Port
    )

    try {
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        return [bool]$listener
    } catch {
        return $false
    }
}

if (-not (Test-PortListening -Port 8080)) {
    # 視窗 1：後端 Django
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$root'; " +
        "`$env:DJANGO_DB='sqlite'; `$env:LOG_DIR='tmp'; `$env:DEBUG='true'; `$env:LOG_LEVEL='DEBUG'; " +
        "`$env:DJANGO_SETTINGS_MODULE='core.settings.label_studio'; `$env:FRONTEND_HMR='true'; " +
        "python label_studio/manage.py runserver"
    )
} else {
    Write-Host "Backend is already running on http://localhost:8080"
}

if (-not (Test-PortListening -Port 8010)) {
    # 視窗 2：前端 HMR（稍延遲讓後端先啟動）
    Start-Sleep -Seconds 2
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$root\web'; " +
        "try { yarn nx reset } catch { Write-Host 'nx reset skipped'; } ; " +
        "yarn dev:win"
    )
} else {
    Write-Host "Frontend HMR is already running on http://localhost:8010"
}

Write-Host "Open Label Studio in browser: http://localhost:8010"
Write-Host "Press Ctrl+C in each window to stop services."
