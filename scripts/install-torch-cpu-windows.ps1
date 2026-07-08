# 相容舊名稱；實際邏輯見 install-torch-windows.ps1（GPU 優先，無 GPU 用 CPU）
# 用法：.\scripts\install-torch-cpu-windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
& (Join-Path $Root "scripts\install-torch-windows.ps1")
exit $LASTEXITCODE
