# 從專案根目錄 .env 載入 KEY=VALUE 至 $env: 與回傳 hashtable。
# 用法：. "$PSScriptRoot\Import-DotEnv.ps1" -Root $root
param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$envFile = Join-Path $Root '.env'
$envVars = @{}

if (-not (Test-Path -LiteralPath $envFile)) {
    return $envVars
}

foreach ($line in (Get-Content -LiteralPath $envFile)) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        $envVars[$key] = $value
        Set-Item -Path "Env:$key" -Value $value
    }
}

return $envVars
