# 從專案根目錄載入 KEY=VALUE 至 $env: 與回傳 hashtable。
# 用法：
#   Docker / compose：. "$PSScriptRoot\Import-DotEnv.ps1" -Root $root
#   本機 dev：      . "$PSScriptRoot\Import-DotEnv.ps1" -Root $root -EnvFile '.env.local'
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [string]$EnvFile = '.env'
)

$envFilePath = Join-Path $Root $EnvFile
$envVars = @{}

if (-not (Test-Path -LiteralPath $envFilePath)) {
    return $envVars
}

foreach ($line in (Get-Content -LiteralPath $envFilePath)) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        $envVars[$key] = $value
        Set-Item -Path "Env:$key" -Value $value
    }
}

return $envVars
