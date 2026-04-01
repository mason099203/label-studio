# 啟動 RQ worker（Windows 需要使用 SimpleWorker，避免 os.fork）
#
# 注意：
# - 需要 Redis 已在 localhost:6379 啟動
# - 本專案 enqueue 的訓練任務使用 low queue
#
<#
.SYNOPSIS
    以 Label Studio 專案依賴啟動 RQ worker。

.NOTES
    若使用 Anaconda 基底環境的 `python`，且已安裝 NumPy 2.x，但 SciPy / Bottleneck 仍為針對 NumPy 1.x 編譯的舊版，
    會出現 `AttributeError: _ARRAY_API not found` 或 `numpy.dtype size changed`。
    解法：優先使用 `poetry install` 後由本腳本呼叫 `poetry run python`，或自行升級 SciPy／Bottleneck 與 NumPy 2 相容的版本。
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "啟動 RQ worker（queue: low, worker: SimpleWorker）..."
Write-Host "如果看到 *** Listening on low... 代表 worker 已就緒。"

Set-Location $root

$env:DJANGO_SETTINGS_MODULE = "core.settings.label_studio"

<#
.SYNOPSIS
    解析應使用的 Python 啟動方式（優先 Poetry，其次專案 .venv）。
.OUTPUTS
    Hashtable：kind 為 poetry | venv | path；venv 時含 path。
#>
function Resolve-LabelStudioPython {
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        return @{ kind = "poetry" }
    }
    $venvPython = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return @{ kind = "venv"; path = $venvPython }
    }
    return @{ kind = "path" }
}

$resolved = Resolve-LabelStudioPython

switch ($resolved.kind) {
    "poetry" {
        & poetry run python label_studio/manage.py rqworker low --worker-class rq.worker.SimpleWorker
    }
    "venv" {
        & $resolved.path label_studio/manage.py rqworker low --worker-class rq.worker.SimpleWorker
    }
    default {
        $msgLines = @(
            'Poetry not found and .venv\\Scripts\\python.exe is missing.'
            'Install Poetry, then run: poetry install (from the repo root).'
            'Hint: poetry.toml config usually creates the venv under .venv.'
        )
        Write-Error ($msgLines -join "`r`n")
        exit 1
    }
}
