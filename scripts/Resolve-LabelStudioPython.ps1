function Resolve-LabelStudioPython {
    param([string]$Root)
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        return @{ kind = 'poetry' }
    }
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPy) {
        return @{ kind = 'venv'; path = $venvPy }
    }
    return @{ kind = 'missing' }
}
