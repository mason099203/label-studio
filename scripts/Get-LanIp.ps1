# 取得本機區網 IPv4（供 LAN 存取 URL 顯示與環境變數）
function Get-LanIpAddress {
    try {
        $candidates = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
            Where-Object {
                $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                $_.PrefixOrigin -ne 'WellKnown'
            } |
            Sort-Object InterfaceMetric, SkipAsSource
        if ($candidates) {
            return $candidates[0].IPAddress
        }
    } catch {
        # fallback below
    }
    return '127.0.0.1'
}
