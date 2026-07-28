/**
 * 與 Triton 相關之本地設定（與 Playground 共用 localStorage 鍵）。
 * @see PlaygroundTab.jsx
 */

export const TRITON_PLAYGROUND_STATE_KEY = "labelstudio.triton.playground";

/** 常見埠號（僅文件／placeholder 參考，不會自動填入）。 */
export const TRITON_HTTP_PORT_REFERENCE = 18000;
export const TRITON_METRICS_PORT_REFERENCE = 8002;

/** @deprecated 使用 TRITON_HTTP_PORT_REFERENCE */
export const TRITON_HTTP_PORT = TRITON_HTTP_PORT_REFERENCE;

/** @deprecated 使用 TRITON_METRICS_PORT_REFERENCE */
export const TRITON_METRICS_PORT = TRITON_METRICS_PORT_REFERENCE;

/**
 * 檢查 URL 是否含明確埠號（非 http/https 預設埠）。
 * @param {string} raw
 * @returns {boolean}
 */
export function httpUrlHasExplicitPort(raw) {
  const u = (raw || "").trim();
  if (!u) return false;
  try {
    const withScheme = /^https?:\/\//i.test(u) ? u : `http://${u}`;
    const parsed = new URL(withScheme);
    return Boolean(parsed.port);
  } catch (_) {
    return false;
  }
}

/**
 * 驗證遠端服務 URL：需含 scheme（或可補 http://）且必須帶明確埠號。
 * @param {string} raw
 * @param {{ label?: string }} [opts]
 * @returns {{ ok: boolean, normalized: string, error?: string }}
 */
export function validateRemoteServiceUrl(raw, opts = {}) {
  const label = opts.label || "URL";
  const trimmed = (raw || "").trim();
  if (!trimmed) {
    return { ok: false, normalized: "", error: `請輸入完整 ${label}（含 http:// 與埠號）` };
  }
  const normalized = normalizeTritonUrl(trimmed);
  if (!normalized) {
    return { ok: false, normalized: "", error: `無效的 ${label}` };
  }
  if (!httpUrlHasExplicitPort(normalized)) {
    return {
      ok: false,
      normalized,
      error: `${label} 必須包含埠號，例如 http://192.168.1.10:18000`,
    };
  }
  return { ok: true, normalized };
}

/**
 * 從完整 URL 或純 hostname 中提取主機名稱（IP 或 domain），去除 scheme、port 及路徑。
 * @param {string} url - 可能是完整 URL 如 `http://10.0.0.1:18000` 或純 hostname 如 `10.0.0.1`
 * @returns {string} 純 hostname，例如 `10.0.0.1`；無效時回傳原始字串
 */
export function extractHostFromUrl(url) {
  const u = (url || "").trim();
  if (!u) return "";
  try {
    const withScheme = /^https?:\/\//i.test(u) ? u : `http://${u}`;
    return new URL(withScheme).hostname;
  } catch (_) {
    return u.replace(/^https?:\/\//i, "").replace(/[:/].*$/, "");
  }
}

/**
 * 正規化 HTTP(S) 基底 URL（不自動補埠號）。
 * @param {string} url
 * @returns {string}
 */
export function normalizeTritonUrl(url) {
  const u = (url || "").trim();
  if (!u) return "";
  try {
    const withScheme = /^https?:\/\//i.test(u) ? u : `http://${u}`;
    const parsed = new URL(withScheme);
    return parsed.origin;
  } catch (_) {
    return u.replace(/\/+$/, "");
  }
}

/**
 * @param {string} hostOrUrl
 * @returns {string}
 */
export function buildTritonBaseUrl(hostOrUrl) {
  return normalizeTritonUrl(hostOrUrl);
}

/**
 * 依 HTTP URL 之主機組成 Metrics URL（埠號須由呼叫端提供完整 URL；此函式僅供已含埠之 host:port）。
 * @deprecated 請在 UI 直接輸入完整 Metrics URL，或於 Logs 頁手動新增監控伺服器。
 * @param {string} host
 * @returns {string}
 */
export function buildTritonMetricsUrl(host) {
  const h = (host || "").trim();
  if (!h) return "";
  if (/^https?:\/\//i.test(h)) {
    try {
      const parsed = new URL(h);
      if (parsed.port) {
        return `${parsed.protocol}//${parsed.hostname}:${parsed.port}/metrics`;
      }
    } catch (_) {
      return "";
    }
    return "";
  }
  if (/:\d+$/.test(h)) {
    return `http://${h}/metrics`;
  }
  return "";
}

/**
 * 讀取已儲存的 Triton 基底 URL 與 Metrics URL。
 * @returns {{ tritonServerUrl: string, tritonMetricsUrl: string }}
 */
export function readTritonUrlState() {
  if (typeof window === "undefined") {
    return { tritonServerUrl: "", tritonMetricsUrl: "" };
  }
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return {
      tritonServerUrl: typeof parsed?.tritonServerUrl === "string" ? parsed.tritonServerUrl : "",
      tritonMetricsUrl: typeof parsed?.tritonMetricsUrl === "string" ? parsed.tritonMetricsUrl : "",
    };
  } catch (_) {
    return { tritonServerUrl: "", tritonMetricsUrl: "" };
  }
}

/**
 * 合併寫入 Triton URL 欄位，保留同鍵內其他 Playground 狀態。
 * @param {{ tritonServerUrl?: string, tritonMetricsUrl?: string }} partial
 */
export function persistTritonUrlFields(partial) {
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    window.localStorage.setItem(
      TRITON_PLAYGROUND_STATE_KEY,
      JSON.stringify({
        ...parsed,
        ...partial,
      }),
    );
  } catch (_) {
    // 忽略儲存錯誤
  }
}

/**
 * localStorage 中手動監控伺服器的欄位名稱。
 * 儲存在 {@link TRITON_PLAYGROUND_STATE_KEY} 物件的子欄位內。
 * @type {string}
 */
const MANUAL_SERVERS_FIELD = "manualMonitorServers";

/**
 * 讀取使用者手動新增的 Triton 監控伺服器清單。
 * 每筆格式：`{ url: string }` — url 為完整 Triton 推論基底 URL（如 `http://10.0.0.1:18000`）。
 * @returns {Array<{ url: string }>}
 */
export function readManualMonitorServers() {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    const servers = parsed?.[MANUAL_SERVERS_FIELD];
    return Array.isArray(servers) ? servers : [];
  } catch (_) {
    return [];
  }
}

/**
 * 寫入手動監控伺服器清單至 localStorage，保留同鍵內其他欄位。
 * @param {Array<{ url: string }>} servers
 */
export function persistManualMonitorServers(servers) {
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(TRITON_PLAYGROUND_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    window.localStorage.setItem(
      TRITON_PLAYGROUND_STATE_KEY,
      JSON.stringify({
        ...parsed,
        [MANUAL_SERVERS_FIELD]: servers,
      }),
    );
  } catch (_) {
    // 忽略儲存錯誤，不影響監控功能
  }
}

/**
 * 連線驗證結果等級。
 * - `"success"` — live + ready，完全正常
 * - `"warning"` — live 但 not ready（degraded）；伺服器可達但尚無模型就緒，仍可部署
 * - `"error"`   — 無法連線或回應異常；應阻擋部署
 * @typedef {"success" | "warning" | "error"} TritonVerifyLevel
 */

/**
 * 請後端連線檢查 Triton `/v2/health/live` 與 `/v2/health/ready`（需具專案檢視權限）。
 *
 * **三段狀態說明**
 * | 後端 status   | live  | ready | level     | 可部署？ |
 * |--------------|-------|-------|-----------|---------|
 * | online       | true  | true  | success   | ✓       |
 * | degraded     | true  | false | warning   | ✓（允許，伺服器空時正常）|
 * | offline      | false | false | error     | ✗       |
 *
 * @param {{ callApi: function }} api
 * @param {string | number} projectPk
 * @param {string} [tritonServerUrl] 空字串則檢查伺服器預設 `TRITON_SERVER_URL`
 * @returns {Promise<{ level: TritonVerifyLevel, ok: boolean, message: string, raw?: object }>}
 */
export async function verifyTritonConnection(api, projectPk, tritonServerUrl) {
  const pk = String(projectPk ?? "").trim();
  if (!pk) {
    return { level: "error", ok: false, message: "請先選擇專案（驗證需要專案權限）" };
  }
  const u = (tritonServerUrl || "").trim();
  if (u) {
    const check = validateRemoteServiceUrl(u, { label: "Triton URL" });
    if (!check.ok) {
      return { level: "error", ok: false, message: check.error || "Triton URL 無效" };
    }
  }
  /** @type {Record<string, string>} */
  const params = { pk };
  if (u) params.triton_url = normalizeTritonUrl(u);
  try {
    const res = await api.callApi("trainingTritonHealth", {
      params,
      errorFilter: () => true,
    });

    if (res?.detail && res?.base_url === undefined && res?.ok === undefined) {
      return { level: "error", ok: false, message: String(res.detail), raw: res };
    }

    const live = Boolean(res?.live);
    const ready = Boolean(res?.ready);
    const addr = res?.base_url || u || "（伺服器預設）";

    if (live && ready) {
      return {
        level: "success",
        ok: true,
        message: `Triton 可連線且已就緒：${addr}`,
        raw: res,
      };
    }

    if (live && !ready) {
      return {
        level: "warning",
        ok: true,
        message: `Triton 已啟動但尚無模型就緒（degraded）：${addr}。伺服器可達，部署後模型將自動載入。`,
        raw: res,
      };
    }

    const detail = res?.detail ? String(res.detail) : "";
    return {
      level: "error",
      ok: false,
      message: detail || `無法連線至 Triton：${addr}`,
      raw: res,
    };
  } catch (e) {
    return { level: "error", ok: false, message: e?.message || "驗證請求失敗" };
  }
}
