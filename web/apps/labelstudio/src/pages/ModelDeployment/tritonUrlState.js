/**
 * 與 Triton 相關之本地設定（與 Playground 共用 localStorage 鍵）。
 * @see PlaygroundTab.jsx
 */

export const TRITON_PLAYGROUND_STATE_KEY = "labelstudio.triton.playground";

/**
 * Triton HTTP 推論服務固定埠號（docker-compose `triton` service）。
 * @type {number}
 */
export const TRITON_HTTP_PORT = 18000;

/**
 * Triton Prometheus Metrics 固定埠號（docker-compose `triton` service）。
 * @type {number}
 */
export const TRITON_METRICS_PORT = 8002;

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
    // URL 無效時回退：去除 scheme 與 port
    return u.replace(/^https?:\/\//i, "").replace(/[:/].*$/, "");
  }
}

/**
 * 根據主機 IP 組成 Triton HTTP 基底 URL（固定埠 18000）。
 * @param {string} host - 主機 IP 或 hostname，例如 `10.0.0.1`
 * @returns {string} 完整 URL，例如 `http://10.0.0.1:18000`；host 為空時回傳空字串
 */
export function buildTritonBaseUrl(host) {
  const h = (host || "").trim();
  if (!h) return "";
  return `http://${h}:${TRITON_HTTP_PORT}`;
}

/**
 * 根據主機 IP 組成 Triton Prometheus Metrics URL（固定埠 8002）。
 * @param {string} host - 主機 IP 或 hostname，例如 `10.0.0.1`
 * @returns {string} 完整 Metrics URL，例如 `http://10.0.0.1:8002/metrics`；host 為空時回傳空字串
 */
export function buildTritonMetricsUrl(host) {
  const h = (host || "").trim();
  if (!h) return "";
  return `http://${h}:${TRITON_METRICS_PORT}/metrics`;
}

/**
 * 正規化 Triton HTTP URL：若使用者未指定埠號，自動補上 {@link TRITON_HTTP_PORT}。
 *
 * @example
 * normalizeTritonUrl("http://10.0.0.1")       // → "http://10.0.0.1:18000"
 * normalizeTritonUrl("http://10.0.0.1:18000") // → "http://10.0.0.1:18000"（原樣）
 * normalizeTritonUrl("http://10.0.0.1:9000")  // → "http://10.0.0.1:9000"（原樣）
 * normalizeTritonUrl("")                       // → ""
 *
 * @param {string} url - 使用者輸入的網址（可含或不含埠號）
 * @returns {string} 補足埠號後的完整 URL；無效或空值時回傳原始字串
 */
export function normalizeTritonUrl(url) {
  const u = (url || "").trim();
  if (!u) return u;
  try {
    const withScheme = /^https?:\/\//i.test(u) ? u : `http://${u}`;
    const parsed = new URL(withScheme);
    // parsed.port 為空字串表示使用者未指定埠號，補上預設值
    if (!parsed.port) {
      parsed.port = String(TRITON_HTTP_PORT);
    }
    // 只回傳 scheme + host + port，捨棄路徑避免污染基底 URL
    return parsed.origin;
  } catch (_) {
    return u;
  }
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
  /** @type {Record<string, string>} */
  const params = { pk };
  const u = (tritonServerUrl || "").trim();
  if (u) params.triton_url = u;
  try {
    const res = await api.callApi("trainingTritonHealth", {
      params,
      errorFilter: () => true,
    });

    // 後端回傳 400 — URL 格式錯誤
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

    // 完全無法連線
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
