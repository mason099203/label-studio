/**
 * 與 Triton 相關之本地設定（與 Playground 共用 localStorage 鍵）。
 * @see PlaygroundTab.jsx
 */

export const TRITON_PLAYGROUND_STATE_KEY = "labelstudio.triton.playground";

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
