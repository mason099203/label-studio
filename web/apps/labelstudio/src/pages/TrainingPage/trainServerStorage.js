const STORAGE_PREFIX = "ls_train_server_";

/**
 * @param {number|string} projectId
 */
export function getTrainServerSettings(projectId) {
  if (!projectId) return { url: "", apiKey: "", verified: false };
  try {
    const raw = localStorage.getItem(`${STORAGE_PREFIX}${projectId}`);
    if (!raw) return { url: "", apiKey: "", verified: false };
    const parsed = JSON.parse(raw);
    return {
      url: parsed.url ?? "",
      apiKey: parsed.apiKey ?? "",
      verified: Boolean(parsed.verified),
      verifiedAt: parsed.verifiedAt ?? null,
      detail: parsed.detail ?? null,
    };
  } catch {
    return { url: "", apiKey: "", verified: false };
  }
}

/**
 * @param {number|string} projectId
 * @param {{ url?: string, apiKey?: string, verified?: boolean, verifiedAt?: string|null, detail?: string|null }} settings
 */
export function setTrainServerSettings(projectId, settings) {
  if (!projectId) return;
  const current = getTrainServerSettings(projectId);
  localStorage.setItem(
    `${STORAGE_PREFIX}${projectId}`,
    JSON.stringify({
      ...current,
      ...settings,
    }),
  );
}

/**
 * @param {string} raw
 */
export function normalizeTrainServerUrl(raw) {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed.replace(/\/+$/, "");
  }
  return `http://${trimmed.replace(/\/+$/, "")}`;
}

/**
 * @param {number|string} projectId
 */
export function getTrainServerQueryParams(projectId) {
  const { url, apiKey } = getTrainServerSettings(projectId);
  const params = {};
  const normalized = normalizeTrainServerUrl(url);
  if (normalized) params.train_server_url = normalized;
  if (apiKey) params.train_server_api_key = apiKey;
  return params;
}

/**
 * @param {number|string} projectId
 */
export function getTrainServerBodyFields(projectId) {
  const { url, apiKey } = getTrainServerSettings(projectId);
  const body = {};
  const normalized = normalizeTrainServerUrl(url);
  if (normalized) body.train_server_url = normalized;
  if (apiKey) body.train_server_api_key = apiKey;
  return body;
}

/**
 * Append Train Server query params to a relative API path (e.g. download URLs).
 * @param {string} path
 * @param {number|string} projectId
 * @param {{ train_server_url?: string }} [run]
 */
export function appendTrainServerToApiPath(path, projectId, run = {}) {
  if (!path) return path;
  const params = new URLSearchParams(path.includes("?") ? path.split("?")[1] : "");
  const stored = getTrainServerQueryParams(projectId);
  const serverUrl = run.train_server_url || stored.train_server_url;
  if (serverUrl && !params.has("train_server_url")) {
    params.set("train_server_url", serverUrl);
  }
  const apiKey = stored.train_server_api_key;
  if (apiKey && !params.has("train_server_api_key")) {
    params.set("train_server_api_key", apiKey);
  }
  const base = path.split("?")[0];
  const qs = params.toString();
  return qs ? `${base}?${qs}` : base;
}
