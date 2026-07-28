import { httpUrlHasExplicitPort } from "../ModelDeployment/tritonUrlState";

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
 * @param {string} raw
 * @returns {{ ok: boolean, normalized: string, error?: string }}
 */
export function validateTrainServerUrl(raw) {
  const normalized = normalizeTrainServerUrl(raw);
  if (!normalized) {
    return { ok: false, normalized: "", error: "請輸入完整 Train Server URL（含 http:// 與埠號）" };
  }
  if (!httpUrlHasExplicitPort(normalized)) {
    return {
      ok: false,
      normalized,
      error: "Train Server URL 必須包含埠號，例如 http://192.168.1.10:8011",
    };
  }
  return { ok: true, normalized };
}

/**
 * @param {string} raw
 */
export function hasTrainServerUrl(raw) {
  return Boolean(normalizeTrainServerUrl(raw));
}

/**
 * Persist URL/API key draft; clears verified until user clicks verify again.
 * @param {number|string} projectId
 * @param {{ url?: string, apiKey?: string }} draft
 */
export function saveTrainServerDraft(projectId, draft) {
  if (!projectId) return;
  setTrainServerSettings(projectId, {
    url: draft.url ?? "",
    apiKey: draft.apiKey ?? "",
    verified: false,
    verifiedAt: null,
    detail: null,
  });
}

/**
 * @param {number|string} projectId
 */
export function getTrainServerQueryParams(projectId) {
  const { url, apiKey, verified } = getTrainServerSettings(projectId);
  const params = {};
  const normalized = normalizeTrainServerUrl(url);
  if (!normalized || !verified) return params;
  params.train_server_url = normalized;
  if (apiKey) params.train_server_api_key = apiKey;
  return params;
}

/**
 * @param {number|string} projectId
 * @param {{ url?: string, apiKey?: string, verified?: boolean }} [override]
 */
export function getTrainServerBodyFields(projectId, override = {}) {
  const stored = getTrainServerSettings(projectId);
  const url = override.url ?? stored.url;
  const apiKey = override.apiKey ?? stored.apiKey;
  const verified = override.verified ?? stored.verified;
  const body = {};
  const normalized = normalizeTrainServerUrl(url);
  if (!normalized || !verified) return body;
  body.train_server_url = normalized;
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

/**
 * 經 Django API 下載訓練產物（帶 session 與 train_server 參數；避免 <a href> 401/404）。
 * @param {import("@humansignal/core").ApiContextType["api"]} api
 * @param {{ projectId: number|string, jobId: string, file: string, trainServerParams?: Record<string, string> }} opts
 */
export async function downloadTrainingArtifact(api, { projectId, jobId, file, trainServerParams = {} }) {
  const response = await api.callApi("trainingJobDownloadRaw", {
    params: {
      pk: projectId,
      job_id: jobId,
      file,
      ...trainServerParams,
    },
    errorFilter: () => true,
  });

  if (!response?.ok) {
    let detail = "";
    try {
      const text = await response.text();
      try {
        const parsed = JSON.parse(text);
        detail = parsed.detail ?? text;
      } catch {
        detail = text;
      }
    } catch {
      detail = "";
    }
    throw new Error(detail || `下載失敗（HTTP ${response?.status ?? "error"}）`);
  }

  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = file;
  link.click();
  URL.revokeObjectURL(link.href);
}
