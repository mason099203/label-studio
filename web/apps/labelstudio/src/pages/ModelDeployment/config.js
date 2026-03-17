/**
 * 模型部署服務設定（對應 docker-compose 中的服務）
 * 每個部署的模型提供 API，需使用指定的 API Key 才能呼叫
 * @typedef {Object} DeployedService
 * @property {string} id - 服務唯一識別
 * @property {string} name - 顯示名稱
 * @property {string} description - 說明
 * @property {string} baseUrl - API 基礎 URL（可被 Kong 代理）
 * @property {number} port - 服務埠號
 * @property {string} apiKey - 預設 API Key（實際應由後端/環境變數提供）
 * @property {'running'|'stopped'|'unknown'} status - 狀態
 */

const DEFAULT_API_KEY = "ls-model-deploy-key-change-in-production";

/** @type {DeployedService[]} */
export const DEPLOYED_SERVICES = [
  {
    id: "triton",
    name: "Triton Inference Server",
    description: "NVIDIA Triton 推論服務，支援多模型部署",
    baseUrl: "/triton",
    port: 8000,
    httpPort: 8000,
    grpcPort: 8001,
    metricsPort: 8002,
    apiKey: DEFAULT_API_KEY,
    status: "running",
  },
  {
    id: "minio",
    name: "MinIO",
    description: "S3 相容物件儲存",
    baseUrl: "/minio",
    port: 9000,
    apiKey: DEFAULT_API_KEY,
    status: "running",
  },
  {
    id: "kong",
    name: "Kong API Gateway",
    description: "API 閘道，統一入口與 Key 驗證",
    baseUrl: "http://localhost:8080",
    port: 8080,
    apiKey: DEFAULT_API_KEY,
    status: "running",
  },
];

/** Grafana / Prometheus 等監控入口（對應 docker-compose） */
export const MONITORING_ENDPOINTS = [
  { name: "Grafana", url: "http://localhost:3000", description: "儀表板與視覺化" },
  { name: "Prometheus", url: "http://localhost:9090", description: "指標查詢" },
  { name: "Triton Metrics", url: "http://localhost:8002/metrics", description: "Triton 推論指標" },
];
