import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useHistory } from "react-router";
import { IconAnalytics, IconFileDownload, IconWarningCircleFilled } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { absoluteURL } from "../../utils/helpers";
import { cn } from "../../utils/bem";
import { appendTrainServerToApiPath, getTrainServerQueryParams } from "../TrainingPage/trainServerStorage";
import "./TrainingProgressPage.scss";

const ACTIVE_STATUSES = new Set(["queued", "preparing", "starting", "running", "started"]);

const STATUS_LABELS = {
  queued: "排隊中",
  preparing: "準備資料",
  starting: "啟動中",
  running: "訓練中",
  started: "訓練中",
  finished: "已完成",
  failed: "失敗",
  unknown: "未知",
};

const formatMetricValue = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  if (value <= 1 && value >= 0) return `${(value * 100).toFixed(2)}%`;
  return value.toFixed(4);
};

const pickDisplayMetrics = (latestMetrics = {}) => {
  const preferred = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/accuracy_top1",
    "metrics/accuracy_top5",
    "map50",
    "map",
    "mp",
    "mr",
    "top1",
    "top5",
    "fitness",
    "train/box_loss",
    "train/cls_loss",
    "val/box_loss",
    "val/cls_loss",
  ];
  const entries = [];
  for (const key of preferred) {
    if (latestMetrics[key] != null) entries.push([key, latestMetrics[key]]);
  }
  if (entries.length >= 4) return entries.slice(0, 8);
  for (const [key, value] of Object.entries(latestMetrics)) {
    if (value == null || key === "epoch") continue;
    if (!entries.some(([k]) => k === key)) entries.push([key, value]);
    if (entries.length >= 8) break;
  }
  return entries;
};

/** 從 API 回應取出 payload；errorFilter 時失敗回應也會被回傳而非 null */
const parseTrainingApiResponse = (res) => {
  if (!res) return { ok: false, error: "無回應" };

  if (res.$meta) {
    if (!res.$meta.ok) {
      const body = res.response;
      const detail =
        (typeof body === "object" && body !== null && (body.detail ?? body.message)) ||
        res.detail ||
        (typeof res.error === "string" ? res.error : null);
      return {
        ok: false,
        error: typeof detail === "string" ? detail : "無法取得訓練狀態",
      };
    }
    // HTTP 200：頂層 error 可能是訓練失敗訊息，不是 API 錯誤
    return { ok: true, data: res };
  }

  if (typeof res.detail === "string" && !res.job_id && !Array.isArray(res.history)) {
    return { ok: false, error: res.detail };
  }

  if (res.job_id || res.status != null || res.epoch != null || res.meta || Array.isArray(res.history)) {
    return { ok: true, data: res };
  }

  if (typeof res.error === "string" && res.error) {
    return { ok: false, error: res.error };
  }

  return { ok: true, data: res };
};

const parseProjectIdFromPath = (pathname) => {
  const match = pathname.match(/\/projects\/(\d+)/);
  return match?.[1] ?? null;
};

/** 訓練進度與即時效果頁面 */
export const TrainingProgressPage = () => {
  const history = useHistory();
  const location = useFixedLocation();
  const pageParams = useParams();
  const api = useAPI();

  const resolvedJobId = pageParams?.job_id ?? (() => {
    const m = location.pathname.match(/\/training\/progress\/([^/]+)/);
    return m?.[1] ?? null;
  })();
  const projectId = pageParams?.id ?? parseProjectIdFromPath(location.pathname);
  const missingJobId = !resolvedJobId;

  const [progress, setProgress] = useState(null);
  const [jobInfo, setJobInfo] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [resolvedTrainServerUrl, setResolvedTrainServerUrl] = useState(null);
  const pollFailStreakRef = useRef(0);

  const trainServerParams = useMemo(() => {
    const base = projectId ? getTrainServerQueryParams(projectId) : {};
    if (resolvedTrainServerUrl && !base.train_server_url) {
      return { ...base, train_server_url: resolvedTrainServerUrl };
    }
    return base;
  }, [projectId, resolvedTrainServerUrl]);

  const closeAndBack = useCallback(() => {
    const suffix = `/training/progress/${resolvedJobId ?? ""}`;
    const path = location.pathname.replace(suffix, "");
    const search = location.search;
    history.replace(`${path}${search !== "?" ? search : ""}`);
  }, [history, location.pathname, location.search, resolvedJobId]);

  const status =
    progress?.status ?? jobInfo?.status ?? jobInfo?.meta?.status ?? (errorMessage ? "failed" : "unknown");

  useEffect(() => {
    if (!projectId || !resolvedJobId) return;
    let cancelled = false;
    let timer = null;

    const poll = async () => {
      try {
        const query = { pk: projectId, job_id: resolvedJobId, ...trainServerParams };
        const [progRes, jobRes] = await Promise.all([
          api.callApi("trainingJobProgress", {
            params: query,
            errorFilter: () => true,
            suppressError: true,
          }),
          api.callApi("trainingJob", {
            params: query,
            errorFilter: () => true,
            suppressError: true,
          }),
        ]);
        if (cancelled) return;

        const prog = parseTrainingApiResponse(progRes);
        const job = parseTrainingApiResponse(jobRes);
        const errors = [prog.error, job.error].filter(Boolean);

        if (prog.ok) {
          setProgress(prog.data);
        }
        if (job.ok) {
          setJobInfo(job.data);
          if (job.data?.train_server_url) {
            setResolvedTrainServerUrl(job.data.train_server_url);
          }
        }

        const trainingError =
          (prog.ok ? prog.data?.error : null) ??
          (job.ok ? job.data?.exc_info ?? job.data?.error : null);

        if (!prog.ok && !job.ok && !trainingError) {
          pollFailStreakRef.current += 1;
          if (pollFailStreakRef.current >= 3) {
            const hint =
              errors[0] === "無回應" || errors[1] === "無回應" || String(errors[0] ?? "").includes("503")
                ? "。請確認 Train Server 容器正在執行（docker ps）且防火牆已開放 8011。"
                : "";
            setErrorMessage(
              errors[0] + (errors[0] !== errors[1] && errors[1] ? `；${errors[1]}` : "") + hint,
            );
          }
        } else {
          pollFailStreakRef.current = 0;
          setErrorMessage(null);
        }

        const st = (prog.ok ? prog.data?.status : null) ?? (job.ok ? job.data?.status ?? job.data?.meta?.status : null);
        const progEpoch = prog.ok ? prog.data?.epoch : null;
        const progTotal = prog.ok ? prog.data?.total_epochs : null;
        const epochDone =
          typeof progEpoch === "number" &&
          typeof progTotal === "number" &&
          progTotal > 0 &&
          progEpoch >= progTotal;
        if (st === "finished" || epochDone || (st === "failed" && epochDone)) return;
      } catch (err) {
        if (!cancelled) setErrorMessage(err?.message ?? "無法取得訓練進度");
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, 2500);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [api, projectId, resolvedJobId, trainServerParams]);

  const totalEpochs =
    progress?.total_epochs ??
    jobInfo?.meta?.total_epochs ??
    jobInfo?.meta?.params?.epochs ??
    jobInfo?.params?.epochs ??
    null;

  const epochRaw = progress?.epoch;
  const epoch =
    epochRaw != null && epochRaw !== ""
      ? epochRaw
      : ACTIVE_STATUSES.has(status) && totalEpochs != null
        ? 0
        : "—";

  const progressPct = useMemo(() => {
    if (progress?.progress_pct != null && progress.progress_pct > 0) {
      return progress.progress_pct;
    }
    if (typeof epochRaw === "number" && typeof totalEpochs === "number" && totalEpochs > 0) {
      return Math.min(100, Math.round((epochRaw / totalEpochs) * 1000) / 10);
    }
    return 0;
  }, [progress?.progress_pct, epochRaw, totalEpochs]);

  const totalEpochsLabel = totalEpochs ?? "—";
  const displayMetrics = pickDisplayMetrics(progress?.latest_metrics ?? progress?.final_metrics ?? {});
  const historyRows = (progress?.history ?? []).slice(-12);

  const epochNum = typeof epochRaw === "number" ? epochRaw : null;
  const totalNum = typeof totalEpochs === "number" ? totalEpochs : null;
  const trainingComplete =
    status === "finished" ||
    (totalNum != null && epochNum != null && epochNum >= totalNum && progressPct >= 99);

  const isFinished = trainingComplete;
  const isFailed =
    (status === "failed" || Boolean(progress?.error || jobInfo?.exc_info)) && !trainingComplete;
  const isActive = ACTIVE_STATUSES.has(status) && !trainingComplete;
  const statusLabel =
    trainingComplete && status === "failed" ? "已完成" : STATUS_LABELS[status] ?? status;
  const completionWarning =
    trainingComplete && (progress?.error || jobInfo?.exc_info || jobInfo?.warning);

  const downloadUrl = (file) =>
    absoluteURL(
      appendTrainServerToApiPath(
        `/api/projects/${projectId}/training/jobs/${resolvedJobId}/download?file=${file}`,
        projectId,
      ),
    );

  return (
    <Modal
      onHide={closeAndBack}
      title="訓練進度"
      style={{ width: 640 }}
      closeOnClickOutside={!isActive}
      allowClose={!isActive}
      visible
    >
      <div className={cn("training-progress").toClassName()}>
        <div className={cn("training-progress").elem("header").toClassName()}>
          <div>
            <div className={cn("training-progress").elem("job-id").toClassName()}>Job：{resolvedJobId}</div>
            <div className={cn("training-progress").elem("message").toClassName()}>
            {progress?.message ?? jobInfo?.meta?.message ?? jobInfo?.message ?? "—"}
            </div>
          </div>
          <span
            className={cn("training-progress")
              .elem("status")
              .mod({ active: isActive, done: isFinished, failed: isFailed })
              .toClassName()}
          >
            {statusLabel}
          </span>
        </div>

        <div className={cn("training-progress").elem("progress-bar-wrap").toClassName()}>
          <div className={cn("training-progress").elem("progress-label").toClassName()}>
            Epoch {epoch} / {totalEpochsLabel}
            {progressPct > 0 ? ` · ${progressPct}%` : isActive ? " · 0%" : ""}
          </div>
          <div
            className={cn("training-progress")
              .elem("progress-track")
              .mod({ queued: status === "queued" || status === "preparing" })
              .toClassName()}
          >
            <div
              className={cn("training-progress")
                .elem("progress-fill")
                .mod({ queued: status === "queued" || status === "preparing" })
                .toClassName()}
              style={
                status === "queued" || status === "preparing"
                  ? undefined
                  : { width: `${Math.min(100, Math.max(0, progressPct || 0))}%` }
              }
            />
          </div>
        </div>

        {!missingJobId && !projectId && (
          <div className={cn("training-progress").elem("error").toClassName()}>
            <IconWarningCircleFilled /> 無法讀取專案 ID，請從專案內 Training 或 Models 頁重新進入。
          </div>
        )}

        {missingJobId && (
          <div className={cn("training-progress").elem("error").toClassName()}>
            <IconWarningCircleFilled /> 無法讀取 Job ID，請從 Training 或 Models 頁重新進入進度畫面。
          </div>
        )}

        {errorMessage && (
          <div className={cn("training-progress").elem("error").toClassName()}>
            <IconWarningCircleFilled /> {errorMessage}
          </div>
        )}

        {isFailed && (progress?.error || jobInfo?.exc_info) && (
          <div className={cn("training-progress").elem("error").toClassName()}>
            <IconWarningCircleFilled />{" "}
            {String(progress?.error ?? jobInfo?.exc_info).slice(0, 1200)}
            {String(progress?.error ?? jobInfo?.exc_info).length > 1200 ? "…" : ""}
          </div>
        )}

        {completionWarning && (
          <div className={cn("training-progress").elem("warning").toClassName()}>
            訓練已完成，但狀態儲存時發生警告（模型權重應仍可下載）：
            {" "}
            {String(completionWarning).slice(0, 400)}
          </div>
        )}

        {displayMetrics.length > 0 && (
          <div className={cn("training-progress").elem("section").toClassName()}>
            <div className={cn("training-progress").elem("section-title").toClassName()}>
              <IconAnalytics /> 目前指標
            </div>
            <div className={cn("training-progress").elem("metrics").toClassName()}>
              {displayMetrics.map(([label, value]) => (
                <div key={label} className={cn("training-progress").elem("metric").toClassName()}>
                  <span className={cn("training-progress").elem("metric-label").toClassName()}>{label}</span>
                  <span className={cn("training-progress").elem("metric-value").toClassName()}>
                    {formatMetricValue(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {historyRows.length > 0 && (
          <div className={cn("training-progress").elem("section").toClassName()}>
            <div className={cn("training-progress").elem("section-title").toClassName()}>Epoch 紀錄（最近）</div>
            <div className={cn("training-progress").elem("history-scroll").toClassName()}>
              <table className={cn("training-progress").elem("history-table").toClassName()}>
                <thead>
                  <tr>
                    <th>Epoch</th>
                    {Object.keys(historyRows[historyRows.length - 1] ?? {})
                      .filter((k) => k !== "epoch")
                      .slice(0, 5)
                      .map((k) => (
                        <th key={k}>{k}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {historyRows.map((row, idx) => {
                    const keys = Object.keys(row).filter((k) => k !== "epoch").slice(0, 5);
                    return (
                      <tr key={`${row.epoch}-${idx}`}>
                        <td>{row.epoch}</td>
                        {keys.map((k) => (
                          <td key={k}>{formatMetricValue(row[k])}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {isFinished && (
          <div className={cn("training-progress").elem("section").toClassName()}>
            <div className={cn("training-progress").elem("section-title").toClassName()}>訓練完成</div>
            <Space size="small">
              <Button
                as="a"
                href={downloadUrl("best.pt")}
                target="_blank"
                rel="noreferrer"
                look="outlined"
                size="small"
                icon={<IconFileDownload />}
              >
                下載 best.pt
              </Button>
              <Button
                as="a"
                href={downloadUrl("last.pt")}
                target="_blank"
                rel="noreferrer"
                look="outlined"
                size="small"
                icon={<IconFileDownload />}
              >
                下載 last.pt
              </Button>
            </Space>
          </div>
        )}

        <div className={cn("training-progress").elem("footer").toClassName()}>
          <Space spread style={{ width: "100%" }}>
            <span className={cn("training-progress").elem("footer-hint").toClassName()}>
              {isActive ? "頁面會自動更新進度與訓練數值" : ""}
            </span>
            <Button onClick={closeAndBack} look="outlined" size="small">
              關閉
            </Button>
          </Space>
        </div>
      </div>
    </Modal>
  );
};

TrainingProgressPage.path = "/training/progress/:job_id";
TrainingProgressPage.modal = true;
