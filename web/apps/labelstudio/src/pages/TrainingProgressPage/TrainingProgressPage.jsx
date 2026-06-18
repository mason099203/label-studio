import { useCallback, useEffect, useMemo, useState } from "react";
import { useHistory } from "react-router";
import { IconAnalytics, IconFileDownload, IconWarningCircleFilled } from "@humansignal/icons";
import { Button } from "@humansignal/ui";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { absoluteURL } from "../../utils/helpers";
import { cn } from "../../utils/bem";
import { getTrainServerQueryParams } from "../TrainingPage/trainServerStorage";
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

/** 訓練進度與即時效果頁面 */
export const TrainingProgressPage = () => {
  const history = useHistory();
  const location = useFixedLocation();
  const pageParams = useParams();
  const api = useAPI();

  const jobId = pageParams?.job_id;
  const projectId = pageParams?.id;
  const missingJobId = !jobId;

  const [progress, setProgress] = useState(null);
  const [jobInfo, setJobInfo] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [previewVersion, setPreviewVersion] = useState(0);

  const trainServerParams = useMemo(
    () => (projectId ? getTrainServerQueryParams(projectId) : {}),
    [projectId],
  );

  const closeAndBack = useCallback(() => {
    const suffix = `/training/progress/${jobId ?? ""}`;
    const path = location.pathname.replace(suffix, "");
    const search = location.search;
    history.replace(`${path}${search !== "?" ? search : ""}`);
  }, [history, location.pathname, location.search, jobId]);

  const status = progress?.status ?? jobInfo?.status ?? "unknown";
  const isActive = ACTIVE_STATUSES.has(status);
  const isFinished = status === "finished";
  const isFailed = status === "failed" || Boolean(progress?.error || jobInfo?.exc_info);

  useEffect(() => {
    if (!projectId || !jobId) return;
    let cancelled = false;
    let timer = null;

    const poll = async () => {
      try {
        const [progRes, jobRes] = await Promise.all([
          api.callApi("trainingJobProgress", {
            params: { pk: projectId, job_id: jobId, ...trainServerParams },
            errorFilter: () => true,
          }),
          api.callApi("trainingJob", {
            params: { pk: projectId, job_id: jobId, ...trainServerParams },
            errorFilter: () => true,
          }),
        ]);
        if (cancelled) return;
        if (progRes) {
          setProgress(progRes);
          setPreviewVersion((v) => v + 1);
        }
        if (jobRes) setJobInfo(jobRes);
        setErrorMessage(null);

        const st = progRes?.status ?? jobRes?.status;
        if (st === "finished" || st === "failed") return;
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
  }, [api, projectId, jobId, trainServerParams]);

  const progressPct = progress?.progress_pct ?? 0;
  const epoch = progress?.epoch ?? "—";
  const totalEpochs = progress?.total_epochs ?? "—";
  const displayMetrics = pickDisplayMetrics(progress?.latest_metrics ?? progress?.final_metrics ?? {});
  const previewImages = progress?.preview_images ?? [];
  const historyRows = (progress?.history ?? []).slice(-12);

  const previewUrl = (name) => {
    const qs = new URLSearchParams({ ...trainServerParams, file: name, _: String(previewVersion) });
    return absoluteURL(`/api/projects/${projectId}/training/jobs/${jobId}/preview/?${qs.toString()}`);
  };

  const downloadUrl = (file) =>
    absoluteURL(`/api/projects/${projectId}/training/jobs/${jobId}/download?file=${file}`);

  return (
    <Modal
      onHide={closeAndBack}
      title="訓練進度"
      style={{ width: 920 }}
      closeOnClickOutside={!isActive}
      allowClose={!isActive}
      visible
    >
      <div className={cn("training-progress").toClassName()}>
        <div className={cn("training-progress").elem("header").toClassName()}>
          <div>
            <div className={cn("training-progress").elem("job-id").toClassName()}>Job：{jobId}</div>
            <div className={cn("training-progress").elem("message").toClassName()}>
              {progress?.message ?? jobInfo?.meta?.message ?? "—"}
            </div>
          </div>
          <span
            className={cn("training-progress")
              .elem("status")
              .mod({ active: isActive, done: isFinished, failed: isFailed })
              .toClassName()}
          >
            {STATUS_LABELS[status] ?? status}
          </span>
        </div>

        <div className={cn("training-progress").elem("progress-bar-wrap").toClassName()}>
          <div className={cn("training-progress").elem("progress-label").toClassName()}>
            Epoch {epoch} / {totalEpochs}
            {progressPct ? ` · ${progressPct}%` : ""}
          </div>
          <div className={cn("training-progress").elem("progress-track").toClassName()}>
            <div
              className={cn("training-progress").elem("progress-fill").toClassName()}
              style={{ width: `${Math.min(100, Math.max(0, progressPct || 0))}%` }}
            />
          </div>
        </div>

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
            <IconWarningCircleFilled /> {progress?.error ?? jobInfo?.exc_info}
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

        {previewImages.length > 0 && (
          <div className={cn("training-progress").elem("section").toClassName()}>
            <div className={cn("training-progress").elem("section-title").toClassName()}>訓練即時效果</div>
            <div className={cn("training-progress").elem("previews").toClassName()}>
              {previewImages.map((name) => (
                <figure key={`${name}-${previewVersion}`} className={cn("training-progress").elem("preview").toClassName()}>
                  <img src={previewUrl(name)} alt={name} loading="lazy" />
                  <figcaption>{name}</figcaption>
                </figure>
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
              {isActive ? "頁面會自動更新進度與預覽圖" : ""}
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
