"""RQ jobs for on-server model training and evaluation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rq import get_current_job

logger = logging.getLogger(__name__)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "run_meta.json"


def _write_run_meta(run_dir: Path, payload: Dict[str, Any]) -> None:
    path = _run_meta_path(run_dir)
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(payload)
    _write_json(path, current)


def yolo_detect_train_job(
    *,
    project_id: int,
    base_weights: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    output_root: str,
) -> Dict[str, Any]:
    """
    Train a YOLO detect model using Ultralytics on the Label Studio server.

    This job updates `job.meta` so the frontend can poll for progress.

    Notes:
    - To change where artifacts are stored, modify `output_root` (set in API).
    - To run on a separate training server later, replace this RQ job with a remote dispatcher.
    """

    job = get_current_job()
    now = datetime.now().isoformat()

    if job is not None:
        job.meta.update(
            {
                "kind": "yolo_detect_train",
                "project_id": project_id,
                "status": "starting",
                "message": "Starting training job",
                "created_at": now,
            }
        )
        job.save_meta()

    output_root_path = Path(output_root)
    run_dir = output_root_path / f"project_{project_id}" / (job.id if job is not None else f"job_{now}")
    _safe_mkdir(run_dir)
    _write_run_meta(
        run_dir,
        {
            "job_id": job.id if job is not None else None,
            "project_id": project_id,
            "status": "starting",
            "message": "Starting training job",
            "created_at": now,
            "params": {
                "base_weights": base_weights,
                "data_yaml": str(data_yaml),
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
            },
        },
    )

    def _fail(message: str, error: Exception | None = None) -> None:
        if job is not None:
            job.meta.update(
                {
                    "status": "failed",
                    "message": message,
                    "error": str(error) if error else None,
                    "failed_at": datetime.now().isoformat(),
                }
            )
            job.save_meta()
        _write_run_meta(
            run_dir,
            {
                "status": "failed",
                "message": message,
                "error": str(error) if error else None,
                "failed_at": datetime.now().isoformat(),
            },
        )

    def _friendly_hint_for_error(exc: Exception) -> str | None:
        # Ultralytics model/weights incompatibility often shows up as a TypeError
        # during model parsing (e.g. SPPF signature mismatch).
        text = str(exc)
        if "SPPF.__init__" in text and "positional arguments" in text:
            return (
                "此權重檔與目前 server 上的 Ultralytics/YOLO 版本不相容。"
                "請先用 yolov8n.pt / yolov8s.pt 測試流程是否可正常訓練，"
                "或安裝與該權重相容的 Ultralytics 版本後再試。"
            )
        return None

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        msg = "Ultralytics is not installed on the server. Install `ultralytics` to enable training."
        logger.exception(msg)
        _fail(msg, exc)
        raise

    data_path = Path(data_yaml)
    if not data_path.exists():
        msg = f"data.yaml not found: {data_yaml}"
        _fail(msg)
        raise FileNotFoundError(msg)

    if job is not None:
        job.meta.update(
            {
                "status": "running",
                "message": "Training in progress",
                "run_dir": str(run_dir),
                "params": {
                    "base_weights": base_weights,
                    "data_yaml": str(data_path),
                    "epochs": int(epochs),
                    "imgsz": int(imgsz),
                    "batch": int(batch),
                },
            }
        )
        job.save_meta()
    _write_run_meta(
        run_dir,
        {
            "status": "running",
            "message": "Training in progress",
            "run_dir": str(run_dir),
        },
    )

    try:
        model = YOLO(base_weights)
        model.train(
            data=str(data_path),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            project=str(run_dir),
            name="train",
        )
    except Exception as exc:
        hint = _friendly_hint_for_error(exc)
        msg = "Training failed."
        if hint:
            msg = f"{msg} {hint}"
        logger.exception("Training job failed")
        _fail(msg, exc)
        raise

    # trainer.save_dir usually points to .../<run_dir>/train
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    save_dir = Path(save_dir) if save_dir else (run_dir / "train")

    artifacts_dir = run_dir / "artifacts"
    _safe_mkdir(artifacts_dir)

    weights_dir = save_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    best_dst = artifacts_dir / "best.pt"
    last_dst = artifacts_dir / "last.pt"

    _copy_if_exists(best_src, best_dst)
    _copy_if_exists(last_src, last_dst)

    for name in [
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]:
        _copy_if_exists(save_dir / name, artifacts_dir / name)

    # Evaluate (val) using best weights if available
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id,
        "base_weights": base_weights,
        "data_yaml": str(data_path),
    }
    try:
        eval_weights = str(best_dst) if best_dst.exists() else (str(last_dst) if last_dst.exists() else base_weights)
        val_res = YOLO(eval_weights).val(data=str(data_path), imgsz=int(imgsz))
        box = getattr(val_res, "box", None) or getattr(val_res, "metrics", None)
        for key in ("map50", "map", "map75", "mp", "mr"):
            val = getattr(box, key, None) if box is not None else getattr(val_res, key, None)
            if val is None:
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except Exception:
                metrics[key] = val
    except Exception as exc:
        metrics["val_error"] = str(exc)

    # Add a hint if metrics are missing/non-finite
    if any(metrics.get(k) is None for k in ("map50", "map", "mp", "mr")):
        metrics["warning"] = (
            "部分指標為空值/NaN，常見原因是驗證集過小或沒有標註。"
            "請確認 data.yaml 的 val.txt 內至少有 1 張圖片且有對應標註。"
        )

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    result = {
        "project_id": project_id,
        "job_id": job.id if job is not None else None,
        "status": "finished",
        "run_dir": str(run_dir),
        "artifacts_dir": str(artifacts_dir),
        "best_path": str(best_dst) if best_dst.exists() else None,
        "last_path": str(last_dst) if last_dst.exists() else None,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }

    if job is not None:
        job.meta.update(
            {
                "status": "finished",
                "message": "Training finished",
                "artifacts_dir": str(artifacts_dir),
                "best_path": result["best_path"],
                "last_path": result["last_path"],
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )
        job.save_meta()
    _write_run_meta(
        run_dir,
        {
            "status": "finished",
            "message": "Training finished",
            "run_dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
            "best_path": result["best_path"],
            "last_path": result["last_path"],
            "metrics_path": str(metrics_path),
            "metrics": metrics,
            "finished_at": datetime.now().isoformat(),
        },
    )

    return result

