"""RQ jobs for on-server model training and evaluation."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rq import get_current_job

logger = logging.getLogger(__name__)


def _get_ultralytics_yolo():
    """
    載入 ultralytics.YOLO。

    將 ModuleNotFoundError 與其他匯入錯誤分開，避免把 torch/cuDNN/OpenCV 等問題一律顯示成「未安裝 ultralytics」。

    @returns YOLO 類別
    @raises ImportError 無法載入時（訊息依原因區分）
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]

        return YOLO
    except ModuleNotFoundError as exc:
        name = getattr(exc, "name", "") or ""
        if name == "ultralytics" or "ultralytics" in str(exc):
            raise ImportError(
                "Ultralytics is not installed on the server. Install `ultralytics` to enable training."
            ) from exc
        raise ImportError(
            f"訓練依賴缺少模組「{name}」。請確認伺服器已安裝 yolo-training 依賴（例如 poetry/Docker 映像含 ultralytics/torch）。原始錯誤: {exc}"
        ) from exc
    except Exception as exc:
        raise ImportError(
            "Ultralytics 匯入失敗（不一定是未安裝 ultralytics）。常見原因：PyTorch 與 CUDA/cuDNN 或 CPU 輪子不一致、動態庫缺失。"
            f" 詳情: {exc}"
        ) from exc


def _friendly_hint_for_yolo_error(exc: Exception, *, classification: bool = False) -> str | None:
    """
    依常見例外文字補充使用者可理解的提示（仍會一併寫入 meta.error 的原始訊息）。

    @param exc 訓練過程例外
    @param classification 是否為 YOLO 分類任務
    @returns 提示字串，若無則 None
    """
    text = str(exc)
    if "SPPF.__init__" in text and "positional arguments" in text:
        return (
            "此權重檔與目前 server 上的 Ultralytics/YOLO 版本不相容。"
            "請先用 yolov8n.pt / yolov8s.pt 測試流程是否可正常訓練，"
            "或安裝與該權重相容的 Ultralytics 版本後再試。"
        )
    if classification:
        tl = text.lower()
        if "classification datasets must be a directory" in tl:
            return (
                "分類訓練需要資料集根目錄（含 train/<類別>/ 與 val/<類別>/），"
                "不可使用 data.yaml。若為偵測/分割專案，請改用 yolo11n.pt / yolo11n-seg.pt，勿用 *-cls.pt。"
            )
        if "no labels" in tl or "found 0 images" in tl or "no images found" in tl:
            return (
                "資料夾結構或影像數量可能有誤。YOLO 分類需：dataset_root/train/<類別名>/<圖片> 與 "
                "dataset_root/val/<類別名>/<圖片>，且每類至少各 1 張。"
            )
        if "yaml" in tl and ("not found" in tl or "missing" in tl):
            return "分類訓練使用 ImageFolder 根目錄；請確認 dataset_config 的 dataset_root 指向正確目錄。"
    tl = text.lower()
    if "classification datasets must be a directory" in tl:
        return (
            "所選權重為分類模型（*-cls.pt），但資料集為偵測/分割格式（data.yaml）。"
            "請改用 yolo11n.pt、yolo11n-seg.pt 等與專案任務一致的權重。"
        )
    if "out of memory" in tl and ("cuda" in tl or "cudnn" in tl):
        return "GPU 記憶體不足，可嘗試調小 batch 或 imgsz，或改用 CPU 版 torch。"
    return None


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


def _release_training_resources() -> None:
    """
    Release Python / torch resources after training.

    Notes:
    - On Windows, DataLoader worker processes are often the biggest source of
      leftover `python.exe` processes, so we also train with `workers=0`.
    """

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


def _copy_common_artifacts(save_dir: Path, artifacts_dir: Path) -> None:
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


def _read_dataset_config(dataset_config: str) -> Dict[str, Any]:
    path = Path(dataset_config)
    if not path.exists():
        raise FileNotFoundError(f"dataset_config not found: {dataset_config}")
    return json.loads(path.read_text(encoding="utf-8"))


def _disable_ultralytics_git_metadata() -> None:
    """
    Disable Ultralytics' Git metadata reading.

    Ultralytics will read `.git` refs (for `GIT.commit`) when saving checkpoints.
    In restricted environments (e.g. RQ worker with limited filesystem permissions),
    reading `.git/refs/heads/...` may raise `PermissionError` and fail the training job.
    """

    try:
        from ultralytics.utils import GIT  # type: ignore

        # Clear cached properties to ensure they won't be re-used.
        for k in ("head", "branch", "commit", "origin"):
            GIT.__dict__.pop(k, None)

        # Mark as non-repo so GIT.commit returns None without reading git files.
        GIT.root = None
        GIT.gitdir = None
    except Exception:
        # Best-effort only; if it fails, training might still work.
        return


def _collect_yolo_val_metrics(val_res, task: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if task == "classify":
        for key in ("top1", "top5", "fitness"):
            val = getattr(val_res, key, None)
            if val is not None:
                try:
                    f = float(val)
                    metrics[key] = f if math.isfinite(f) else None
                except Exception:
                    metrics[key] = val
        return metrics

    metric_groups = []
    if task in {"detect", "obb", "pose"}:
        metric_groups.append(getattr(val_res, "box", None))
    if task in {"segment", "semantic"}:
        metric_groups.append(getattr(val_res, "seg", None) or getattr(val_res, "mask", None))
    if task == "semantic":
        metric_groups.append(getattr(val_res, "semantic", None))

    for group in metric_groups:
        if group is None:
            continue
        for key in ("map50", "map", "map75", "mp", "mr", "fitness"):
            val = getattr(group, key, None)
            if val is None:
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except Exception:
                metrics[key] = val

    if not metrics:
        for key in ("map50", "map", "map75", "mp", "mr", "fitness"):
            val = getattr(val_res, key, None)
            if val is not None:
                try:
                    f = float(val)
                    metrics[key] = f if math.isfinite(f) else None
                except Exception:
                    metrics[key] = val
    return metrics


def yolo_detect_train_job(
    *,
    project_id: int,
    base_weights: str,
    data_yaml: str | None = None,
    dataset_config: str | None = None,
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

    dataset_meta: Dict[str, Any] = {}
    if dataset_config:
        try:
            dataset_meta = _read_dataset_config(dataset_config)
        except Exception:
            dataset_meta = {}

    training_model = dataset_meta.get("training_model") or "yolo_detect"
    task_type = dataset_meta.get("task_type") or "detect"
    from .yolo_catalog import reconcile_yolo_task, resolve_train_data_path, resolve_yolo_task, resolve_yolo_weights_name

    yolo_task = resolve_yolo_task(training_model, task_type)
    job_kind_map = {
        "detect": "yolo_detect_train",
        "classification": "yolo_classification_train",
        "segmentation": "yolo_segmentation_train",
        "semantic_segmentation": "yolo_semantic_segmentation_train",
        "semantic": "yolo_semantic_segmentation_train",
        "pose": "yolo_pose_train",
        "obb": "yolo_obb_train",
    }
    job_kind = job_kind_map.get(task_type, f"yolo_{task_type}_train")

    if job is not None:
        job.meta.update(
            {
                "kind": job_kind,
                "project_id": project_id,
                "status": "starting",
                "message": "Starting training job",
                "created_at": now,
                "training_model": training_model,
                "task_type": task_type,
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
            "kind": job_kind,
            "training_model": training_model,
            "task_type": task_type,
            "task": yolo_task,
            "params": {
                "base_weights": base_weights,
                "data_yaml": str(dataset_meta.get("data_yaml") or data_yaml or ""),
                "dataset_config": dataset_config,
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "training_model": training_model,
                "task_type": task_type,
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

    try:
        YOLO = _get_ultralytics_yolo()
    except ImportError as exc:
        msg = str(exc)
        logger.exception(msg)
        _fail(msg, exc.__cause__ if exc.__cause__ is not None else exc)
        raise

    if dataset_config:
        dataset_meta = _read_dataset_config(dataset_config)
        data_yaml = dataset_meta.get("data_yaml")

    dataset_root = Path(str(dataset_meta.get("dataset_root") or ""))
    data_path = None
    if task_type == "classification" or yolo_task == "classify":
        if not dataset_root.exists():
            msg = f"classification dataset root not found: {dataset_root}"
            _fail(msg)
            raise FileNotFoundError(msg)
    else:
        data_path = Path(str(data_yaml)) if data_yaml else None
        if data_path is None or not data_path.exists():
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
                    "data_yaml": str(data_path) if data_path else str(dataset_root),
                    "dataset_config": dataset_config,
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
        # Prevent Ultralytics from reading `.git` refs when saving checkpoints.
        _disable_ultralytics_git_metadata()
        target_weights = resolve_yolo_weights_name(base_weights, yolo_task)
        try:
            p = Path(target_weights)
            if p.suffix.lower() == ".pt" and not p.exists() and p.parent != Path("."):
                alt = p.parent / p.name
                target_weights = str(alt) if alt.exists() else p.name
        except Exception:
            pass

        yolo_task = reconcile_yolo_task(training_model, task_type, target_weights)
        if yolo_task == "classify":
            train_data = resolve_train_data_path(yolo_task, dataset_meta, dataset_root)
        else:
            train_data = str(data_path)

        model = YOLO(target_weights)
        train_kwargs: Dict[str, Any] = {
            "data": train_data,
            "task": yolo_task,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": 0,
            "project": str(run_dir),
            "name": "train",
        }
        model.train(**train_kwargs)
    except Exception as exc:
        hint = _friendly_hint_for_yolo_error(exc, classification=(yolo_task == "classify"))
        msg = "Training failed."
        if hint:
            msg = f"{msg} {hint}"
        logger.exception("Training job failed")
        _fail(msg, exc)
        _release_training_resources()
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

    _copy_common_artifacts(save_dir, artifacts_dir)

    # Evaluate (val) using best weights if available
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id,
        "base_weights": base_weights,
        "data_yaml": str(data_path),
    }
    try:
        eval_weights = str(best_dst) if best_dst.exists() else (str(last_dst) if last_dst.exists() else target_weights)
        val_res = YOLO(eval_weights).val(data=str(data_path), imgsz=int(imgsz), workers=0, task=yolo_task)
        metrics.update(_collect_yolo_val_metrics(val_res, yolo_task))
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
    _release_training_resources()
    return result


def yolo_classification_train_job(
    *,
    project_id: int,
    base_weights: str,
    dataset_config: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    output_root: str,
) -> Dict[str, Any]:
    """
    Train a YOLO classification model using a JSON dataset manifest.
    """

    job = get_current_job()
    now = datetime.now().isoformat()

    if job is not None:
        job.meta.update(
            {
                "kind": "yolo_classification_train",
                "project_id": project_id,
                "status": "starting",
                "message": "Starting classification training job",
                "created_at": now,
            }
        )
        job.save_meta()

    output_root_path = Path(output_root)
    run_dir = output_root_path / f"project_{project_id}" / (job.id if job is not None else f"job_{now}")
    _safe_mkdir(run_dir)

    dataset_meta = _read_dataset_config(dataset_config)
    dataset_root = Path(str(dataset_meta.get("dataset_root") or ""))

    _write_run_meta(
        run_dir,
        {
            "job_id": job.id if job is not None else None,
            "project_id": project_id,
            "status": "starting",
            "message": "Starting classification training job",
            "created_at": now,
            "params": {
                "base_weights": base_weights,
                "dataset_config": dataset_config,
                "dataset_root": str(dataset_root),
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "task_type": "classification",
                "training_model": "yolo_classify",
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

    try:
        YOLO = _get_ultralytics_yolo()
    except ImportError as exc:
        msg = str(exc)
        logger.exception(msg)
        _fail(msg, exc.__cause__ if exc.__cause__ is not None else exc)
        raise

    if not dataset_root.exists():
        msg = f"classification dataset root not found: {dataset_root}"
        _fail(msg)
        raise FileNotFoundError(msg)

    if job is not None:
        job.meta.update(
            {
                "status": "running",
                "message": "Classification training in progress",
                "run_dir": str(run_dir),
                "params": {
                    "base_weights": base_weights,
                    "dataset_config": dataset_config,
                    "dataset_root": str(dataset_root),
                    "epochs": int(epochs),
                    "imgsz": int(imgsz),
                    "batch": int(batch),
                    "task_type": "classification",
                    "training_model": "yolo_classify",
                },
            }
        )
        job.save_meta()
    _write_run_meta(
        run_dir,
        {
            "status": "running",
            "message": "Classification training in progress",
            "run_dir": str(run_dir),
        },
    )

    try:
        target_weights = base_weights
        try:
            p = Path(base_weights)
            if p.suffix.lower() == ".pt" and not p.name.endswith("-cls.pt"):
                cls_name = p.stem + "-cls.pt"
                alt = p.parent / cls_name
                target_weights = str(alt) if alt.exists() else cls_name
        except Exception:
            pass

        # Prevent Ultralytics from reading `.git` refs when saving checkpoints.
        _disable_ultralytics_git_metadata()
        model = YOLO(target_weights)
        model.train(
            data=str(dataset_root),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            workers=0,
            project=str(run_dir),
            name="train",
        )
    except Exception as exc:
        hint = _friendly_hint_for_yolo_error(exc, classification=True)
        msg = "Classification training failed."
        if hint:
            msg = f"{msg} {hint}"
        logger.exception("Classification training job failed")
        _fail(msg, exc)
        _release_training_resources()
        raise

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
    _copy_common_artifacts(save_dir, artifacts_dir)

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id,
        "base_weights": base_weights,
        "dataset_config": dataset_config,
        "dataset_root": str(dataset_root),
    }
    try:
        eval_weights = str(best_dst) if best_dst.exists() else (str(last_dst) if last_dst.exists() else base_weights)
        val_res = YOLO(eval_weights).val(data=str(dataset_root), imgsz=int(imgsz), workers=0)
        for key in ("top1", "top5", "fitness"):
            val = getattr(val_res, key, None)
            if val is None:
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except Exception:
                metrics[key] = val
    except Exception as exc:
        metrics["val_error"] = str(exc)

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
                "message": "Classification training finished",
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
            "message": "Classification training finished",
            "run_dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
            "best_path": result["best_path"],
            "last_path": result["last_path"],
            "metrics_path": str(metrics_path),
            "metrics": metrics,
            "finished_at": datetime.now().isoformat(),
        },
    )

    _release_training_resources()
    return result


def cnn_classification_train_job(
    *,
    project_id: int,
    base_weights: str,
    dataset_config: str,
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 32,
    output_root: str,
    device: Optional[str] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Train a custom PyTorch CNN classification model.

    @param {int} project_id      - Label Studio 專案 ID
    @param {str} base_weights    - 基礎權重（目前保留，供未來 fine-tune 使用）
    @param {str} dataset_config  - dataset_config JSON 路徑
    @param {int} epochs          - 訓練回合數（預設 50）
    @param {int} imgsz           - 圖片尺寸（CNN 固定為 224）
    @param {int} batch           - 每批次樣本數（預設 32）
    @param {str} output_root     - 輸出根目錄
    @param {Optional[str]} device - 計算裝置："cuda" | "cpu" | None（None 時自動偵測）
    @param {bool} use_amp        - 是否啟用 AMP 混合精度訓練（CUDA 裝置才有效，預設 True）
    """
    from .cnn_trainer import CNNTrainer

    job = get_current_job()
    now = datetime.now().isoformat()

    if job is not None:
        job.meta.update({
            "kind": "cnn_classification_train",
            "project_id": project_id,
            "status": "starting",
            "message": "Starting CNN classification training job",
            "created_at": now,
        })
        job.save_meta()

    output_root_path = Path(output_root)
    run_dir = output_root_path / f"project_{project_id}" / (job.id if job is not None else f"job_{now}")
    _safe_mkdir(run_dir)

    dataset_meta = _read_dataset_config(dataset_config)
    items_path = Path(dataset_meta.get("items_json") or "")
    if not items_path.exists():
        raise FileNotFoundError(f"Items JSON not found: {items_path}")
    
    items = json.loads(items_path.read_text(encoding="utf-8"))
    classes = dataset_meta.get("classes", [])

    def _reporter(epoch, total, t_loss, t_acc, v_loss, v_acc):
        if job is not None:
            job.meta.update({
                "status": "running",
                "message": f"Epoch {epoch}/{total}: Val Acc {v_acc:.2f}%",
                "progress": epoch / total,
                "metrics": {"top1": v_acc / 100.0, "loss": v_loss}
            })
            job.save_meta()

    trainer = CNNTrainer(
        project_id=project_id,
        run_dir=run_dir,
        class_names=classes,
        batch_size=batch,
        learning_rate=0.001,
        num_epochs=epochs,
        device=device,
        use_amp=use_amp,
        job_reporter=_reporter,
    )

    # Simple train/val split is usually handled in datasets.py for yolo_classify,
    # but here we rely on the manifest.
    # We will split here for cnn if not already split.
    import random
    random.seed(42)
    random.shuffle(items)
    split = int(len(items) * 0.8)
    train_items = items[:split]
    val_items = items[split:]

    try:
        model, best_acc = trainer.run(train_items, val_items)
    except Exception as exc:
        logger.exception("CNN training failed")
        if job:
            job.meta.update({"status": "failed", "message": str(exc)})
            job.save_meta()
        raise

    artifacts_dir = run_dir / "artifacts"
    _safe_mkdir(artifacts_dir)
    
    # Copy best.pt (TorchScript) and best.pth to artifacts
    for f in ["best.pt", "best.pth"]:
        src = run_dir / f
        if src.exists():
            shutil.copy2(src, artifacts_dir / f)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "top1": best_acc / 100.0,
        "classes": classes
    }
    _write_json(run_dir / "metrics.json", metrics)

    if job is not None:
        job.meta.update({
            "status": "finished",
            "message": "CNN training finished",
            "artifacts_dir": str(artifacts_dir),
            "best_path": str(artifacts_dir / "best.pt"),
            "metrics": metrics,
        })
        job.save_meta()

    return {"status": "finished", "run_dir": str(run_dir)}

