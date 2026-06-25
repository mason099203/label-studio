"""YOLO training execution for the standalone Train Server."""

from __future__ import annotations

import gc
import json
import logging
import math
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

from .yolo_catalog import (
    build_training_params,
    reconcile_yolo_task,
    resolve_train_data_path,
    resolve_yolo_task,
    resolve_yolo_weights_name,
)
from .progress import collect_preview_images

logger = logging.getLogger(__name__)


def _get_ultralytics_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install `ultralytics` in the Train Server image."
        ) from exc


def _disable_ultralytics_git_metadata() -> None:
    try:
        from ultralytics.utils import GIT

        for k in ("head", "branch", "commit", "origin"):
            GIT.__dict__.pop(k, None)
        GIT.root = None
        GIT.gitdir = None
    except Exception:
        return


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_run_meta(run_dir: Path, payload: Dict[str, Any]) -> None:
    path = run_dir / "run_meta.json"
    current: Dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(payload)
    _write_json(path, current)


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


def _release_training_resources() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _download_weights_http(model_name: str, dest: Path) -> bool:
    """
    不透過 torch/Ultralytics 載入，直接從 GitHub assets 下載 .pt（Windows 上 torch DLL 異常時仍可用）。
    """
    tags = ("v8.3.0", "v8.4.0", "v0.0.0")
    for tag in tags:
        url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{model_name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "label-studio-train-server"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 1024:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            logger.info("Downloaded weights via HTTP: %s -> %s", url, dest)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.debug("HTTP weight download failed for %s: %s", url, exc)
    return False


def ensure_yolo_weights(model_name: str, models_dir: Path) -> Path:
    """
    若 models_dir 中無權重，先 HTTP 下載；失敗再透過 Ultralytics YOLO 自動下載。
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    local = models_dir / model_name
    if local.exists():
        return local

    if _download_weights_http(model_name, local):
        return local

    try:
        YOLO = _get_ultralytics_yolo()
    except ImportError as exc:
        raise ImportError(
            f"Cannot download {model_name}. HTTP download failed and Ultralytics unavailable: {exc}"
        ) from exc

    _disable_ultralytics_git_metadata()
    try:
        model = YOLO(model_name)
    except OSError as exc:
        hint = (
            "PyTorch failed to load on this machine (common on Windows: c10.dll / WinError 1114). "
            "Run: .\\scripts\\install-torch-cpu-windows.ps1 then restart Train Server."
        )
        raise ImportError(f"{hint} Original error: {exc}") from exc

    downloaded = getattr(model, "ckpt_path", None) or getattr(model, "model", None)
    src = Path(str(downloaded)) if downloaded else None
    if src and src.exists() and src.resolve() != local.resolve():
        shutil.copy2(src, local)
    elif not local.exists():
        cwd_candidate = Path.cwd() / model_name
        if cwd_candidate.exists():
            shutil.copy2(cwd_candidate, local)
        else:
            raise FileNotFoundError(f"Failed to download or locate weights: {model_name}")
    return local


def _patch_dataset_paths(dataset_root: Path, dataset_meta: Dict[str, Any]) -> None:
    """解壓後更新 dataset_config / data.yaml 中的絕對路徑。"""
    dataset_config = dataset_root / "dataset_config.json"
    if dataset_config.exists():
        meta = json.loads(dataset_config.read_text(encoding="utf-8"))
        meta["dataset_root"] = str(dataset_root)
        if meta.get("data_yaml"):
            meta["data_yaml"] = str(dataset_root / "data.yaml")
        if meta.get("train_dir"):
            meta["train_dir"] = str(dataset_root / "train")
        if meta.get("val_dir"):
            meta["val_dir"] = str(dataset_root / "val")
        _write_json(dataset_config, meta)
        dataset_meta.update(meta)

    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        text = data_yaml.read_text(encoding="utf-8")
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("path:"):
                lines.append(f"path: {dataset_root.resolve()}")
            else:
                lines.append(line)
        data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_yolo_training(
    *,
    project_id: int,
    job_id: str,
    base_weights: str,
    dataset_root: Path,
    dataset_meta: Dict[str, Any],
    output_root: Path,
    param_overrides: Dict[str, Any] | None = None,
    on_status: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    training_model = str(
        (param_overrides or {}).get("training_model")
        or dataset_meta.get("training_model")
        or "yolo_detect"
    )
    task_type = dataset_meta.get("task_type")
    task = reconcile_yolo_task(training_model, task_type, base_weights)
    train_params = build_training_params(task=task, overrides=param_overrides)

    _patch_dataset_paths(dataset_root, dataset_meta)

    run_dir = output_root / f"project_{project_id}" / job_id
    _safe_mkdir(run_dir)

    run_name = (param_overrides or {}).get("run_name") or (param_overrides or {}).get("name")

    def _emit(payload: Dict[str, Any]) -> None:
        meta_update = {
            "job_id": job_id,
            "project_id": project_id,
            "status": payload.get("status"),
            "message": payload.get("message"),
            "task": task,
            "training_model": training_model,
            "params": {**train_params, "base_weights": base_weights},
            "run_dir": str(run_dir),
        }
        if run_name:
            meta_update["name"] = str(run_name)
        if payload.get("epoch") is not None:
            meta_update["epoch"] = payload.get("epoch")
        if payload.get("total_epochs") is not None:
            meta_update["total_epochs"] = payload.get("total_epochs")
        if payload.get("progress_pct") is not None:
            meta_update["progress_pct"] = payload.get("progress_pct")
        _merge_run_meta(run_dir, meta_update)
        if on_status:
            on_status(payload)

    _merge_run_meta(
        run_dir,
        {
            "job_id": job_id,
            "project_id": project_id,
            "status": "starting",
            "message": f"Starting YOLO {task} training",
            "task": task,
            "training_model": training_model,
            "params": {**train_params, "base_weights": base_weights},
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            **({"name": str(run_name)} if run_name else {}),
        },
    )

    _emit(
        {
            "status": "starting",
            "message": f"Starting YOLO {task} training",
            "task": task,
            "training_model": training_model,
            "params": train_params,
        }
    )

    YOLO = _get_ultralytics_yolo()
    _disable_ultralytics_git_metadata()

    weights_path = resolve_yolo_weights_name(base_weights, task)
    if not Path(weights_path).exists():
        weights_path = str(ensure_yolo_weights(Path(weights_path).name, output_root.parent / "original"))

    task = reconcile_yolo_task(training_model, task_type, weights_path)
    train_params = build_training_params(task=task, overrides=param_overrides)
    data_path = resolve_train_data_path(task, dataset_meta, dataset_root)

    _emit({"status": "running", "message": "Training in progress", "run_dir": str(run_dir)})

    train_kwargs: Dict[str, Any] = {
        "data": data_path,
        "task": task,
        "epochs": int(train_params.get("epochs", 100)),
        "imgsz": int(train_params.get("imgsz", 640)),
        "batch": int(train_params.get("batch", 16)),
        "workers": int(train_params.get("workers", 0)),
        "patience": int(train_params.get("patience", 100)),
        "project": str(run_dir),
        "name": "train",
    }
    for key in (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "kobj",
        "mask_ratio",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "close_mosaic",
        "amp",
        "seed",
    ):
        if key in train_params:
            train_kwargs[key] = train_params[key]

    if train_params.get("device"):
        train_kwargs["device"] = train_params["device"]

    total_epochs = int(train_kwargs["epochs"])
    progress_path = run_dir / "progress.json"

    def _write_progress(epoch: int, trainer_metrics: Dict[str, Any] | None = None) -> None:
        metrics_dict = dict(trainer_metrics or {})
        progress_pct = round(epoch / total_epochs * 100, 1) if total_epochs else 0
        payload = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress_pct": progress_pct,
            "latest_metrics": metrics_dict,
            "message": f"Epoch {epoch}/{total_epochs}",
            "updated_at": datetime.now().isoformat(),
        }
        _write_json(progress_path, payload)
        train_sub = run_dir / "train"
        if train_sub.exists():
            collect_preview_images(train_sub, dest_dir=run_dir / "artifacts" / "live")
        _emit({**payload, "status": "running", "run_dir": str(run_dir)})

    model = YOLO(weights_path)

    def on_train_epoch_end(trainer) -> None:
        try:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            raw_metrics = getattr(trainer, "metrics", None) or {}
            if hasattr(raw_metrics, "items"):
                metrics_dict = dict(raw_metrics)
            else:
                metrics_dict = {}
            _write_progress(epoch, metrics_dict)
        except Exception as exc:
            logger.debug("Progress callback error: %s", exc)

    def on_train_start(trainer) -> None:
        _write_progress(0, {})
        _emit({"status": "running", "message": "Training started", "run_dir": str(run_dir), "total_epochs": total_epochs})

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_start", on_train_start)
    model.train(**train_kwargs)

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    save_dir = Path(save_dir) if save_dir else (run_dir / "train")

    artifacts_dir = run_dir / "artifacts"
    _safe_mkdir(artifacts_dir)

    weights_dir = save_dir / "weights"
    best_dst = artifacts_dir / "best.pt"
    last_dst = artifacts_dir / "last.pt"
    _copy_if_exists(weights_dir / "best.pt", best_dst)
    _copy_if_exists(weights_dir / "last.pt", last_dst)
    _copy_common_artifacts(save_dir, artifacts_dir)

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id,
        "task": task,
        "training_model": training_model,
        "base_weights": base_weights,
        "data_path": data_path,
    }

    try:
        eval_weights = str(best_dst) if best_dst.exists() else (str(last_dst) if last_dst.exists() else weights_path)
        val_res = YOLO(eval_weights).val(data=data_path, imgsz=int(train_params.get("imgsz", 640)), workers=0, task=task)
        if task == "classify":
            for key in ("top1", "top5", "fitness"):
                val = getattr(val_res, key, None)
                if val is not None:
                    try:
                        f = float(val)
                        metrics[key] = f if math.isfinite(f) else None
                    except Exception:
                        metrics[key] = val
        else:
            for key in ("map50", "map", "map75", "mp", "mr", "fitness"):
                box = getattr(val_res, "box", None)
                if box is not None and hasattr(box, key):
                    val = getattr(box, key)
                else:
                    val = getattr(val_res, key, None)
                if val is not None:
                    try:
                        f = float(val)
                        metrics[key] = f if math.isfinite(f) else None
                    except Exception:
                        metrics[key] = val
    except Exception as exc:
        metrics["val_error"] = str(exc)
        logger.warning("Validation after training failed: %s", exc)

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    run_meta = {
        "job_id": job_id,
        "project_id": project_id,
        "status": "finished",
        "message": "Training finished",
        "task": task,
        "training_model": training_model,
        "params": {**train_params, "base_weights": base_weights},
        "run_dir": str(run_dir),
        "artifacts_dir": str(artifacts_dir),
        "best_path": str(best_dst) if best_dst.exists() else None,
        "last_path": str(last_dst) if last_dst.exists() else None,
        "metrics": metrics,
        "finished_at": datetime.now().isoformat(),
        "progress_pct": 100,
    }
    if run_name:
        run_meta["name"] = str(run_name)
    _merge_run_meta(run_dir, run_meta)

    _release_training_resources()
    _emit(run_meta)
    return run_meta
