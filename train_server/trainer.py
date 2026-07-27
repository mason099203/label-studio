"""YOLO training execution for the standalone Train Server."""

from __future__ import annotations

import gc
import json
import logging
import math
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from .yolo_catalog import (
    build_training_params,
    reconcile_yolo_task,
    resolve_train_data_path,
    resolve_yolo_task,
    resolve_yolo_weights_name,
)
from .device_utils import resolve_dataloader_workers, resolve_training_device
from .metrics_utils import collect_yolo_val_metrics, json_safe, normalize_trainer_metrics
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
    safe = json_safe(data)
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _resolve_split_line(root: Path, raw: str, image_exts: set[str]) -> Path | None:
    """Resolve one train/val line to an existing image file under dataset root."""
    line = raw.strip().replace("\\", "/")
    if not line:
        return None
    p = Path(line)
    candidate = p if p.is_absolute() else root / line
    if candidate.is_file():
        return candidate
    stem = p.stem
    parent = p.parent if str(p.parent) not in {"", "."} else Path("images")
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"):
        alt = root / parent / f"{stem}{ext}"
        if alt.is_file():
            return alt
    matches = [
        m
        for m in root.rglob(f"{stem}.*")
        if m.is_file() and m.suffix.lower() in image_exts
    ]
    if matches:
        return next((m for m in matches if "images" in m.parts), matches[0])
    matches = [m for m in root.rglob(p.name) if m.is_file()]
    if matches:
        return next((m for m in matches if "images" in m.parts), matches[0])
    return None


def _normalize_split_files(dataset_root: Path) -> None:
    """Rewrite train.txt / val.txt so paths resolve under dataset_root (remote Train Server)."""
    root = dataset_root.resolve()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}

    for name in ("train.txt", "val.txt"):
        split_file = root / name
        if not split_file.exists():
            continue
        resolved: list[str] = []
        for raw in split_file.read_text(encoding="utf-8").splitlines():
            match = _resolve_split_line(root, raw, image_exts)
            if match is None:
                continue
            try:
                resolved.append(match.resolve().relative_to(root).as_posix())
            except ValueError:
                resolved.append(str(match.resolve()))
        split_file.write_text("\n".join(resolved) + ("\n" if resolved else ""), encoding="utf-8")


def _absolutize_split_files(dataset_root: Path) -> None:
    """Rewrite train/val lists with absolute paths so Ultralytics finds files regardless of cwd."""
    root = dataset_root.resolve()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}

    for name in ("train.txt", "val.txt"):
        split_file = root / name
        if not split_file.exists():
            continue
        abs_lines: list[str] = []
        for raw in split_file.read_text(encoding="utf-8").splitlines():
            match = _resolve_split_line(root, raw, image_exts)
            if match is None or not match.is_file():
                continue
            try:
                if match.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            abs_lines.append(str(match.resolve()))
        if split_file.read_text(encoding="utf-8").strip() and not abs_lines:
            raise FileNotFoundError(
                f"No resolvable image paths in {split_file.name} under {root}. "
                "The dataset zip may be missing images/ — regenerate in Label Studio."
            )
        split_file.write_text("\n".join(abs_lines) + ("\n" if abs_lines else ""), encoding="utf-8")


def _clear_ultralytics_caches(dataset_root: Path) -> None:
    for cache in dataset_root.rglob("*.cache"):
        try:
            cache.unlink()
        except OSError:
            pass


def _validate_dataset_images(dataset_root: Path) -> None:
    root = dataset_root.resolve()
    missing: list[str] = []
    total = 0
    for name in ("train.txt", "val.txt"):
        split_file = root / name
        if not split_file.exists():
            continue
        for raw in split_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip().replace("\\", "/")
            if not line:
                continue
            total += 1
            p = Path(line)
            candidate = p if p.is_absolute() else root / line
            if candidate.is_file():
                try:
                    if candidate.stat().st_size > 0:
                        continue
                except OSError:
                    pass
            missing.append(line)
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)}/{total} training images are missing under {root} (e.g. {sample}). "
            "Regenerate the dataset in Label Studio (Training → 生成訓練資料集) and retry."
        )


def _patch_dataset_paths(dataset_root: Path, dataset_meta: Dict[str, Any]) -> None:
    """解壓後更新 dataset_config / data.yaml 中的絕對路徑。"""
    from .dataset_images import canonicalize_yolo_images_dir

    images_dir = dataset_root / "images"
    if images_dir.is_dir():
        canonicalize_yolo_images_dir(images_dir)

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

    _clear_ultralytics_caches(dataset_root)
    _normalize_split_files(dataset_root)
    _absolutize_split_files(dataset_root)
    _validate_dataset_images(dataset_root)

    images_dir = dataset_root / "images"
    if images_dir.is_dir():
        image_count = sum(1 for p in images_dir.rglob("*.jpg") if p.is_file())
        logger.info("Dataset ready: %s JPEG images under %s", image_count, images_dir)
        if image_count == 0:
            raise FileNotFoundError(
                f"No JPEG images under {images_dir}. Regenerate the training dataset in Label Studio."
            )


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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
    dataloader_workers = resolve_dataloader_workers(
        train_params.get("workers"),
        user_specified="workers" in param_overrides,
    )

    _emit({"status": "running", "message": "Training in progress", "run_dir": str(run_dir)})

    train_kwargs: Dict[str, Any] = {
        "data": data_path,
        "task": task,
        "epochs": int(train_params.get("epochs", 100)),
        "imgsz": int(train_params.get("imgsz", 640)),
        "batch": int(train_params.get("batch", 16)),
        "workers": dataloader_workers,
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

    train_kwargs["device"] = resolve_training_device(train_params.get("device"))

    total_epochs = int(train_kwargs["epochs"])
    progress_path = run_dir / "progress.json"

    def _write_progress(epoch: int, trainer_metrics: Dict[str, Any] | None = None) -> None:
        metrics_dict = normalize_trainer_metrics(trainer_metrics)
        progress_pct = round(epoch / total_epochs * 100, 1) if total_epochs else 0
        payload = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress_pct": progress_pct,
            "latest_metrics": metrics_dict,
            "message": f"Epoch {epoch}/{total_epochs}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
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
            raw_metrics = getattr(trainer, "metrics", None)
            _write_progress(epoch, normalize_trainer_metrics(raw_metrics))
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
    if not best_dst.exists():
        for candidate in sorted(save_dir.rglob("best.pt")):
            if candidate.is_file():
                _copy_if_exists(candidate, best_dst)
                break
    if not last_dst.exists():
        for candidate in sorted(save_dir.rglob("last.pt")):
            if candidate.is_file():
                _copy_if_exists(candidate, last_dst)
                break
    _copy_common_artifacts(save_dir, artifacts_dir)
    logger.info(
        "Training artifacts for job %s: best=%s last=%s dir=%s",
        job_id,
        best_dst.exists(),
        last_dst.exists(),
        artifacts_dir,
    )

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "task": task,
        "training_model": training_model,
        "base_weights": base_weights,
        "data_path": data_path,
    }

    try:
        eval_weights = str(best_dst) if best_dst.exists() else (str(last_dst) if last_dst.exists() else weights_path)
        val_res = YOLO(eval_weights).val(data=data_path, imgsz=int(train_params.get("imgsz", 640)), workers=0, task=task)
        metrics.update(collect_yolo_val_metrics(val_res, task))
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
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "progress_pct": 100,
    }
    if run_name:
        run_meta["name"] = str(run_name)
    _merge_run_meta(run_dir, run_meta)

    _release_training_resources()
    _emit(run_meta)
    return run_meta
