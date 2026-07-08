"""HTTP client for the standalone Train Server."""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

logger = logging.getLogger(__name__)

TRAIN_SERVER_URL = os.environ.get("TRAIN_SERVER_URL", "").strip().rstrip("/")
TRAIN_SERVER_API_KEY = os.environ.get("TRAIN_SERVER_API_KEY", "").strip()
TRAIN_SERVER_TIMEOUT = float(os.environ.get("TRAIN_SERVER_TIMEOUT", "120"))
TRAIN_SERVER_SHARED_DATA_ROOT = os.environ.get("TRAIN_SERVER_SHARED_DATA_ROOT", "").strip()
TRAIN_SERVER_FORCE_SHARED_PATH = os.environ.get("TRAIN_SERVER_FORCE_SHARED_PATH", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_local_train_server_url(base_url: str) -> bool:
    """Train Server 跑在本機（非 Docker 掛載）時可用 host 絕對路徑。"""
    if not base_url:
        return False
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")


def _is_windows_absolute_path(path: str) -> bool:
    if len(path) >= 2 and path[1] == ":":
        return True
    return path.startswith("\\\\")


_train_server_output_root_cache: Dict[str, str | None] = {}


def _probe_train_server_output_root(train_server_url: str) -> str | None:
    """Probe Train Server /models for output_root (cached). Used to detect containerized layout."""
    url = (train_server_url or TRAIN_SERVER_URL or "").strip().rstrip("/")
    if not url:
        return None
    if url in _train_server_output_root_cache:
        return _train_server_output_root_cache[url]
    output_root: str | None = None
    try:
        cfg = resolve_train_server_config(url=url)
        resp = _request(cfg, "GET", "/models", timeout=8, retries=1)
        if resp.ok:
            raw = resp.json().get("output_root")
            if isinstance(raw, str) and raw.strip():
                output_root = raw.strip()
    except Exception as exc:
        logger.debug("Could not probe Train Server output_root at %s: %s", url, exc)
    _train_server_output_root_cache[url] = output_root
    return output_root


def _train_server_looks_containerized(train_server_url: str) -> bool:
    output_root = _probe_train_server_output_root(train_server_url)
    return bool(output_root and output_root.startswith("/"))


def _shared_path_unusable_for_train_server(path: str, train_server_url: str) -> bool:
    """Return True when path cannot exist on the Train Server host (e.g. Windows path to remote Linux)."""
    server_url = (train_server_url or TRAIN_SERVER_URL or "").strip()
    if _is_windows_absolute_path(path) and not _is_local_train_server_url(server_url):
        return True
    if _is_windows_absolute_path(path) and _train_server_looks_containerized(server_url):
        return True
    return False


def _resolve_shared_dataset_path(dataset_root: Path, *, train_server_url: str = "") -> str | None:
    """
    當 Label Studio 與 Train Server 共用 data/ 目錄時，傳 dataset_path 略過 zip。

    - **遠端 Train Server**（非 localhost）：預設回傳 None → 改上傳 zip。
      僅在雙方掛載同一 NFS/共享儲存時，設 `TRAIN_SERVER_SHARED_DATA_ROOT` +
      `TRAIN_SERVER_FORCE_SHARED_PATH=1`，傳 Train Server 端可見的 POSIX 路徑。
    - **本機 Docker Train Server**（localhost + ./data:/data）：設 `TRAIN_SERVER_SHARED_DATA_ROOT=/data`
    - **本機原生 Train Server**（localhost + start-train-server.ps1）：可用 host 絕對路徑
    """
    if not dataset_root.is_dir():
        return None
    repo_data = _repo_root() / "data"
    try:
        rel = dataset_root.resolve().relative_to(repo_data.resolve())
    except ValueError:
        return None

    shared_root = TRAIN_SERVER_SHARED_DATA_ROOT
    server_url = (train_server_url or TRAIN_SERVER_URL or "").strip()
    is_local = _is_local_train_server_url(server_url)

    if shared_root:
        if not is_local and not TRAIN_SERVER_FORCE_SHARED_PATH:
            logger.info(
                "Remote Train Server (%s): skip shared dataset_path; will upload zip instead. "
                "Set TRAIN_SERVER_FORCE_SHARED_PATH=1 only if Train Server mounts the same storage.",
                server_url,
            )
            return None
        mapped = f"{shared_root.rstrip('/')}/{rel.as_posix()}"
        logger.info("Mapped dataset to Train Server path: %s", mapped)
        return mapped

    if is_local:
        host_path = str(dataset_root.resolve())
        if _shared_path_unusable_for_train_server(host_path, server_url):
            logger.info(
                "Train Server on localhost looks containerized; uploading dataset zip instead of Windows path. "
                "To use shared dataset_path, bind-mount repo data to /data and set TRAIN_SERVER_SHARED_DATA_ROOT=/data.",
            )
            return None
        logger.info("Using host dataset path for local Train Server: %s", host_path)
        return host_path

    return None


def _count_images_under(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _list_zip_image_entries(names: List[str]) -> List[str]:
    entries: List[str] = []
    for raw in names:
        norm = raw.replace("\\", "/").lstrip("./")
        if not norm or norm.endswith("/"):
            continue
        lower = norm.lower()
        if lower.startswith("images/"):
            entries.append(raw)
            continue
        if "/images/" in lower:
            parts = lower.split("/")
            if "images" in parts and parts.index("images") < len(parts) - 1:
                entries.append(raw)
    return entries


def _zip_entry_matches_split_line(entry: str, line: str) -> bool:
    norm_entry = entry.replace("\\", "/").lstrip("./")
    norm_line = line.replace("\\", "/").lstrip("./")
    return norm_entry == norm_line or norm_entry.endswith(f"/{norm_line}")


def _is_classification_dataset(dataset_meta: Dict[str, Any] | None, dataset_root: Path) -> bool:
    meta = dataset_meta or {}
    if meta.get("training_model") == "yolo_classify" or meta.get("task_type") == "classification":
        return True
    root = Path(dataset_root)
    return (root / "train").is_dir() and (root / "val").is_dir() and not (root / "images").is_dir()


def _list_zip_classification_image_entries(names: List[str]) -> List[str]:
    entries: List[str] = []
    for raw in names:
        norm = raw.replace("\\", "/").lstrip("./")
        if not norm or norm.endswith("/"):
            continue
        lower = norm.lower()
        if lower.startswith("train/") or lower.startswith("val/"):
            if Path(norm).suffix.lower() in _IMAGE_EXTS:
                entries.append(raw)
    return entries


def _list_zip_dataset_image_entries(names: List[str], *, classification: bool) -> List[str]:
    if classification:
        return _list_zip_classification_image_entries(names)
    return _list_zip_image_entries(names)


def _verify_zip_includes_split_images(
    buffer: io.BytesIO,
    dataset_root: Path,
    *,
    classification: bool = False,
) -> None:
    """Ensure every path listed in train/val.txt is present in the zip archive."""
    buffer.seek(0)
    missing: List[str] = []
    total = 0
    with zipfile.ZipFile(buffer, "r") as zf:
        names = list(zf.namelist())
        image_entries = _list_zip_dataset_image_entries(names, classification=classification)
        if not image_entries:
            sample = ", ".join(names[:8])
            layout = "train/ and val/" if classification else "images/"
            raise FileNotFoundError(
                f"Dataset zip contains no image files under {layout} "
                f"Sample zip entries: {sample}. Regenerate the training dataset before submitting."
            )
        if classification:
            buffer.seek(0)
            return
        for split in ("train.txt", "val.txt"):
            split_path = dataset_root / split
            if not split_path.exists():
                continue
            for raw in split_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip().replace("\\", "/")
                if not line:
                    continue
                total += 1
                if line in names or any(_zip_entry_matches_split_line(n, line) for n in names):
                    continue
                missing.append(line)
    buffer.seek(0)
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Dataset zip is missing {len(missing)}/{total} images referenced in train/val splits (e.g. {sample}). "
            "Regenerate the training dataset before submitting to Train Server."
        )


def _zip_dataset_dir(dataset_root: Path, dataset_meta: Dict[str, Any] | None = None) -> io.BytesIO:
    classification = _is_classification_dataset(dataset_meta, dataset_root)
    if classification:
        image_count = _count_images_under(dataset_root / "train") + _count_images_under(dataset_root / "val")
    else:
        image_count = _count_images_under(dataset_root / "images")
        if image_count == 0:
            image_count = _count_images_under(dataset_root)
    if image_count == 0:
        raise FileNotFoundError(
            f"No image files under {dataset_root}. Regenerate the training dataset "
            "(Training → 生成訓練資料集) and ensure CONVERTER_DOWNLOAD_RESOURCES=true."
        )

    buffer = io.BytesIO()
    zipped_images = 0
    skipped = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in dataset_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(dataset_root)
            if rel.parts and rel.parts[0] == "export":
                continue
            if path.suffix.lower() == ".zip":
                continue
            arcname = rel.as_posix()
            real = path.resolve() if path.is_symlink() else path
            if not real.is_file():
                skipped += 1
                continue
            try:
                if real.stat().st_size <= 0:
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue
            zf.write(real, arcname)
            if real.suffix.lower() in _IMAGE_EXTS:
                zipped_images += 1
    if skipped:
        logger.warning("Skipped %s broken/empty files when zipping dataset from %s", skipped, dataset_root)
    if zipped_images == 0:
        raise FileNotFoundError(
            f"Dataset zip would contain 0 images from {dataset_root}. "
            "Regenerate the training dataset before submitting to Train Server."
        )
    buffer.seek(0)
    with zipfile.ZipFile(buffer, "r") as zf:
        if not _list_zip_dataset_image_entries(zf.namelist(), classification=classification):
            layout = "train/ and val/" if classification else "images/"
            raise FileNotFoundError(
                f"Dataset zip from {dataset_root} has no image entries under {layout} "
                "Regenerate the training dataset before submitting."
            )
    buffer.seek(0)
    logger.info(
        "Prepared dataset zip: %s images from %s (classify=%s)",
        zipped_images,
        dataset_root,
        classification,
    )
    _verify_zip_includes_split_images(buffer, dataset_root, classification=classification)
    buffer.seek(0)
    return buffer


@dataclass(frozen=True)
class TrainServerConfig:
    base_url: str
    api_key: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)


def resolve_train_server_config(
    *,
    url: str | None = None,
    api_key: str | None = None,
) -> TrainServerConfig:
    base = (url or TRAIN_SERVER_URL or "").strip().rstrip("/")
    key = (api_key if api_key is not None else TRAIN_SERVER_API_KEY or "").strip()
    return TrainServerConfig(base_url=base, api_key=key)


def is_train_server_enabled(config: TrainServerConfig | None = None) -> bool:
    cfg = config or resolve_train_server_config()
    return cfg.enabled


def _headers(config: TrainServerConfig) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if config.api_key:
        headers["X-API-Key"] = config.api_key
    return headers


def is_train_server_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (RequestsConnectionError, RequestsTimeout)):
        return True
    msg = str(exc).lower()
    return "failed to establish a new connection" in msg or "connection refused" in msg or "max retries exceeded" in msg


def _request(config: TrainServerConfig, method: str, path: str, **kwargs) -> requests.Response:
    if not config.base_url:
        raise RuntimeError("Train Server URL is not configured")
    url = f"{config.base_url}{path}"
    timeout = kwargs.pop("timeout", TRAIN_SERVER_TIMEOUT)
    retries = int(kwargs.pop("retries", 1))
    retry_delay = float(kwargs.pop("retry_delay", 1.5))
    last_exc: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            return requests.request(method, url, headers=_headers(config), timeout=timeout, **kwargs)
        except (RequestsConnectionError, RequestsTimeout) as exc:
            last_exc = exc
            if attempt + 1 < retries:
                logger.debug(
                    "Train Server request failed (attempt %s/%s) %s %s: %s",
                    attempt + 1,
                    retries,
                    method,
                    url,
                    exc,
                )
                time.sleep(retry_delay)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Train Server request failed")


def check_health(config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", "/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    data["base_url"] = cfg.base_url
    data["ok"] = data.get("status") == "ok"
    return data


def list_remote_models(task: str | None = None, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    params = {"task": task} if task else None
    resp = _request(cfg, "GET", "/models", params=params)
    resp.raise_for_status()
    return resp.json()


def ensure_remote_model(model_name: str, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "POST", "/models/ensure", json={"model_name": model_name})
    resp.raise_for_status()
    return resp.json()


def list_remote_tasks(config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", "/tasks")
    resp.raise_for_status()
    return resp.json()


def create_remote_job(
    *,
    project_id: int,
    base_weights: str,
    dataset_config_path: str,
    dataset_meta: Dict[str, Any],
    training_model: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    patience: int | None = None,
    run_name: str | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    train_params: Dict[str, Any] | None = None,
    device: str | None = None,
    dataset_path: str | None = None,
    config: TrainServerConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    dataset_root = Path(str(dataset_meta.get("dataset_root") or ""))
    files = {}
    data = {
        "project_id": str(project_id),
        "base_weights": base_weights,
        "dataset_config": json.dumps(dataset_meta, ensure_ascii=False),
    }
    if training_model:
        data["training_model"] = training_model
    if epochs is not None:
        data["epochs"] = str(epochs)
    if imgsz is not None:
        data["imgsz"] = str(imgsz)
    if batch is not None:
        data["batch"] = str(batch)
    if patience is not None:
        data["patience"] = str(patience)
    if run_name:
        data["run_name"] = run_name
    if optimizer:
        data["optimizer"] = optimizer
    if lr0 is not None:
        data["lr0"] = str(lr0)
    if lrf is not None:
        data["lrf"] = str(lrf)
    if train_params:
        data["train_params"] = json.dumps(train_params, ensure_ascii=False)
    if device:
        data["device"] = device

    use_shared_path = False
    if dataset_path and _is_local_train_server_url(cfg.base_url):
        use_shared_path = True
    elif dataset_path and TRAIN_SERVER_FORCE_SHARED_PATH:
        use_shared_path = True
        logger.warning("Using explicit dataset_path for remote Train Server (TRAIN_SERVER_FORCE_SHARED_PATH=1)")
    elif dataset_path:
        logger.info(
            "Ignoring dataset_path for remote Train Server (%s); uploading zip instead.",
            cfg.base_url,
        )

    if dataset_root.exists():
        from .datasets import validate_dataset_for_training

        validate_dataset_for_training(dataset_root, dataset_meta)
        explicit_root = dataset_root
        if use_shared_path and dataset_path:
            explicit = Path(str(dataset_path))
            if explicit.is_dir():
                explicit_root = explicit
        shared = _resolve_shared_dataset_path(explicit_root, train_server_url=cfg.base_url)
        if shared and _shared_path_unusable_for_train_server(shared, cfg.base_url):
            logger.warning(
                "Refusing dataset_path %s for Train Server %s; uploading zip instead.",
                shared,
                cfg.base_url,
            )
            shared = None
        if shared:
            logger.warning(
                "Using shared dataset_path %s for Train Server %s — ensure images/ exists on that host.",
                shared,
                cfg.base_url,
            )
            data["dataset_path"] = shared
        else:
            logger.info("Uploading dataset zip to Train Server (%s)", cfg.base_url)
            zip_buffer = _zip_dataset_dir(dataset_root, dataset_meta=dataset_meta)
            zip_size_mb = zip_buffer.getbuffer().nbytes / (1024 * 1024)
            logger.info("Dataset zip size: %.2f MB from %s", zip_size_mb, dataset_root)
            files["dataset_zip"] = ("dataset.zip", zip_buffer, "application/zip")
    else:
        raise FileNotFoundError(f"Dataset root not found for upload: {dataset_root}")

    resp = _request(
        cfg,
        "POST",
        "/jobs",
        data=data,
        files=files or None,
        timeout=max(TRAIN_SERVER_TIMEOUT, 600),
    )
    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"Train Server rejected job: {detail}")
    return resp.json()


def get_remote_job(job_id: str, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}", retries=4, retry_delay=2.0, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_remote_job_progress(job_id: str, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}/progress", retries=4, retry_delay=2.0, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_remote_artifacts(job_id: str, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}/artifacts")
    resp.raise_for_status()
    return resp.json()


def download_remote_artifact(
    job_id: str,
    file_name: str,
    dest: Path,
    config: TrainServerConfig | None = None,
) -> Path:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}/download", params={"file": file_name}, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest


def fetch_remote_preview(
    job_id: str,
    file_name: str,
    dest: Path,
    config: TrainServerConfig | None = None,
) -> Path:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}/preview", params={"file": file_name}, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest


def list_remote_runs(project_id: int, config: TrainServerConfig | None = None) -> List[Dict[str, Any]]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/projects/{project_id}/runs")
    resp.raise_for_status()
    return resp.json().get("runs") or []
