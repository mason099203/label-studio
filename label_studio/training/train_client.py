"""HTTP client for the standalone Train Server."""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TRAIN_SERVER_URL = os.environ.get("TRAIN_SERVER_URL", "").strip().rstrip("/")
TRAIN_SERVER_API_KEY = os.environ.get("TRAIN_SERVER_API_KEY", "").strip()
TRAIN_SERVER_TIMEOUT = float(os.environ.get("TRAIN_SERVER_TIMEOUT", "120"))


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


def _request(config: TrainServerConfig, method: str, path: str, **kwargs) -> requests.Response:
    if not config.base_url:
        raise RuntimeError("Train Server URL is not configured")
    url = f"{config.base_url}{path}"
    timeout = kwargs.pop("timeout", TRAIN_SERVER_TIMEOUT)
    return requests.request(method, url, headers=_headers(config), timeout=timeout, **kwargs)


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


def _zip_dataset_dir(dataset_root: Path) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in dataset_root.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(dataset_root).as_posix()
                zf.write(path, arcname)
    buffer.seek(0)
    return buffer


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
    if dataset_path:
        data["dataset_path"] = dataset_path
    elif dataset_root.exists():
        zip_buffer = _zip_dataset_dir(dataset_root)
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
    resp = _request(cfg, "GET", f"/jobs/{job_id}")
    resp.raise_for_status()
    return resp.json()


def get_remote_job_progress(job_id: str, config: TrainServerConfig | None = None) -> Dict[str, Any]:
    cfg = config or resolve_train_server_config()
    resp = _request(cfg, "GET", f"/jobs/{job_id}/progress")
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
