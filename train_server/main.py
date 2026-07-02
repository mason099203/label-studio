"""
Label Studio Train Server — 獨立 YOLO 訓練服務。

與 Label Studio 主應用分離部署，透過 REST API 接收訓練任務。
參考 Ultralytics 文件：https://docs.ultralytics.com/zh
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .job_manager import JobManager
from .trainer import ensure_yolo_weights
from .yolo_catalog import (
    YOLO_PRESET_MODELS,
    YOLO_TASK_DEFAULTS,
    get_preset_models_for_task,
    get_task_defaults,
    is_weight_compatible_with_task,
    validate_yolo_training_request,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_server")

TRAIN_SERVER_API_KEY = os.environ.get("TRAIN_SERVER_API_KEY", "").strip()
TRAIN_SERVER_OUTPUT_ROOT = Path(
    os.environ.get("TRAIN_SERVER_OUTPUT_ROOT", "/data/training/models/trained")
).resolve()
TRAIN_SERVER_MODELS_DIR = Path(
    os.environ.get("TRAIN_SERVER_MODELS_DIR", "/data/training/models/original")
).resolve()
TRAIN_SERVER_MAX_WORKERS = int(os.environ.get("TRAIN_SERVER_MAX_WORKERS", "1"))

app = FastAPI(title="Label Studio Train Server", version="1.0.0")
job_manager = JobManager(output_root=TRAIN_SERVER_OUTPUT_ROOT, max_workers=TRAIN_SERVER_MAX_WORKERS)


def _verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not TRAIN_SERVER_API_KEY:
        return
    if x_api_key != TRAIN_SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "train-server"}


@app.get("/tasks")
def list_tasks(_: None = Depends(_verify_api_key)) -> Dict[str, Any]:
    return {
        "tasks": [
            {"task": task, "defaults": defaults, "preset_count": len(YOLO_PRESET_MODELS.get(task, []))}
            for task, defaults in YOLO_TASK_DEFAULTS.items()
        ]
    }


@app.get("/models")
def list_models(
    task: Optional[str] = Query(default=None),
    _: None = Depends(_verify_api_key),
) -> Dict[str, Any]:
    TRAIN_SERVER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_files = {p.name: p for p in TRAIN_SERVER_MODELS_DIR.glob("*.pt")}

    preset_models: List[Dict[str, Any]] = []
    tasks = [task] if task else list(YOLO_PRESET_MODELS.keys())
    for t in tasks:
        for item in get_preset_models_for_task(t):
            name = item["name"]
            local = local_files.get(name)
            preset_models.append(
                {
                    **item,
                    "available": local is not None,
                    "path": str(local) if local else name,
                    "size_bytes": local.stat().st_size if local else None,
                }
            )

    cached = []
    for p in sorted(TRAIN_SERVER_MODELS_DIR.glob("*.pt"), key=lambda x: x.name.lower()):
        if any(m["name"] == p.name for m in preset_models):
            continue
        if task and not is_weight_compatible_with_task(p.name, task):
            continue
        cached.append(
            {
                "id": p.name,
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "available": True,
                "source": "local",
            }
        )

    return {
        "models": preset_models + cached,
        "root": str(TRAIN_SERVER_MODELS_DIR),
        "output_root": str(TRAIN_SERVER_OUTPUT_ROOT),
    }


@app.post("/models/ensure")
def ensure_model(
    payload: Dict[str, Any],
    _: None = Depends(_verify_api_key),
) -> Dict[str, Any]:
    model_name = str(payload.get("model_name") or payload.get("name") or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"

    try:
        path = ensure_yolo_weights(model_name, TRAIN_SERVER_MODELS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model_name": model_name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "downloaded": True,
    }


@app.post("/jobs")
async def create_job(
    project_id: int = Form(...),
    base_weights: str = Form(...),
    dataset_config: str = Form(...),
    training_model: Optional[str] = Form(default=None),
    epochs: Optional[int] = Form(default=None),
    imgsz: Optional[int] = Form(default=None),
    batch: Optional[int] = Form(default=None),
    patience: Optional[int] = Form(default=None),
    run_name: Optional[str] = Form(default=None),
    optimizer: Optional[str] = Form(default=None),
    lr0: Optional[float] = Form(default=None),
    lrf: Optional[float] = Form(default=None),
    train_params: Optional[str] = Form(default=None),
    device: Optional[str] = Form(default=None),
    dataset_path: Optional[str] = Form(default=None),
    dataset_zip: UploadFile | None = File(default=None),
    _: None = Depends(_verify_api_key),
) -> Dict[str, Any]:
    try:
        dataset_meta = json.loads(dataset_config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset_config JSON: {exc}") from exc

    if training_model:
        dataset_meta["training_model"] = training_model

    compat_err = validate_yolo_training_request(
        base_weights=base_weights,
        dataset_meta=dataset_meta,
        training_model=training_model,
    )
    if compat_err:
        raise HTTPException(status_code=400, detail=compat_err)

    model_name = Path(base_weights).name if not Path(base_weights).is_absolute() else Path(base_weights).name
    try:
        weights_path = ensure_yolo_weights(model_name, TRAIN_SERVER_MODELS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot resolve base_weights: {exc}") from exc

    param_overrides: Dict[str, Any] = {}
    if training_model:
        param_overrides["training_model"] = training_model
    if epochs is not None:
        param_overrides["epochs"] = epochs
    if imgsz is not None:
        param_overrides["imgsz"] = imgsz
    if batch is not None:
        param_overrides["batch"] = batch
    if patience is not None:
        param_overrides["patience"] = patience
    if run_name:
        param_overrides["run_name"] = run_name.strip()
    if optimizer:
        param_overrides["optimizer"] = optimizer.strip()
    if lr0 is not None:
        param_overrides["lr0"] = lr0
    if lrf is not None:
        param_overrides["lrf"] = lrf
    if train_params:
        try:
            extra = json.loads(train_params)
            if isinstance(extra, dict):
                param_overrides.update({k: v for k, v in extra.items() if v is not None})
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid train_params JSON: {exc}") from exc
    if device:
        param_overrides["device"] = device

    zip_path: Path | None = None
    if dataset_zip and dataset_zip.filename:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.close()
        zip_path = Path(tmp.name)
        with open(zip_path, "wb") as out:
            shutil.copyfileobj(dataset_zip.file, out)

    job_id = job_manager.create_job(
        project_id=project_id,
        base_weights=str(weights_path),
        dataset_zip_path=zip_path,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        param_overrides=param_overrides or None,
    )

    return {"job_id": job_id, "status": "queued", "training_model": dataset_meta.get("training_model")}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: None = Depends(_verify_api_key)) -> Dict[str, Any]:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    meta = job.get("meta") or {}
    job_params = job.get("params") or {}
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "params": job_params,
        "meta": {
            **meta,
            "params": job_params or meta.get("params") or {},
            "status": job.get("status"),
            "message": job.get("message"),
            "error": job.get("error"),
            "metrics": meta.get("metrics"),
            "best_path": meta.get("best_path"),
            "last_path": meta.get("last_path"),
            "artifacts_dir": meta.get("artifacts_dir"),
        },
        "created_at": job.get("created_at"),
        "error": job.get("error"),
    }


@app.get("/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str, _: None = Depends(_verify_api_key)) -> Dict[str, Any]:
    artifacts_dir = job_manager.get_artifacts_dir(job_id)
    if not artifacts_dir:
        return {"artifacts": []}
    artifacts = []
    for p in sorted(artifacts_dir.glob("*"), key=lambda x: x.name.lower()):
        if p.is_file():
            artifacts.append({"name": p.name, "size_bytes": p.stat().st_size})
    return {"artifacts": artifacts, "root": str(artifacts_dir)}


@app.get("/jobs/{job_id}/download")
def download_artifact(
    job_id: str,
    file: str = Query(...),
    _: None = Depends(_verify_api_key),
):
    artifacts_dir = job_manager.get_artifacts_dir(job_id)
    if not artifacts_dir:
        raise HTTPException(status_code=404, detail="Artifacts not found")
    target = artifacts_dir / file
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(target), filename=target.name, media_type="application/octet-stream")


def _resolve_preview_path(job_id: str, file: str) -> Path:
    run_dir = job_manager.get_run_dir(job_id)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Run not found")
    safe_name = Path(file).name
    live = run_dir / "artifacts" / "live" / safe_name
    if live.exists():
        return live
    train_dir = run_dir / "train"
    direct = train_dir / safe_name
    if direct.exists():
        return direct
    raise HTTPException(status_code=404, detail="Preview not found")


@app.get("/jobs/{job_id}/progress")
def get_job_progress(job_id: str, _: None = Depends(_verify_api_key)) -> Dict[str, Any]:
    progress = job_manager.get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress


@app.get("/jobs/{job_id}/preview")
def preview_artifact(
    job_id: str,
    file: str = Query(...),
    _: None = Depends(_verify_api_key),
):
    target = _resolve_preview_path(job_id, file)
    media = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=str(target), filename=target.name, media_type=media)


@app.get("/projects/{project_id}/runs")
def list_project_runs(project_id: int, _: None = Depends(_verify_api_key)) -> Dict[str, Any]:
    return {"runs": job_manager.list_runs(project_id)}


def main() -> None:
    import uvicorn

    host = os.environ.get("TRAIN_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("TRAIN_SERVER_PORT", "8011"))
    uvicorn.run("train_server.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
