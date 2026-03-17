"""
Utilities for exporting training artifacts into a Triton model repository.

The deployment flow converts a trained YOLO checkpoint into a Triton-loadable
TorchScript model, writes the Triton config files, and stores a small metadata
file so the frontend Playground can discover project-specific deployments.
"""

from __future__ import annotations

import logging
import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Optional

import shutil

logger = logging.getLogger(__name__)


def get_triton_model_repository_root() -> Path:
    """
    Root path for Triton model repository.
    Uses TRITON_MODEL_REPOSITORY env, else default under repo data/triton_models.
    """
    env_path = os.environ.get("TRITON_MODEL_REPOSITORY")
    if env_path:
        return Path(env_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "triton_models"


def get_triton_server_url() -> str:
    """
    Base HTTP URL for Triton.
    """
    return os.environ.get("TRITON_SERVER_URL", "http://localhost:8000").rstrip("/")


def sanitize_triton_model_name(raw_name: str, fallback: str = "triton_model") -> str:
    """
    Convert an arbitrary string into a Triton-safe model name.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", (raw_name or "").strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or fallback


def _build_triton_pbtxt(model_name: str, imgsz: int) -> str:
    """
    Build the Triton config content used by the current YOLO deployment flow.
    """
    return f'''name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 1

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, {imgsz}, {imgsz}]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: {[-1, -1] if "yolo" in model_name.lower() else "[-1]"}
  }}
]
'''


def export_torchscript_pt_to_triton(
    best_pt_path: str | Path,
    model_name: str,
    project_id: Optional[int] = None,
    run_id: Optional[str] = None,
    triton_repo_root: Optional[str | Path] = None,
    imgsz: int = 224,
    deployed_by_user_id: Optional[int] = None,
    deployed_by_username: Optional[str] = None,
) -> dict:
    """
    Export a general TorchScript model (.pt) into Triton's libtorch layout.
    """
    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        return {"error": f"TorchScript file not found: {best_pt_path}"}

    repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
    repo_root.mkdir(parents=True, exist_ok=True)

    model_name = sanitize_triton_model_name(model_name)
    model_dir = repo_root / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    model_pt_path = version_dir / "model.pt"
    config_path = model_dir / "config.pbtxt"
    metadata_path = model_dir / "deployment_meta.json"

    try:
        shutil.copy2(best_pt_path, model_pt_path)
    except Exception as exc:
        return {"error": f"Failed to copy model: {exc}"}

    config_content = _build_triton_pbtxt(model_name=model_name, imgsz=imgsz)
    config_path.write_text(config_content, encoding="utf-8")

    metadata = {
        "model_name": model_name,
        "project_id": project_id,
        "run_id": run_id,
        "imgsz": imgsz,
        "model_pt_path": str(model_pt_path.resolve()),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "torchscript_cnn",
        "deployed_by_user_id": deployed_by_user_id,
        "deployed_by_username": deployed_by_username,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "model_name": model_name,
        "repo_path": str(repo_root),
        "model_dir": str(model_dir),
        "metadata_path": str(metadata_path),
        "infer_url": f"{get_triton_server_url()}/v2/models/{model_name}/infer",
    }


def export_yolo_pt_to_triton(
    best_pt_path: str | Path,
    model_name: str,
    project_id: Optional[int] = None,
    run_id: Optional[str] = None,
    triton_repo_root: Optional[str | Path] = None,
    imgsz: int = 640,
    deployed_by_user_id: Optional[int] = None,
    deployed_by_username: Optional[str] = None,
) -> dict:
    """
    Export a trained YOLO checkpoint into Triton's libtorch model layout.

    Args:
        best_pt_path: Path to best.pt (e.g. from training artifacts).
        model_name: Triton model name (used as directory name under repo).
        project_id: Optional project id used for deployment metadata.
        run_id: Optional training run id used for deployment metadata.
        triton_repo_root: Override repo root; default from get_triton_model_repository_root().
        imgsz: Input size (H, W) for config.pbtxt; default 640.

    Returns:
        dict with keys describing repository paths, metadata, and infer URL.
    """
    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        return {"error": f"best.pt not found: {best_pt_path}"}

    repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
    repo_root.mkdir(parents=True, exist_ok=True)

    model_name = sanitize_triton_model_name(model_name)
    model_dir = repo_root / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    model_pt_path = version_dir / "model.pt"
    config_path = model_dir / "config.pbtxt"
    legacy_bptxt_path = model_dir / f"{model_name}.bptxt"
    metadata_path = model_dir / "deployment_meta.json"

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return {"error": f"Ultralytics not installed: {exc}"}

    try:
        model = YOLO(str(best_pt_path))
        exported = model.export(format="torchscript", imgsz=imgsz)
        src_torchscript = Path(exported).resolve() if exported else (best_pt_path.parent / (best_pt_path.stem + ".torchscript"))
        if not src_torchscript.exists():
            return {"error": f"TorchScript export did not produce {src_torchscript}"}
        shutil.copy2(src_torchscript, model_pt_path)
    except Exception as exc:
        logger.exception("Failed to export trained model to TorchScript for Triton")
        return {"error": str(exc)}

    config_content = _build_triton_pbtxt(model_name=model_name, imgsz=imgsz)
    config_path.write_text(config_content, encoding="utf-8")
    legacy_bptxt_path.write_text(config_content, encoding="utf-8")

    metadata = {
        "model_name": model_name,
        "project_id": project_id,
        "run_id": run_id,
        "imgsz": imgsz,
        "source_pt_path": str(best_pt_path.resolve()),
        "exported_torchscript_path": str(src_torchscript.resolve()),
        "model_pt_path": str(model_pt_path.resolve()),
        "config_path": str(config_path.resolve()),
        "legacy_bptxt_path": str(legacy_bptxt_path.resolve()),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "deployed_by_user_id": deployed_by_user_id,
        "deployed_by_username": deployed_by_username,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "model_name": model_name,
        "repo_path": str(repo_root),
        "model_dir": str(model_dir),
        "version_dir": str(version_dir),
        "model_pt_path": str(model_pt_path),
        "config_path": str(config_path),
        "legacy_bptxt_path": str(legacy_bptxt_path),
        "metadata_path": str(metadata_path),
        "infer_url": f"{get_triton_server_url()}/v2/models/{model_name}/infer",
    }


def list_triton_model_deployments(
    project_ids: Optional[Iterable[int]] = None,
    deployed_by_user_ids: Optional[Iterable[int]] = None,
) -> list[dict]:
    """
    List Triton deployments discovered from repository metadata files.
    """
    repo_root = get_triton_model_repository_root()
    if not repo_root.exists():
        return []

    allowed_projects = {int(project_id) for project_id in project_ids or []}
    allowed_users = {int(user_id) for user_id in deployed_by_user_ids or []}
    items = []

    for metadata_path in repo_root.glob("*/deployment_meta.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Invalid Triton deployment metadata: %s", metadata_path)
            continue

        project_id = metadata.get("project_id")
        if allowed_projects and project_id not in allowed_projects:
            continue
        deployed_by_user_id = metadata.get("deployed_by_user_id")
        if allowed_users and deployed_by_user_id not in allowed_users:
            continue

        model_name = metadata.get("model_name")
        model_dir = metadata_path.parent
        version_dir = model_dir / "1"
        items.append(
            {
                **metadata,
                "model_dir": str(model_dir),
                "version_dir": str(version_dir),
                "exists": version_dir.exists(),
                "infer_url": f"{get_triton_server_url()}/v2/models/{model_name}/infer",
            }
        )

    return sorted(items, key=lambda item: item.get("deployed_at") or "", reverse=True)
