"""Dataset preparation helpers for on-server training (YOLO Detect)."""

from __future__ import annotations

import json
import random
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from django.conf import settings


def get_repo_root() -> Path:
    """
    Resolve repo root directory.

    NOTE:
    Label Studio `settings.BASE_DIR` points to `label_studio/core/settings`, so we move up.
    If you move `data/` somewhere else later, update this resolver or switch to BASE_DATA_DIR.
    """

    # Derive from file location to be stable across different runtime contexts
    return Path(__file__).resolve().parents[2]  # label_studio/training/datasets.py -> label-studio


def get_datasets_root() -> Path:
    """Root directory for prepared training datasets."""

    return get_repo_root() / "data" / "training" / "datasets"


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_lines_if_exists(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _find_export_dirs(root: Path) -> Tuple[Path, Path]:
    """
    Attempt to find images/labels directories in extracted export.

    Expected layout for YOLO_WITH_IMAGES is typically:
      <root>/images/...
      <root>/labels/...
    """

    images = None
    labels = None

    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "images":
            images = p
        if p.is_dir() and p.name.lower() == "labels":
            labels = p
        if images and labels:
            break

    if images is None or labels is None:
        raise FileNotFoundError("Could not locate images/labels directories in extracted YOLO export")

    return images, labels


def prepare_yolo_dataset_from_project_export(
    *,
    project_id: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    export_format: str = "YOLO_WITH_IMAGES",
) -> Dict[str, Any]:
    """
    Prepare a YOLO dataset from a Label Studio project using export conversion.

    Steps:
    - Export project using converter to YOLO_WITH_IMAGES (zip).
    - Extract zip into dataset directory.
    - Generate train/val split file lists.
    - Generate Ultralytics `data.yaml`.

    Returns:
        Dict with `dataset_root`, `data_yaml`, `images_dir`, `labels_dir`, and `classes`.
    """

    from tasks.functions import export_project

    ds_root = get_datasets_root() / f"project_{project_id}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_root.mkdir(parents=True, exist_ok=True)

    export_path = export_project(project_id, export_format, str(ds_root))
    export_file = Path(export_path)

    extracted_root = ds_root / "export"
    extracted_root.mkdir(parents=True, exist_ok=True)

    # Export is usually a zip for YOLO_WITH_IMAGES
    if export_file.suffix.lower() == ".zip":
        with zipfile.ZipFile(export_file, "r") as zf:
            zf.extractall(extracted_root)
    else:
        # Some configurations might produce a directory-like output; handle conservatively
        raise ValueError(f"Unexpected export file type: {export_file.name}")

    images_dir, labels_dir = _find_export_dirs(extracted_root)

    # Optional classes file (converter may output this)
    classes = []
    for candidate in ["classes.txt", "class_names.txt", "labels.txt"]:
        classes = _read_lines_if_exists(extracted_root / candidate)
        if classes:
            break

    # Fallback: infer number of classes from label files (max class id + 1)
    if not classes:
        max_cls = -1
        for txt in labels_dir.rglob("*.txt"):
            for line in _read_lines_if_exists(txt):
                try:
                    cls_id = int(line.split()[0])
                    max_cls = max(max_cls, cls_id)
                except Exception:
                    continue
        if max_cls >= 0:
            classes = [f"class_{i}" for i in range(max_cls + 1)]

    # List images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    image_files.sort(key=lambda p: p.name.lower())

    rnd = random.Random(int(seed))
    rnd.shuffle(image_files)
    # Ensure we always have a non-empty validation set when possible,
    # otherwise Ultralytics metrics can become NaN.
    n = len(image_files)
    cut = int(n * float(train_ratio))
    if n >= 2:
        cut = max(1, min(cut, n - 1))  # at least 1 train and 1 val
    train_imgs = image_files[:cut]
    val_imgs = image_files[cut:]

    # Use absolute paths to avoid path resolution issues on Windows servers
    train_txt = ds_root / "train.txt"
    val_txt = ds_root / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs), encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs), encoding="utf-8")

    # Ultralytics data.yaml
    names_map = ", ".join(f"{i}: '{name}'" for i, name in enumerate(classes)) if classes else ""
    data_yaml = ds_root / "data.yaml"
    _write_yaml(
        data_yaml,
        "\n".join(
            [
                f"path: {str(ds_root.resolve())}",
                "train: train.txt",
                "val: val.txt",
                f"nc: {len(classes)}" if classes else "nc: 0",
                f"names: {{{names_map}}}" if classes else "names: {}",
                "",
            ]
        ),
    )

    meta = {
        "project_id": project_id,
        "export_format": export_format,
        "dataset_root": str(ds_root),
        "export_file": str(export_file),
        "extracted_root": str(extracted_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml": str(data_yaml),
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
        "classes": classes,
        "image_count": len(image_files),
        "train_count": len(train_imgs),
        "val_count": len(val_imgs),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
    }

    (ds_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

