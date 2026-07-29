"""Helpers to locate YOLO dataset layout inside extracted archives or directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_zip_entry(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def is_classification_dataset(dataset_meta: Dict[str, Any] | None, dataset_root: Path) -> bool:
    meta = dataset_meta or {}
    training_model = str(meta.get("training_model") or "")
    task_type = str(meta.get("task_type") or "")
    if training_model == "yolo_classify" or task_type == "classification":
        return True
    root = Path(dataset_root)
    return (root / "train").is_dir() and (root / "val").is_dir() and not (root / "images").is_dir()


def list_zip_image_entries(names: Iterable[str]) -> List[str]:
    """Return zip member paths that refer to files under an images/ directory."""
    entries: List[str] = []
    for raw in names:
        norm = normalize_zip_entry(raw)
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


def list_zip_classification_image_entries(names: Iterable[str]) -> List[str]:
    """Return zip member paths for YOLO classify layout (train/<class>/*, val/<class>/*)."""
    entries: List[str] = []
    for raw in names:
        norm = normalize_zip_entry(raw)
        if not norm or norm.endswith("/"):
            continue
        lower = norm.lower()
        if lower.startswith("train/") or lower.startswith("val/"):
            if Path(norm).suffix.lower() in _IMAGE_EXTS:
                entries.append(raw)
    return entries


def list_zip_dataset_image_entries(
    names: Iterable[str],
    *,
    classification: bool,
) -> List[str]:
    if classification:
        return list_zip_classification_image_entries(names)
    return list_zip_image_entries(names)


def unwrap_single_dataset_root(root: Path) -> Path:
    """If *root* contains a single child directory, descend into it (zip extract layout)."""
    children = [p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(children) == 1 and not any(p.is_file() for p in root.iterdir() if p.name != "__MACOSX"):
        return children[0]
    return root


def find_images_labels_dirs(root: Path) -> Tuple[Path, Path]:
    """Locate images/ and labels/ under *root* (searches recursively)."""
    images = None
    labels = None
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if p.name.lower() == "images" and images is None:
            images = p
        elif p.name.lower() == "labels" and labels is None:
            labels = p
        if images and labels:
            break
    if images is None or labels is None:
        raise FileNotFoundError(
            f"Could not locate images/ and labels/ under {root}. "
            "Regenerate the training dataset in Label Studio and resubmit."
        )
    return images, labels


def find_classification_dirs(root: Path) -> Tuple[Path, Path]:
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"YOLO classification dataset must contain train/ and val/ under {root}. "
            "Regenerate the training dataset in Label Studio and resubmit."
        )
    return train_dir, val_dir


def validate_dataset_layout(root: Path, dataset_meta: Dict[str, Any] | None = None) -> None:
    """Ensure extracted or shared dataset has image files for detect or classify layout."""
    if is_classification_dataset(dataset_meta, root):
        train_dir, val_dir = find_classification_dirs(root)
        train_count = count_image_files(train_dir)
        val_count = count_image_files(val_dir)
        if train_count == 0 or val_count == 0:
            raise FileNotFoundError(
                f"Classification dataset under {root} is missing images "
                f"(train={train_count}, val={val_count}). "
                "Regenerate the training dataset in Label Studio and resubmit."
            )
        return

    images_dir, _labels_dir = find_images_labels_dirs(root)
    if count_image_files(images_dir) == 0:
        raise FileNotFoundError(
            f"Detection dataset has no image files under {images_dir}. "
            "Regenerate the training dataset in Label Studio and resubmit."
        )


def count_image_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
