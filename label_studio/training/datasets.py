"""Dataset preparation helpers for on-server training."""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from django.conf import settings
from django.db.models import Prefetch
from projects.models import Project
from tasks.models import Annotation, Task


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


def _write_json(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _get_first_input_data_key(info: Dict[str, Any]) -> str | None:
    for input_tag in info.get("inputs", []):
        value = input_tag.get("value")
        if isinstance(value, str):
            normalized = value[1:] if value.startswith("$") else value
            if normalized:
                return normalized
    return None


def detect_training_interface(project: Project) -> Dict[str, Any]:
    """
    Detect the project training interface from parsed label config.

    Supported mappings:
    - Image + RectangleLabels -> detect
    - Image + Choices -> classification
    """

    parsed = project.get_parsed_config() or {}

    for control_name, info in parsed.items():
        control_type = str(info.get("type") or "")
        inputs = info.get("inputs", [])
        input_types = {str(item.get("type") or "") for item in inputs}
        data_key = _get_first_input_data_key(info)

        if control_type == "RectangleLabels" and "Image" in input_types:
            return {
                "task_type": "detect",
                "training_model": "yolo_detect",
                "control_name": control_name,
                "data_key": data_key,
                "labels": info.get("labels", []),
            }

        if control_type == "Choices" and "Image" in input_types:
            return {
                "task_type": "classification",
                "training_model": "yolo_classify",
                "control_name": control_name,
                "data_key": data_key,
                "labels": info.get("labels", []),
            }

    raise ValueError(
        "目前僅支援 Image + RectangleLabels（detect）或 Image + Choices（classification）這兩種訓練介面。",
    )


def _export_project_json(project_id: int, output_path: Path, export_format: str = "JSON_MIN") -> Path:
    from tasks.functions import export_project

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = export_project(project_id, export_format, str(output_path))
    return Path(export_path)


def _resolve_local_image_path(raw_value: str) -> Path | None:
    if not raw_value:
        return None

    direct = Path(raw_value)
    if direct.exists():
        return direct.resolve()

    if raw_value.startswith("file:///"):
        file_path = Path(raw_value.replace("file:///", "", 1))
        if file_path.exists():
            return file_path.resolve()

    parsed = urlparse(raw_value)

    if raw_value.startswith("/data/local-files/") or parsed.path.startswith("/data/local-files/"):
        relative = parse_qs(parsed.query).get("d", [None])[0]
        root = getattr(settings, "LOCAL_FILES_DOCUMENT_ROOT", None)

        if relative and root:
            target = Path(root) / unquote(relative)
            if target.exists():
                return target.resolve()

    media_prefixes = ["/media/", "/data/upload/"]
    for prefix in media_prefixes:
        if raw_value.startswith(prefix):
            relative_path = raw_value[len(prefix) :].lstrip("/\\")
            target = Path(settings.MEDIA_ROOT) / relative_path
            if target.exists():
                return target.resolve()

    return None


def _extract_classification_label(annotation_results: Iterable[Annotation], control_name: str) -> str | None:
    for annotation in annotation_results:
        if annotation.was_cancelled or not annotation.result:
            continue

        for item in annotation.result:
            if item.get("from_name") != control_name:
                continue
            if str(item.get("type") or "").lower() != "choices":
                continue

            choices = ((item.get("value") or {}).get("choices")) or []
            if choices:
                return str(choices[0])

    return None


def _copy_classification_split(records: List[Dict[str, Any]], split_root: Path) -> None:
    for record in records:
        image_path = Path(record["source_image"])
        target_dir = split_root / record["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{record['task_id']}_{image_path.name}"
        shutil.copy2(image_path, target_dir / target_name)


def _prepare_detect_dataset(
    *,
    project: Project,
    ds_root: Path,
    train_ratio: float,
    seed: int,
    export_format: str = "YOLO_WITH_IMAGES",
) -> Dict[str, Any]:
    from tasks.functions import export_project

    export_path = export_project(project.id, export_format, str(ds_root))
    export_file = Path(export_path)
    records_json = _export_project_json(project.id, ds_root / "records.json", export_format="JSON_MIN")

    extracted_root = ds_root / "export"
    extracted_root.mkdir(parents=True, exist_ok=True)

    if export_file.suffix.lower() == ".zip":
        with zipfile.ZipFile(export_file, "r") as zf:
            zf.extractall(extracted_root)
    else:
        raise ValueError(f"Unexpected export file type: {export_file.name}")

    images_dir, labels_dir = _find_export_dirs(extracted_root)

    classes = []
    for candidate in ["classes.txt", "class_names.txt", "labels.txt"]:
        classes = _read_lines_if_exists(extracted_root / candidate)
        if classes:
            break

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

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    image_files.sort(key=lambda p: p.name.lower())

    rnd = random.Random(int(seed))
    rnd.shuffle(image_files)
    n = len(image_files)
    cut = int(n * float(train_ratio))
    if n >= 2:
        cut = max(1, min(cut, n - 1))
    train_imgs = image_files[:cut]
    val_imgs = image_files[cut:]

    train_txt = ds_root / "train.txt"
    val_txt = ds_root / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs), encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs), encoding="utf-8")

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

    dataset_config = ds_root / "dataset_config.json"
    dataset_manifest = {
        "task_type": "detect",
        "training_model": "yolo_detect",
        "project_id": project.id,
        "dataset_root": str(ds_root),
        "data_yaml": str(data_yaml),
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
        "records_json": str(records_json),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "classes": classes,
        "image_count": len(image_files),
        "train_count": len(train_imgs),
        "val_count": len(val_imgs),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "export_format": export_format,
    }
    _write_json(dataset_config, dataset_manifest)
    return dataset_manifest | {"dataset_config": str(dataset_config), "export_file": str(export_file), "extracted_root": str(extracted_root)}


def _prepare_classification_dataset(
    *,
    project: Project,
    ds_root: Path,
    train_ratio: float,
    seed: int,
    export_format: str = "JSON_MIN",
) -> Dict[str, Any]:
    spec = detect_training_interface(project)
    control_name = spec["control_name"]
    data_key = spec["data_key"]

    if not data_key:
        raise ValueError("Classification 專案找不到對應的 Image 欄位。")

    export_file = _export_project_json(project.id, ds_root / "records.json", export_format=export_format)

    tasks = (
        Task.objects.filter(project_id=project.id)
        .prefetch_related(
            Prefetch(
                "annotations",
                queryset=Annotation.objects.filter(was_cancelled=False).order_by("-updated_at"),
            )
        )
        .order_by("id")
    )

    items: List[Dict[str, Any]] = []
    classes = list(spec.get("labels") or [])

    for task in tasks:
        image_value = task.data.get(data_key) if isinstance(task.data, dict) else None
        if not isinstance(image_value, str):
            continue

        image_path = _resolve_local_image_path(image_value)
        if image_path is None or not image_path.exists():
            continue

        label = _extract_classification_label(task.annotations.all(), control_name)
        if not label:
            continue

        if label not in classes:
            classes.append(label)

        items.append(
            {
                "task_id": task.id,
                "label": label,
                "source_image": str(image_path),
            }
        )

    rnd = random.Random(int(seed))
    rnd.shuffle(items)

    n = len(items)
    cut = int(n * float(train_ratio))
    if n >= 2:
        cut = max(1, min(cut, n - 1))
    train_items = items[:cut]
    val_items = items[cut:]

    train_root = ds_root / "train"
    val_root = ds_root / "val"
    _copy_classification_split(train_items, train_root)
    _copy_classification_split(val_items, val_root)

    items_json = ds_root / "dataset_items.json"
    items_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_config = ds_root / "dataset_config.json"
    dataset_manifest = {
        "task_type": "classification",
        "training_model": "yolo_classify",
        "project_id": project.id,
        "dataset_root": str(ds_root),
        "train_dir": str(train_root),
        "val_dir": str(val_root),
        "records_json": str(export_file),
        "items_json": str(items_json),
        "classes": classes,
        "image_count": len(items),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "export_format": export_format,
        "control_name": control_name,
        "data_key": data_key,
    }
    _write_json(dataset_config, dataset_manifest)
    return dataset_manifest | {"dataset_config": str(dataset_config)}


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

    project = Project.objects.get(pk=project_id)
    ds_root = get_datasets_root() / f"project_{project_id}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_root.mkdir(parents=True, exist_ok=True)

    meta = _prepare_detect_dataset(
        project=project,
        ds_root=ds_root,
        train_ratio=train_ratio,
        seed=seed,
        export_format=export_format,
    )
    (ds_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def prepare_training_dataset_for_project(
    *,
    project_id: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Prepare a training dataset based on the project's labeling interface.

    Output is always a JSON dataset manifest (`dataset_config.json`).
    """

    project = Project.objects.get(pk=project_id)
    spec = detect_training_interface(project)
    ds_root = get_datasets_root() / f"project_{project_id}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_root.mkdir(parents=True, exist_ok=True)

    if spec["task_type"] == "detect":
        meta = _prepare_detect_dataset(
            project=project,
            ds_root=ds_root,
            train_ratio=train_ratio,
            seed=seed,
            export_format="YOLO_WITH_IMAGES",
        )
    elif spec["task_type"] == "classification":
        meta = _prepare_classification_dataset(
            project=project,
            ds_root=ds_root,
            train_ratio=train_ratio,
            seed=seed,
            export_format="JSON_MIN",
        )
    else:
        raise ValueError(f"Unsupported task_type: {spec['task_type']}")

    meta = {**spec, **meta}
    (ds_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

