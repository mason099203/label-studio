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


def _control_has_image_input(info: Dict[str, Any]) -> bool:
    input_types = {str(item.get("type") or "") for item in info.get("inputs", [])}
    return "Image" in input_types


def _find_control_by_type(parsed: Dict[str, Any], control_type: str) -> Tuple[str | None, Dict[str, Any] | None]:
    for control_name, info in parsed.items():
        if str(info.get("type") or "") == control_type:
            return control_name, info
    return None, None


def _project_has_control_type(parsed: Dict[str, Any], control_type: str) -> bool:
    return any(str(info.get("type") or "") == control_type for info in parsed.values())


def _rectangle_labels_use_obb(project: Project) -> bool:
    config = project.label_config or ""
    if not config.strip():
        return False
    try:
        from label_studio.core.label_config import parse_config_to_xml

        root = parse_config_to_xml(config)
        if root is None:
            return False
        for elem in root.iter():
            if elem.tag != "RectangleLabels":
                continue
            val = (elem.get("model_obb") or "").strip().lower()
            if val in {"true", "1", "yes"}:
                return True
    except Exception:
        pass
    return False


def _get_pose_kpt_shape(parsed: Dict[str, Any]) -> Tuple[int, int]:
    _, kp_info = _find_control_by_type(parsed, "KeyPointLabels")
    if not kp_info:
        return 0, 3

    labels = list(kp_info.get("labels") or [])
    labels_attrs = kp_info.get("labels_attrs") or {}
    indices: List[int] = []
    for attrs in labels_attrs.values():
        try:
            indices.append(int(attrs.get("model_index")))
        except (TypeError, ValueError):
            continue

    if indices:
        return max(indices) + 1, 3
    return len(labels), 3


def _default_export_format_for_task(task_type: str) -> str:
    if task_type == "obb":
        return "YOLO_OBB_WITH_IMAGES"
    if task_type == "classification":
        return "JSON_MIN"
    if task_type in {"semantic", "semantic_segmentation"}:
        return "JSON_MIN"
    return "YOLO_WITH_IMAGES"


def _build_training_spec(
    *,
    task_type: str,
    training_model: str,
    control_name: str,
    data_key: str | None,
    labels: List[str],
    keypoint_control_name: str | None = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "task_type": task_type,
        "training_model": training_model,
        "control_name": control_name,
        "data_key": data_key,
        "labels": labels,
    }
    if keypoint_control_name:
        spec["keypoint_control_name"] = keypoint_control_name
    return spec


def detect_training_interface(project: Project) -> Dict[str, Any]:
    """
    Detect the project training interface from parsed label config.

    Supported mappings:
    - Image + Choices -> classification
    - Image + RectangleLabels + KeyPointLabels -> pose
    - Image + RectangleLabels (model_obb="true") -> obb
    - Image + BrushLabels / MaskLabels / BitmaskLabels -> semantic segmentation
    - Image + RectangleLabels -> detect
    - Image + PolygonLabels -> instance segmentation
    """

    parsed = project.get_parsed_config() or {}
    if not parsed:
        raise ValueError("專案標註設定（label_config）為空或無法解析。")

    semantic_brush_types = {"BrushLabels", "MaskLabels", "BitmaskLabels"}
    for control_name, info in parsed.items():
        control_type = str(info.get("type") or "")
        if control_type == "Choices" and _control_has_image_input(info):
            return _build_training_spec(
                task_type="classification",
                training_model="yolo_classify",
                control_name=control_name,
                data_key=_get_first_input_data_key(info),
                labels=list(info.get("labels") or []),
            )

    if _project_has_control_type(parsed, "KeyPointLabels"):
        if not _project_has_control_type(parsed, "RectangleLabels"):
            raise ValueError(
                "姿態估計（pose）需要同時設定 RectangleLabels 與 KeyPointLabels，"
                "且 KeyPoint 需關聯到 Rectangle 父區域。",
            )
        rect_name, rect_info = _find_control_by_type(parsed, "RectangleLabels")
        kp_name, _ = _find_control_by_type(parsed, "KeyPointLabels")
        return _build_training_spec(
            task_type="pose",
            training_model="yolo_pose",
            control_name=rect_name or "",
            keypoint_control_name=kp_name,
            data_key=_get_first_input_data_key(rect_info or {}),
            labels=list((rect_info or {}).get("labels") or []),
        )

    if _project_has_control_type(parsed, "RectangleLabels") and _rectangle_labels_use_obb(project):
        rect_name, rect_info = _find_control_by_type(parsed, "RectangleLabels")
        return _build_training_spec(
            task_type="obb",
            training_model="yolo_obb",
            control_name=rect_name or "",
            data_key=_get_first_input_data_key(rect_info or {}),
            labels=list((rect_info or {}).get("labels") or []),
        )

    for control_name, info in parsed.items():
        control_type = str(info.get("type") or "")
        if control_type in semantic_brush_types and _control_has_image_input(info):
            return _build_training_spec(
                task_type="semantic_segmentation",
                training_model="yolo_semantic",
                control_name=control_name,
                data_key=_get_first_input_data_key(info),
                labels=list(info.get("labels") or []),
            )

    for control_name, info in parsed.items():
        control_type = str(info.get("type") or "")
        if control_type == "RectangleLabels" and _control_has_image_input(info):
            return _build_training_spec(
                task_type="detect",
                training_model="yolo_detect",
                control_name=control_name,
                data_key=_get_first_input_data_key(info),
                labels=list(info.get("labels") or []),
            )

    for control_name, info in parsed.items():
        control_type = str(info.get("type") or "")
        if control_type == "PolygonLabels" and _control_has_image_input(info):
            return _build_training_spec(
                task_type="segmentation",
                training_model="yolo_segment",
                control_name=control_name,
                data_key=_get_first_input_data_key(info),
                labels=list(info.get("labels") or []),
            )

    raise ValueError(
        "目前支援的訓練介面：Image + Choices（classification）、"
        "Image + RectangleLabels（detect/obb）、Image + PolygonLabels（instance segmentation）、"
        "Image + BrushLabels/MaskLabels（semantic segmentation）、"
        "Image + RectangleLabels + KeyPointLabels（pose）。",
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
            # Try direct relative to MEDIA_ROOT
            target = Path(settings.MEDIA_ROOT) / relative_path
            if target.exists():
                return target.resolve()
            
            # Try with 'upload' prefix if it was /data/upload/
            if prefix == "/data/upload/":
                target_upload = Path(settings.MEDIA_ROOT) / "upload" / relative_path
                if target_upload.exists():
                    return target_upload.resolve()

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


def _prepare_yolo_zip_dataset(
    *,
    project: Project,
    ds_root: Path,
    train_ratio: float,
    seed: int,
    task_type: str,
    training_model: str,
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
    yaml_lines = [
        f"path: {str(ds_root.resolve())}",
        "train: train.txt",
        "val: val.txt",
        f"nc: {len(classes)}" if classes else "nc: 0",
        f"names: {{{names_map}}}" if classes else "names: {}",
    ]
    if task_type == "pose":
        kpt_count, kpt_dims = _get_pose_kpt_shape(project.get_parsed_config() or {})
        if kpt_count > 0:
            yaml_lines.append(f"kpt_shape: [{kpt_count}, {kpt_dims}]")

    data_yaml = ds_root / "data.yaml"
    _write_yaml(data_yaml, "\n".join(yaml_lines + [""]))

    dataset_config = ds_root / "dataset_config.json"
    dataset_manifest = {
        "task_type": task_type,
        "training_model": training_model,
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
    if task_type == "pose":
        kpt_count, kpt_dims = _get_pose_kpt_shape(project.get_parsed_config() or {})
        dataset_manifest["kpt_shape"] = [kpt_count, kpt_dims]

    _write_json(dataset_config, dataset_manifest)
    return dataset_manifest | {
        "dataset_config": str(dataset_config),
        "export_file": str(export_file),
        "extracted_root": str(extracted_root),
    }


def _prepare_detect_dataset(
    *,
    project: Project,
    ds_root: Path,
    train_ratio: float,
    seed: int,
    export_format: str = "YOLO_WITH_IMAGES",
) -> Dict[str, Any]:
    return _prepare_yolo_zip_dataset(
        project=project,
        ds_root=ds_root,
        train_ratio=train_ratio,
        seed=seed,
        task_type="detect",
        training_model="yolo_detect",
        export_format=export_format,
    )


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
        if image_value is None and isinstance(task.data, dict):
            image_value = task.data.get("$undefined$") or task.data.get("image")
            if image_value is None and len(task.data) == 1:
                image_value = list(task.data.values())[0]

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


def _prepare_semantic_segmentation_dataset(
    *,
    project: Project,
    ds_root: Path,
    train_ratio: float,
    seed: int,
    export_format: str = "JSON_MIN",
) -> Dict[str, Any]:
    """Build YOLO semantic segmentation dataset (images + single-channel mask PNGs)."""
    from PIL import Image

    from .brush_utils import build_semantic_class_mask

    spec = detect_training_interface(project)
    control_name = spec["control_name"]
    data_key = spec["data_key"]
    if not data_key:
        raise ValueError("Semantic segmentation 專案找不到對應的 Image 欄位。")

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

    class_names = list(spec.get("labels") or [])
    items: List[Dict[str, Any]] = []

    for task in tasks:
        image_value = task.data.get(data_key) if isinstance(task.data, dict) else None
        if image_value is None and isinstance(task.data, dict):
            image_value = task.data.get("$undefined$") or task.data.get("image")
            if image_value is None and len(task.data) == 1:
                image_value = list(task.data.values())[0]

        if not isinstance(image_value, str):
            continue

        image_path = _resolve_local_image_path(image_value)
        if image_path is None or not image_path.exists():
            continue

        brush_items: List[Dict[str, Any]] = []
        for annotation in task.annotations.all():
            if annotation.was_cancelled or not annotation.result:
                continue
            for result_item in annotation.result:
                if result_item.get("from_name") != control_name:
                    continue
                if str(result_item.get("type") or "").lower() in {"brushlabels", "bitmasklabels", "masklabels"}:
                    brush_items.append(result_item)

        if not brush_items:
            continue

        width = int(brush_items[0].get("original_width") or (brush_items[0].get("value") or {}).get("original_width") or 0)
        height = int(brush_items[0].get("original_height") or (brush_items[0].get("value") or {}).get("original_height") or 0)
        if width <= 0 or height <= 0:
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                continue

        mask = build_semantic_class_mask(
            brush_items,
            class_names=class_names,
            width=width,
            height=height,
        )
        if not mask.any():
            continue

        items.append(
            {
                "task_id": task.id,
                "source_image": str(image_path),
                "width": width,
                "height": height,
                "_mask_array": mask,
            }
        )

    if not items:
        raise ValueError("找不到可用的 Brush/Mask 語意分割標註與本機影像。")

    rnd = random.Random(int(seed))
    rnd.shuffle(items)

    n = len(items)
    cut = int(n * float(train_ratio))
    if n >= 2:
        cut = max(1, min(cut, n - 1))
    train_items = items[:cut]
    val_items = items[cut:]

    def _materialize_split(split_items: List[Dict[str, Any]], images_root: Path, masks_root: Path) -> None:
        images_root.mkdir(parents=True, exist_ok=True)
        masks_root.mkdir(parents=True, exist_ok=True)
        for record in split_items:
            src = Path(record["source_image"])
            stem = f"task_{record['task_id']}_{src.stem}"
            dst_img = images_root / f"{stem}{src.suffix.lower() or '.jpg'}"
            dst_mask = masks_root / f"{stem}.png"
            shutil.copy2(src, dst_img)
            mask_arr = record.pop("_mask_array")
            Image.fromarray(mask_arr, mode="L").save(dst_mask)
            record["image_path"] = str(dst_img.resolve())
            record["mask_path"] = str(dst_mask.resolve())

    train_images = ds_root / "images" / "train"
    val_images = ds_root / "images" / "val"
    train_masks = ds_root / "masks" / "train"
    val_masks = ds_root / "masks" / "val"
    _materialize_split(train_items, train_images, train_masks)
    _materialize_split(val_items, val_images, val_masks)

    items_json = ds_root / "dataset_items.json"
    items_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    names_map = ", ".join(f"{i}: '{name}'" for i, name in enumerate(class_names))
    yaml_lines = [
        f"path: {str(ds_root.resolve())}",
        "train: images/train",
        "val: images/val",
        "train_masks: masks/train",
        "val_masks: masks/val",
        f"nc: {len(class_names)}" if class_names else "nc: 0",
        f"names: {{{names_map}}}" if class_names else "names: {}",
    ]
    data_yaml = ds_root / "data.yaml"
    _write_yaml(data_yaml, "\n".join(yaml_lines + [""]))

    dataset_config = ds_root / "dataset_config.json"
    dataset_manifest = {
        "task_type": "semantic_segmentation",
        "training_model": "yolo_semantic",
        "project_id": project.id,
        "dataset_root": str(ds_root),
        "data_yaml": str(data_yaml),
        "train_images_dir": str(train_images),
        "val_images_dir": str(val_images),
        "train_masks_dir": str(train_masks),
        "val_masks_dir": str(val_masks),
        "records_json": str(export_file),
        "items_json": str(items_json),
        "classes": class_names,
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
    export_format: str | None = None,
) -> Dict[str, Any]:
    """
    Prepare a training dataset based on the project's labeling interface.

    Output is always a JSON dataset manifest (`dataset_config.json`).
    """

    project = Project.objects.get(pk=project_id)
    spec = detect_training_interface(project)
    ds_root = get_datasets_root() / f"project_{project_id}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_root.mkdir(parents=True, exist_ok=True)

    task_type = spec["task_type"]
    training_model = spec["training_model"]
    default_export = _default_export_format_for_task(task_type)
    chosen_export = export_format or default_export

    if task_type == "classification":
        meta = _prepare_classification_dataset(
            project=project,
            ds_root=ds_root,
            train_ratio=train_ratio,
            seed=seed,
            export_format=chosen_export,
        )
    elif task_type in {"semantic", "semantic_segmentation"}:
        meta = _prepare_semantic_segmentation_dataset(
            project=project,
            ds_root=ds_root,
            train_ratio=train_ratio,
            seed=seed,
            export_format=chosen_export,
        )
    elif task_type in {"detect", "segmentation", "pose", "obb"}:
        meta = _prepare_yolo_zip_dataset(
            project=project,
            ds_root=ds_root,
            train_ratio=train_ratio,
            seed=seed,
            task_type=task_type,
            training_model=training_model,
            export_format=chosen_export,
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    meta = {**spec, **meta}
    (ds_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def resolve_deploy_task_context(
    project: Project,
    run_dir: Path,
    run_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Resolve task type / training model / default imgsz for Triton deployment.

    Priority: run_meta.params → dataset_config.json → project label interface.
    """

    from .yolo_catalog import get_task_defaults, resolve_yolo_task

    meta: Dict[str, Any] = dict(run_meta or {})
    if not meta and (run_dir / "run_meta.json").exists():
        try:
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    params = meta.get("params") or {}
    training_model = params.get("training_model") or meta.get("training_model")
    task_type = params.get("task_type") or meta.get("task")
    kpt_shape = None

    dataset_config_path = params.get("dataset_config")
    if dataset_config_path:
        dc = Path(str(dataset_config_path))
        if dc.exists():
            try:
                ds = json.loads(dc.read_text(encoding="utf-8"))
                training_model = training_model or ds.get("training_model")
                task_type = task_type or ds.get("task_type")
                kpt_shape = ds.get("kpt_shape")
            except Exception:
                pass

    if not training_model or not task_type:
        try:
            spec = detect_training_interface(project)
            training_model = training_model or spec.get("training_model")
            task_type = task_type or spec.get("task_type")
        except Exception:
            pass

    is_cnn = meta.get("kind") == "cnn_classification_train" or training_model == "cnn_classify"
    yolo_task = resolve_yolo_task(training_model, task_type)
    defaults = get_task_defaults("classify" if is_cnn else yolo_task)

    if is_cnn:
        resolved_task_type = "classification"
        resolved_training_model = "cnn_classify"
    else:
        resolved_task_type = task_type or (
            "classification"
            if yolo_task == "classify"
            else "semantic_segmentation"
            if yolo_task == "semantic"
            else "segmentation"
            if yolo_task == "segment"
            else yolo_task
        )
        if training_model:
            resolved_training_model = training_model
        elif yolo_task == "classify":
            resolved_training_model = "yolo_classify"
        else:
            resolved_training_model = f"yolo_{yolo_task}"

    return {
        "task": yolo_task,
        "task_type": resolved_task_type,
        "training_model": resolved_training_model,
        "kpt_shape": kpt_shape,
        "default_imgsz": int(defaults.get("imgsz", 640)),
        "is_cnn": is_cnn,
    }

