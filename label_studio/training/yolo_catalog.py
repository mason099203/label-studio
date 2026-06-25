"""
Ultralytics YOLO 任務、預設模型與訓練參數目錄。

參考：https://docs.ultralytics.com/zh
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# 各任務預設訓練參數（對齊 Ultralytics CLI / model.train 預設值）
YOLO_TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "detect": {
        "task": "detect",
        "training_model": "yolo_detect",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "patience": 100,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
    },
    "segment": {
        "task": "segment",
        "training_model": "yolo_segment",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "patience": 100,
        "mask_ratio": 4,
    },
    "classify": {
        "task": "classify",
        "training_model": "yolo_classify",
        "epochs": 100,
        "imgsz": 224,
        "batch": 16,
        "workers": 8,
        "patience": 100,
    },
    "pose": {
        "task": "pose",
        "training_model": "yolo_pose",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "patience": 100,
        "kobj": 1.0,
    },
    "obb": {
        "task": "obb",
        "training_model": "yolo_obb",
        "epochs": 100,
        "imgsz": 1024,
        "batch": 16,
        "workers": 8,
        "patience": 100,
    },
    "semantic": {
        "task": "semantic",
        "training_model": "yolo_semantic",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "patience": 100,
    },
}

# 各尺寸後綴
_SIZE_SUFFIXES = ("n", "s", "m", "l", "x")

# YOLO11（穩定生產）與 YOLO26（新一代）預訓練權重
_YOLO11_VERSION = "yolo11"
_YOLO26_VERSION = "yolo26"


def _build_weight_names(version: str, task_suffix: str = "") -> List[str]:
    return [f"{version}{size}{task_suffix}.pt" for size in _SIZE_SUFFIXES]


YOLO_PRESET_MODELS: Dict[str, List[Dict[str, str]]] = {
    "detect": [
        {"id": name, "name": name, "family": fam, "task": "detect"}
        for fam, names in (
            ("YOLO11", _build_weight_names(_YOLO11_VERSION)),
            ("YOLO26", _build_weight_names(_YOLO26_VERSION)),
        )
        for name in names
    ],
    "segment": [
        {"id": name, "name": name, "family": fam, "task": "segment"}
        for fam, names in (
            ("YOLO11", _build_weight_names(_YOLO11_VERSION, "-seg")),
            ("YOLO26", _build_weight_names(_YOLO26_VERSION, "-seg")),
        )
        for name in names
    ],
    "classify": [
        {"id": name, "name": name, "family": fam, "task": "classify"}
        for fam, names in (
            ("YOLO11", _build_weight_names(_YOLO11_VERSION, "-cls")),
            ("YOLO26", _build_weight_names(_YOLO26_VERSION, "-cls")),
        )
        for name in names
    ],
    "pose": [
        {"id": name, "name": name, "family": fam, "task": "pose"}
        for fam, names in (
            ("YOLO11", _build_weight_names(_YOLO11_VERSION, "-pose")),
            ("YOLO26", _build_weight_names(_YOLO26_VERSION, "-pose")),
        )
        for name in names
    ],
    "obb": [
        {"id": name, "name": name, "family": fam, "task": "obb"}
        for fam, names in (
            ("YOLO11", _build_weight_names(_YOLO11_VERSION, "-obb")),
            ("YOLO26", _build_weight_names(_YOLO26_VERSION, "-obb")),
        )
        for name in names
    ],
    "semantic": [
        {"id": name, "name": name, "family": fam, "task": "semantic"}
        for fam, names in (
            ("YOLO26", _build_weight_names(_YOLO26_VERSION, "-sem")),
        )
        for name in names
    ],
}

# Label Studio 現有 training_model 對應 YOLO task
TRAINING_MODEL_TO_TASK: Dict[str, str] = {
    "yolo_detect": "detect",
    "yolo_classify": "classify",
    "yolo_segment": "segment",
    "yolo_pose": "pose",
    "yolo_obb": "obb",
    "yolo_semantic": "semantic",
}

YOLO_TASK_WEIGHT_SUFFIX: Dict[str, str] = {
    "detect": "",
    "classify": "-cls",
    "segment": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "semantic": "-sem",
}


def get_task_defaults(task: str) -> Dict[str, Any]:
    return dict(YOLO_TASK_DEFAULTS.get(task, YOLO_TASK_DEFAULTS["detect"]))


def get_preset_models_for_task(task: str) -> List[Dict[str, str]]:
    return list(YOLO_PRESET_MODELS.get(task, YOLO_PRESET_MODELS["detect"]))


def resolve_yolo_task(training_model: str | None, task_type: str | None = None) -> str:
    if training_model and training_model in TRAINING_MODEL_TO_TASK:
        return TRAINING_MODEL_TO_TASK[training_model]
    if task_type == "classification":
        return "classify"
    if task_type == "segmentation":
        return "segment"
    if task_type == "pose":
        return "pose"
    if task_type == "obb":
        return "obb"
    if task_type in {"semantic", "semantic_segmentation"}:
        return "semantic"
    return "detect"


def resolve_yolo_weights_name(base_weights: str, task: str) -> str:
    """
    Map a generic YOLO weight name to the task-specific variant when needed.

    Example: yolo11n.pt + segment -> yolo11n-seg.pt
    """
    path_like = base_weights.replace("\\", "/")
    name = path_like.rsplit("/", 1)[-1]
    if not name.endswith(".pt"):
        return base_weights

    suffix = YOLO_TASK_WEIGHT_SUFFIX.get(task, "")
    if not suffix:
        return base_weights

    for known in ("-cls", "-seg", "-pose", "-obb", "-sem"):
        if known in name:
            return base_weights

    stem = name[:-3]
    task_name = f"{stem}{suffix}.pt"
    parent = base_weights.rsplit("/", 1)[0] if "/" in path_like else base_weights.rsplit("\\", 1)[0] if "\\" in base_weights else ""
    if parent and parent != base_weights:
        sep = "\\" if "\\" in base_weights and "/" not in base_weights else "/"
        return f"{parent}{sep}{task_name}"
    return task_name


def infer_yolo_task_from_weights(weights_path: str | None) -> str | None:
    """Infer Ultralytics task from weight filename suffix (e.g. yolo11n-cls.pt)."""
    if not weights_path:
        return None
    name = Path(str(weights_path)).name.lower()
    if "-cls" in name:
        return "classify"
    if "-seg" in name:
        return "segment"
    if "-pose" in name:
        return "pose"
    if "-obb" in name:
        return "obb"
    if "-sem" in name:
        return "semantic"
    return None


def reconcile_yolo_task(
    training_model: str | None,
    task_type: str | None,
    weights_path: str | None,
) -> str:
    """
    Align metadata task with weight suffix so Ultralytics task matches `data` format.

    Weight suffix takes precedence when it conflicts with dataset_config metadata.
    """
    meta_task = resolve_yolo_task(training_model, task_type)
    weights_task = infer_yolo_task_from_weights(weights_path)
    if weights_task and weights_task != meta_task:
        return weights_task
    return meta_task


def infer_dataset_layout(dataset_meta: Dict[str, Any]) -> str:
    """Return ``classify`` (ImageFolder) or ``yaml`` (detect/segment/pose/obb/semantic)."""
    if dataset_meta.get("train_dir"):
        return "classify"
    if dataset_meta.get("data_yaml") or dataset_meta.get("labels_dir"):
        return "yaml"
    return "unknown"


def is_weight_compatible_with_task(weights_path: str | None, task: str) -> bool:
    """Whether a weight filename matches the requested Ultralytics task."""
    inferred = infer_yolo_task_from_weights(weights_path)
    if inferred is None:
        return task == "detect"
    return inferred == task


def validate_yolo_training_request(
    *,
    base_weights: str,
    dataset_meta: Dict[str, Any],
    training_model: str | None = None,
    dataset_root: Path | None = None,
) -> str | None:
    """
    Validate weights, metadata, and on-disk dataset layout before enqueueing training.

    Returns a user-facing error message, or None when the request looks consistent.
    """
    tm = training_model or dataset_meta.get("training_model")
    task_type = dataset_meta.get("task_type")
    task = reconcile_yolo_task(tm, task_type, base_weights)
    layout = infer_dataset_layout(dataset_meta)

    if task == "classify" and layout == "yaml":
        return (
            "權重或任務為「分類」，但資料集為偵測/分割格式（含 data.yaml）。"
            "偵測/分割專案請使用 yolo11n.pt、yolo11n-seg.pt 等對應權重；"
            "分類專案（Image + Choices）請重新「生成訓練資料集」並選 yolo11n-cls.pt。"
        )
    if task != "classify" and layout == "classify":
        suffix = YOLO_TASK_WEIGHT_SUFFIX.get(task, "")
        example = f"yolo11n{suffix}.pt" if suffix else "yolo11n.pt"
        return (
            f"資料集為分類格式（train/<類別>/），但權重/任務為「{task}」。"
            f"請改用 {example}，或確認專案標註介面與資料集類型一致。"
        )

    root = dataset_root
    if root is None:
        raw = dataset_meta.get("dataset_root")
        root = Path(str(raw)) if raw else None

    if root and root.exists():
        try:
            resolve_train_data_path(task, dataset_meta, root)
        except (ValueError, FileNotFoundError) as exc:
            return str(exc)
    return None


def resolve_train_data_path(
    task: str,
    dataset_meta: Dict[str, Any],
    dataset_root: Path,
) -> str:
    """Return Ultralytics `data` argument for the given task and dataset layout."""
    root = Path(dataset_root)

    if task == "classify":
        train_dir = dataset_meta.get("train_dir")
        if train_dir:
            train_parent = Path(str(train_dir)).parent
            if train_parent.exists() and (train_parent / "train").is_dir():
                return str(train_parent.resolve())
        if (root / "train").is_dir():
            return str(root.resolve())
        raise ValueError(
            "YOLO 分類需要資料集根目錄含 train/<類別>/ 與 val/<類別>/ 子目錄。"
            "請確認專案為 Image + Choices，並重新「生成訓練資料集」。"
            "若為偵測/分割專案，請改用 yolo11n.pt / yolo11n-seg.pt 等對應權重，勿用 *-cls.pt。"
        )

    data_yaml = dataset_meta.get("data_yaml")
    if data_yaml and Path(str(data_yaml)).exists():
        return str(Path(str(data_yaml)).resolve())
    candidate = root / "data.yaml"
    if candidate.exists():
        return str(candidate.resolve())
    raise FileNotFoundError(
        f"找不到 data.yaml（task={task}）。請重新生成 YOLO 資料集（detect/segment/pose/obb/semantic）。"
    )


def build_training_params(
    *,
    task: str,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    params = get_task_defaults(task)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
    return params
