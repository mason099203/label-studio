"""
Ultralytics YOLO 任務、預設模型與訓練參數目錄（Train Server 獨立副本）。

參考：https://docs.ultralytics.com/zh
"""

from __future__ import annotations

from typing import Any, Dict, List

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
}

_SIZE_SUFFIXES = ("n", "s", "m", "l", "x")
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
}

TRAINING_MODEL_TO_TASK: Dict[str, str] = {
    "yolo_detect": "detect",
    "yolo_classify": "classify",
    "yolo_segment": "segment",
    "yolo_pose": "pose",
    "yolo_obb": "obb",
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
    return "detect"


def build_training_params(*, task: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = get_task_defaults(task)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
    return params
