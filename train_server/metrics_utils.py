"""JSON-safe metric extraction for Ultralytics training results."""

from __future__ import annotations

import math
from typing import Any, Dict


def json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-serializable types."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if callable(value):
        return None
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if callable(v):
                continue
            safe = json_safe(v)
            if safe is not None:
                out[str(k)] = safe
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value if not callable(v)]
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(value)


def normalize_trainer_metrics(raw: Any) -> Dict[str, Any]:
    """Extract epoch metrics from Ultralytics trainer without methods/non-serializable objects."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = raw.items()
    elif hasattr(raw, "items"):
        try:
            items = raw.items()
        except Exception:
            return {}
    else:
        return {}

    out: Dict[str, Any] = {}
    for key, val in items:
        if callable(val):
            continue
        safe = json_safe(val)
        if safe is not None:
            out[str(key)] = safe
    return out


def collect_yolo_val_metrics(val_res: Any, task: str) -> Dict[str, Any]:
    """Collect validation metrics from Ultralytics val() result."""
    metrics: Dict[str, Any] = {}
    if task == "classify":
        for key in ("top1", "top5", "fitness"):
            val = getattr(val_res, key, None)
            if val is None or callable(val):
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except (TypeError, ValueError):
                continue
        return metrics

    groups = []
    if task in {"detect", "obb", "pose"}:
        groups.append(getattr(val_res, "box", None))
    if task in {"segment", "semantic"}:
        groups.append(getattr(val_res, "seg", None) or getattr(val_res, "mask", None))
    if task == "semantic":
        groups.append(getattr(val_res, "semantic", None))

    for group in groups:
        if group is None:
            continue
        for key in ("map50", "map", "map75", "mp", "mr", "fitness"):
            val = getattr(group, key, None)
            if val is None or callable(val):
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except (TypeError, ValueError):
                continue

    if not metrics:
        for key in ("map50", "map", "map75", "mp", "mr", "fitness"):
            val = getattr(val_res, key, None)
            if val is None or callable(val):
                continue
            try:
                f = float(val)
                metrics[key] = f if math.isfinite(f) else None
            except (TypeError, ValueError):
                continue
    return metrics
