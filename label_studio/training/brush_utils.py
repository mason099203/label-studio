"""Brush / mask annotation helpers for YOLO semantic segmentation datasets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from label_studio_sdk.converter.brush import decode_rle

_BRUSH_RESULT_TYPES = {"brushlabels", "bitmasklabels", "masklabels"}


def _normalize_brush_result_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    result_type = str(item.get("type") or "").lower()
    if result_type not in _BRUSH_RESULT_TYPES:
        return None

    value = item.get("value") or {}
    rle = value.get("rle") if isinstance(value, dict) else None
    if rle is None:
        rle = item.get("rle")
    if not rle:
        return None

    labels = None
    if isinstance(value, dict):
        for key in ("brushlabels", "labels", "bitmasklabels"):
            if key in value and value[key]:
                labels = list(value[key])
                break
    labels = labels or item.get("brushlabels") or item.get("labels") or ["no_label"]

    width = int(item.get("original_width") or value.get("original_width") or 0)
    height = int(item.get("original_height") or value.get("original_height") or 0)
    if width <= 0 or height <= 0:
        return None

    return {
        "rle": rle,
        "labels": [str(x) for x in labels],
        "original_width": width,
        "original_height": height,
    }


def brush_result_to_binary_mask(item: Dict[str, Any]) -> Tuple[np.ndarray, List[str]] | None:
    """Decode one brush/mask annotation item to a boolean (H, W) mask."""
    normalized = _normalize_brush_result_item(item)
    if not normalized:
        return None

    width = normalized["original_width"]
    height = normalized["original_height"]
    flat = decode_rle(normalized["rle"])
    rgba = np.reshape(flat, [height, width, 4])
    mask = rgba[:, :, 3] > 0
    return mask, normalized["labels"]


def build_semantic_class_mask(
    annotation_items: Iterable[Dict[str, Any]],
    *,
    class_names: List[str],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Merge brush regions into a single uint8 class-id mask (0=background, 255=ignore unused).

    Later regions overwrite earlier ones on overlap (last annotation wins).
    """
    class_index = {name: idx + 1 for idx, name in enumerate(class_names)}
    out = np.zeros((height, width), dtype=np.uint8)

    for item in annotation_items:
        decoded = brush_result_to_binary_mask(item)
        if decoded is None:
            continue
        mask, labels = decoded
        if mask.shape[0] != height or mask.shape[1] != width:
            continue
        label_name = labels[0] if labels else "no_label"
        if label_name not in class_index:
            class_index[label_name] = len(class_index) + 1
            class_names.append(label_name)
        class_id = class_index[label_name]
        out[mask] = np.uint8(class_id)

    return out
