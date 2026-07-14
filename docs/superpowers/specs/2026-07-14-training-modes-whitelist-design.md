# Design: Limit Training to Classification / Bounding Box / Mask

**Date:** 2026-07-14  
**Status:** Approved for implementation planning  
**Approach:** Backend single source of truth + frontend early disable of Training actions

## Goal

Restrict the Label Studio Training module so users can only train these three modes:

| User-facing mode | Label config | `task_type` | `training_model` |
|------------------|--------------|-------------|------------------|
| Classification | Image + Choices | `classification` | `yolo_classify`, `cnn_classify` |
| Bounding box | Image + RectangleLabels (non-OBB) | `detect` | `yolo_detect` |
| Mask segmentation | Image + BrushLabels / MaskLabels / BitmaskLabels | `semantic_segmentation` | `yolo_semantic` |

All other currently detected training interfaces must fail clearly and must not start a job.

## Explicitly blocked

| Mode | Detection today | Action |
|------|-----------------|--------|
| Pose | RectangleLabels + KeyPointLabels | Reject (`ValueError` / API 400) |
| OBB | RectangleLabels + `model_obb="true"` | Reject |
| Polygon instance segmentation | PolygonLabels | Reject (`yolo_segment`) |
| Anything else unsupported by detection | already raises | Keep raising; refresh error message to list only the three allowed modes |

Out of scope: deleting trainer code paths, changing Playground inference task-type dropdown, Train Server Dockerfile, or Model Deployment UI beyond Training page.

## Backend

### 1. `detect_training_interface` (`label_studio/training/datasets.py`)

Primary gate. Today it returns specs for pose / obb / polygon. Change behavior:

- When pose or OBB would be detected → raise `ValueError` with a Chinese message that Training currently only supports Classification、Bounding Box、Mask Segmentation.
- When PolygonLabels would be detected → raise the same style of `ValueError` (do not return `yolo_segment`).
- Classification, detect (plain RectangleLabels), and brush/mask semantic segmentation continue to return specs unchanged.
- Update the final “unsupported interface” message to list only the three allowed interfaces.

This automatically blocks dataset generation and any caller that relies on interface detection.

### 2. Jobs API whitelist (`label_studio/training/api.py`)

`ProjectTrainingJobsAPI.post` currently allows:

`yolo_detect`, `yolo_classify`, `yolo_segment`, `yolo_pose`, `yolo_obb`, `yolo_semantic`, `cnn_classify`

Shrink to:

`yolo_detect`, `yolo_classify`, `yolo_semantic`, `cnn_classify`

Reject others with the existing unsupported-model response pattern. This prevents starting training from an old `dataset_config.json` that still names a blocked `training_model`.

### 3. Error handling

Prefer consistent Chinese user-facing text, e.g.:

> 目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。

Interface / prepare endpoints that already surface `ValueError` should continue to return 400 with that detail. Align prepare-dataset error handling with interface API if prepare currently bubbles to 500.

## Frontend

### Training page (`web/apps/labelstudio/src/pages/TrainingPage/TrainingPage.jsx`)

On load (alongside existing project / models loading), call the training interface API.

- If interface succeeds and `training_model` / `task_type` is one of the allowed three → normal flow.
- If interface fails or returns a blocked type → set `unsupportedTraining = true` (or equivalent).

When unsupported:

- Disable **Start Training** (`canStart` false).
- Disable **prepare / generate dataset** action as well (user cannot prepare a blocked dataset).
- Show `disabledReason` (or equivalent banner text) explaining only the three modes are supported.

Do not rely on frontend alone; backend remains authoritative.

### Out of scope for this change

- Playground `PLAYGROUND_TASK_TYPES` need not be narrowed in this iteration (inference preview ≠ training start).
- No new shared package constant between JS and Python required for v1; keep lists mirrored in comments if helpful.

## Tests

- Extend `label_studio/tests/test_training_datasets.py` (or adjacent):
  - Classification / detect / brush-mask still pass `detect_training_interface`.
  - Pose / OBB / Polygon raise `ValueError`.
- Optional API test: create job with `training_model=yolo_pose` (or other blocked value) returns 400.

## Acceptance criteria

1. Projects with Choices / RectangleLabels (non-OBB) / Brush|Mask|Bitmask can prepare dataset and start training as today.
2. Projects with pose, OBB, or Polygon cannot prepare or start training; frontend Start Training and prepare buttons are disabled with a clear reason.
3. Direct `POST .../training/jobs/` with blocked `training_model` is rejected even if an old dataset_config exists.
4. No unrelated Trainer code deletion; blocked modes remain in codebase for a future reopen.
