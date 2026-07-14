/** Playground / 部署共用的訓練任務類型定義（對齊後端 detect_training_interface）。 */

/** Training 目前允許的 task_type（與後端 detect_training_interface 白名單對齊） */
export const ALLOWED_TRAINING_TASK_TYPES = new Set([
  "classification",
  "detect",
  "semantic_segmentation",
]);

/**
 * @param {string | null | undefined} taskType
 * @returns {boolean}
 */
export function isAllowedTrainingTaskType(taskType) {
  return ALLOWED_TRAINING_TASK_TYPES.has(taskType);
}

export const PLAYGROUND_TASK_TYPES = {
  detect: {
    label: "物件偵測（矩形框）",
    spatial: true,
    defaultImgsz: 640,
    detail: "Image + RectangleLabels",
  },
  segmentation: {
    label: "實例分割（多邊形）",
    spatial: true,
    defaultImgsz: 640,
    detail: "Image + PolygonLabels",
  },
  semantic_segmentation: {
    label: "語意分割（Brush/Mask）",
    spatial: true,
    defaultImgsz: 640,
    detail: "Image + BrushLabels / MaskLabels",
  },
  pose: {
    label: "姿態估計（框 + 關鍵點）",
    spatial: true,
    defaultImgsz: 640,
    detail: "Image + RectangleLabels + KeyPointLabels",
  },
  obb: {
    label: "旋轉框 OBB",
    spatial: true,
    defaultImgsz: 1024,
    detail: "Image + RectangleLabels（model_obb）",
  },
  classification: {
    label: "影像分類",
    spatial: false,
    defaultImgsz: 224,
    detail: "Image + Choices",
  },
};

/**
 * @param {string | null | undefined} value
 * @returns {keyof typeof PLAYGROUND_TASK_TYPES}
 */
export function normalizePlaygroundTaskType(value) {
  if (value && Object.prototype.hasOwnProperty.call(PLAYGROUND_TASK_TYPES, value)) {
    return value;
  }
  if (value === "semantic") return "semantic_segmentation";
  return "detect";
}

/**
 * @param {string} taskType
 * @returns {boolean}
 */
export function isSpatialYoloTaskType(taskType) {
  const row = PLAYGROUND_TASK_TYPES[normalizePlaygroundTaskType(taskType)];
  return Boolean(row?.spatial);
}

/**
 * @param {string} taskType
 * @param {number | null | undefined} deployedImgsz
 * @returns {number}
 */
export function playgroundInputSizeForTask(taskType, deployedImgsz) {
  const parsed = Number(deployedImgsz);
  if (Number.isFinite(parsed) && parsed > 0) return parsed;
  return PLAYGROUND_TASK_TYPES[normalizePlaygroundTaskType(taskType)]?.defaultImgsz ?? 640;
}

/**
 * 依 Labeling Interface XML 推斷 Playground 任務類型（對齊後端 detect_training_interface）。
 * @param {string | null | undefined} labelConfig
 * @returns {{ taskType: keyof typeof PLAYGROUND_TASK_TYPES, detail: string } | null}
 */
export function inferPlaygroundTaskTypeFromLabelConfig(labelConfig) {
  if (typeof labelConfig !== "string" || !labelConfig.trim()) return null;
  const xml = labelConfig.replace(/<!--[\s\S]*?-->/g, " ");

  const hasImage = /<Image\b/i.test(xml);
  const hasChoices = /<Choices\b/i.test(xml);
  const hasHyperText = /<HyperText\b/i.test(xml);
  const hasRectangle = /<RectangleLabels\b/i.test(xml);
  const hasKeyPoint = /<KeyPointLabels\b/i.test(xml);
  const hasPolygon = /<PolygonLabels\b/i.test(xml);
  const hasBrush = /<(BrushLabels|MaskLabels|BitmaskLabels)\b/i.test(xml);
  const obbEnabled = /<RectangleLabels\b[^>]*\bmodel_obb\s*=\s*["'](?:true|1|yes)["']/i.test(xml);

  if (hasImage && hasChoices) {
    return { taskType: "classification", detail: PLAYGROUND_TASK_TYPES.classification.detail };
  }

  if (hasChoices && !hasHyperText) {
    return { taskType: "classification", detail: "分類介面（Choices）" };
  }

  if (hasKeyPoint && hasImage) {
    if (!hasRectangle) {
      return {
        taskType: "pose",
        detail: "偵測到 KeyPointLabels；完整 pose 需同時設定 RectangleLabels",
      };
    }
    return { taskType: "pose", detail: PLAYGROUND_TASK_TYPES.pose.detail };
  }

  if (hasRectangle && hasImage && obbEnabled) {
    return { taskType: "obb", detail: PLAYGROUND_TASK_TYPES.obb.detail };
  }

  if (hasBrush && hasImage) {
    return {
      taskType: "semantic_segmentation",
      detail: PLAYGROUND_TASK_TYPES.semantic_segmentation.detail,
    };
  }

  if (hasRectangle && hasImage) {
    return { taskType: "detect", detail: PLAYGROUND_TASK_TYPES.detect.detail };
  }

  if (hasPolygon && hasImage) {
    return { taskType: "segmentation", detail: PLAYGROUND_TASK_TYPES.segmentation.detail };
  }

  const spatialMatch = xml.match(
    /<(RectangleLabels|PolygonLabels|KeyPointLabels|EllipseLabels|BrushLabels|MaskLabels|MagicWand)\b/i,
  );
  if (spatialMatch) {
    return { taskType: "detect", detail: `偵測／區域標註（${spatialMatch[1]}）` };
  }

  return null;
}

/**
 * @param {string | null | undefined} taskType
 * @returns {string}
 */
export function playgroundTaskTypeLabel(taskType) {
  return PLAYGROUND_TASK_TYPES[normalizePlaygroundTaskType(taskType)]?.label ?? String(taskType || "detect");
}

/** Ultralytics 任務鍵（對齊 yolo_catalog.resolve_yolo_task）。 */
export const YOLO_TASK_ALIASES = {
  detect: "detect",
  segmentation: "segment",
  semantic_segmentation: "semantic",
  semantic: "semantic",
  pose: "pose",
  obb: "obb",
  classification: "classify",
};

/**
 * @param {string | null | undefined} taskType
 * @returns {string}
 */
export function toUltralyticsTaskKey(taskType) {
  const normalized = normalizePlaygroundTaskType(taskType);
  return YOLO_TASK_ALIASES[normalized] ?? "detect";
}
