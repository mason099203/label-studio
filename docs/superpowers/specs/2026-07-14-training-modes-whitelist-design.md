# 設計：Training 僅開放 Classification／Bounding Box／Mask

**日期：** 2026-07-14  
**狀態：** 已核准，待撰寫實作計畫  
**作法：** 後端單一真相來源 + 前端提早禁用 Training 操作

## 目標

限制 Label Studio Training 模組，使用者只能訓練以下三種模式：

| 使用者面向 | Label config | `task_type` | `training_model` |
|------------|--------------|-------------|------------------|
| Classification | Image + Choices | `classification` | `yolo_classify`、`cnn_classify` |
| Bounding box | Image + RectangleLabels（非 OBB） | `detect` | `yolo_detect` |
| Mask segmentation | Image + BrushLabels / MaskLabels / BitmaskLabels | `semantic_segmentation` | `yolo_semantic` |

其他目前可被偵測到的訓練介面必須明確失敗，且不得啟動 job。

## 明確封鎖

| 模式 | 目前偵測方式 | 處理 |
|------|--------------|------|
| Pose | RectangleLabels + KeyPointLabels | 拒絕（`ValueError`／API 400） |
| OBB | RectangleLabels + `model_obb="true"` | 拒絕 |
| Polygon 實例分割 | PolygonLabels | 拒絕（`yolo_segment`） |
| 其他無法對應的介面 | 原本就 raise | 維持 raise；錯誤訊息改為只列出允許的三種 |

本變更不包含：刪除 trainer 程式碼、收斂 Playground 推論任務類型下拉、改 Train Server Dockerfile，或超越 Training 頁的 Model Deployment UI。

## 後端

### 1. `detect_training_interface`（`label_studio/training/datasets.py`）

主要閘道。目前會回傳 pose／obb／polygon 的 spec。改為：

- 偵測到 pose 或 OBB → 拋 `ValueError`，中文說明目前僅支援 Classification、Bounding Box、Mask Segmentation。
- 偵測到 PolygonLabels → 同樣拋 `ValueError`（不回傳 `yolo_segment`）。
- Classification、一般 RectangleLabels detect、Brush/Mask 語意分割 → 行為不變。
- 最後一段「不支援介面」訊息改為只列出允許的三種。

如此會一併擋住資料集產生，以及任何依賴介面偵測的呼叫端。

### 2. Jobs API 白名單（`label_studio/training/api.py`）

`ProjectTrainingJobsAPI.post` 目前允許：

`yolo_detect`、`yolo_classify`、`yolo_segment`、`yolo_pose`、`yolo_obb`、`yolo_semantic`、`cnn_classify`

縮成：

`yolo_detect`、`yolo_classify`、`yolo_semantic`、`cnn_classify`

其餘用既有「unsupported training_model」回應拒絕，避免舊的 `dataset_config.json` 仍帶被封鎖的 `training_model` 卻能開訓。

### 3. 錯誤處理

對使用者統一中文說明，例如：

> 目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。

Interface／prepare 等已把 `ValueError` 轉成 400 的端點維持該行為。若 prepare-dataset 目前會變成 500，對齊成與 interface API 相同的 400。

## 前端

### Training 頁（`web/apps/labelstudio/src/pages/TrainingPage/TrainingPage.jsx`）

進頁時（與現有專案／模型載入並行）呼叫 training interface API。

- 成功且 `training_model`／`task_type` 屬於允許三種 → 正常流程。
- 失敗或屬於被封鎖類型 → 設 `unsupportedTraining = true`（或同等狀態）。

不支援時：

- 禁用 **Start Training**（`canStart` 為 false）。
- 一併禁用 **產生／準備 dataset**（避免產生被封鎖類型的資料集）。
- 顯示 `disabledReason`（或同等提示）說明目前僅支援上述三種模式。

前端不可單獨充當閘道；後端仍為權威。

### 本變更不做

- Playground 的 `PLAYGROUND_TASK_TYPES` 本迭代不必收斂（推論預覽 ≠ 啟動訓練）。
- v1 不強制 JS／Python 共用常數；必要時在註解中兩邊對齊即可。

## 測試

- 擴充 `label_studio/tests/test_training_datasets.py`（或鄰近測試）：
  - Classification／detect／brush-mask 仍可通過 `detect_training_interface`。
  - Pose／OBB／Polygon 應拋 `ValueError`。
- 可選 API 測試：以 `training_model=yolo_pose`（或其他被封鎖值）建立 job 應回 400。

## 驗收條件

1. Choices／RectangleLabels（非 OBB）／Brush|Mask|Bitmask 專案可如今日一般準備資料集並開始訓練。
2. Pose、OBB、Polygon 專案不可準備或開始訓練；前端 Start Training 與準備資料集按鈕為禁用，並顯示清楚原因。
3. 即使存在舊的 dataset_config，直接 `POST .../training/jobs/` 帶被封鎖的 `training_model` 仍被拒絕。
4. 不刪除無關 Trainer 程式；被擋模式保留在程式庫中以便日後重開。
