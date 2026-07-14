# Training 模式白名單 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Training 僅允許 Classification、Bounding Box（detect）、Mask（semantic）三種模式；其餘在介面偵測與 Jobs API 拒絕，Training 頁禁用「生成資料集／開始訓練」。

**Architecture:** 後端以 `detect_training_interface` 為單一真相來源，對 pose／obb／polygon 拋 `ValueError`；Jobs API 縮短 `training_model` 白名單防舊 dataset_config 繞過。前端進頁呼叫 `trainingInterface`，失敗或不在允許集合時設 `unsupportedTraining`，一併禁用兩顆按鈕並顯示中文原因。

**Tech Stack:** Django / DRF、pytest、React（TrainingPage.jsx）

**Spec:** `docs/superpowers/specs/2026-07-14-training-modes-whitelist-design.md`

---

## 檔案對照

| 檔案 | 責任 |
|------|------|
| `label_studio/training/datasets.py` | 允許／拒絕介面偵測；共用錯誤訊息常數 |
| `label_studio/training/api.py` | Jobs 白名單；prepare 的 `ValueError` → 400 |
| `label_studio/tests/test_training_datasets.py` | 介面偵測合約測試（改寫既有 pose／obb／polygon） |
| `web/apps/labelstudio/src/pages/TrainingPage/TrainingPage.jsx` | 進頁檢查；禁用按鈕與 `disabledReason` |

不動：Playground、trainer 本體、`train_server/`、Deploy 既有 `dataset_config` 路徑（`resolve_deploy_task_context` 對 `detect_training_interface` 失敗已有 `except Exception: pass`）。

**共用錯誤訊息（全任務一字不差）：**

```text
目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。
```

**允許的 `training_model`（Jobs）：** `yolo_detect`、`yolo_classify`、`yolo_semantic`、`cnn_classify`

**允許的 `task_type`（前端）：** `classification`、`detect`、`semantic_segmentation`

---

### Task 1: 改寫介面偵測測試（先紅）

**Files:**
- Modify: `label_studio/tests/test_training_datasets.py`
- （實作在 Task 2）

- [ ] **Step 1: 將 pose／obb／polygon 測試改為預期 raise**

把下列三個測試改成 `pytest.raises(ValueError, match="目前僅支援")`（刪掉成功 assert）：

```python
def test_detect_training_interface_rejects_polygon_segmentation():
    project = StubProject(
        {
            "label": {
                "type": "PolygonLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["cat"],
            }
        }
    )
    with pytest.raises(ValueError, match="目前僅支援"):
        detect_training_interface(project)


def test_detect_training_interface_rejects_pose():
    project = StubProject(
        {
            "bbox": {
                "type": "RectangleLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["person"],
            },
            "kp": {
                "type": "KeyPointLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["nose", "eye"],
                "labels_attrs": {
                    "nose": {"model_index": "0"},
                    "eye": {"model_index": "1"},
                },
            },
        }
    )
    with pytest.raises(ValueError, match="目前僅支援"):
        detect_training_interface(project)


def test_detect_training_interface_rejects_obb_from_model_obb_attr():
    label_config = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_obb="true">
    <Label value="ship"/>
  </RectangleLabels>
</View>"""
    project = StubProject(
        {
            "label": {
                "type": "RectangleLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["ship"],
            }
        },
        label_config=label_config,
    )
    with pytest.raises(ValueError, match="目前僅支援"):
        detect_training_interface(project)
```

保留不變：
- `test_detect_training_interface_accepts_plain_image_value_for_choices`
- `test_detect_training_interface_detects_rectangle_detect`
- `test_detect_training_interface_detects_brush_semantic_segmentation`
- `test_detect_training_interface_rejects_pose_without_rectangle`（仍可 match `"RectangleLabels"`）
- `test_resolve_deploy_task_context_from_dataset_meta`（仍從 dataset_config 讀 `yolo_segment`，不呼叫成功路徑的 detect）

並把舊函式名改掉：
- `test_detect_training_interface_detects_polygon_segmentation` → `test_detect_training_interface_rejects_polygon_segmentation`
- `test_detect_training_interface_detects_pose` → `test_detect_training_interface_rejects_pose`
- `test_detect_training_interface_detects_obb_from_model_obb_attr` → `test_detect_training_interface_rejects_obb_from_model_obb_attr`

- [ ] **Step 2: 跑測試確認失敗（尚未改實作）**

```bash
cd label_studio
DJANGO_DB=sqlite DJANGO_SETTINGS_MODULE=core.settings.label_studio poetry run pytest tests/test_training_datasets.py::test_detect_training_interface_rejects_polygon_segmentation tests/test_training_datasets.py::test_detect_training_interface_rejects_pose tests/test_training_datasets.py::test_detect_training_interface_rejects_obb_from_model_obb_attr -v
```

Expected: FAIL（函式仍回傳 spec，不會 raise）

- [ ] **Step 3: Commit 測試改寫**

```bash
git add label_studio/tests/test_training_datasets.py
git commit -m "test: expect training interface to reject pose/obb/polygon"
```

---

### Task 2: 實作 `detect_training_interface` 白名單

**Files:**
- Modify: `label_studio/training/datasets.py`（約 436–530 行 `detect_training_interface`）

- [ ] **Step 1: 在模組層加入常數（靠近其他 training helpers）**

```python
TRAINING_UNSUPPORTED_INTERFACE_MSG = (
    "目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、"
    "Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。"
)
```

- [ ] **Step 2: 改寫 `detect_training_interface` 分支**

在既有結構上：

1. docstring「Supported mappings」改成只列三種允許介面，並註明 pose／obb／polygon 目前拒絕。
2. **KeyPointLabels 分支**：不論是否有 RectangleLabels，改為：

```python
if _project_has_control_type(parsed, "KeyPointLabels"):
    raise ValueError(TRAINING_UNSUPPORTED_INTERFACE_MSG)
```

（可刪除原本「需要同時設定 RectangleLabels…」那段，因為 pose 整類不開放；`test_detect_training_interface_rejects_pose_without_rectangle` 的 match 改成 `"目前僅支援"`——若 Task 1 已留 `"RectangleLabels"`，在本 Task 同步更新該測試的 match。）

3. **OBB 分支**（`RectangleLabels` + `model_obb`）：不 return spec，改：

```python
if _project_has_control_type(parsed, "RectangleLabels") and _rectangle_labels_use_obb(project):
    raise ValueError(TRAINING_UNSUPPORTED_INTERFACE_MSG)
```

4. **Brush/Mask**、**RectangleLabels detect**、**Choices**：保持現有 return。
5. **PolygonLabels 分支**：不 return，改：

```python
if control_type == "PolygonLabels" and _control_has_image_input(info):
    raise ValueError(TRAINING_UNSUPPORTED_INTERFACE_MSG)
```

6. 函式末尾通用 `raise ValueError(...)` 改為：

```python
raise ValueError(TRAINING_UNSUPPORTED_INTERFACE_MSG)
```

同時更新 Task 1 留下的 `test_detect_training_interface_rejects_pose_without_rectangle`：

```python
with pytest.raises(ValueError, match="目前僅支援"):
    detect_training_interface(project)
```

- [ ] **Step 3: 跑測試通過**

```bash
cd label_studio
DJANGO_DB=sqlite DJANGO_SETTINGS_MODULE=core.settings.label_studio poetry run pytest tests/test_training_datasets.py -v
```

Expected: 全部 PASS（含 deploy context 測試）

- [ ] **Step 4: Commit**

```bash
git add label_studio/training/datasets.py label_studio/tests/test_training_datasets.py
git commit -m "fix: limit detect_training_interface to classify/detect/mask"
```

---

### Task 3: Jobs 白名單 + prepare `ValueError` → 400

**Files:**
- Modify: `label_studio/training/api.py`
  - `ProjectTrainingJobsAPI.post` 約 927 行
  - `ProjectTrainingDatasetPrepareAPI.post` 約 1067–1085 行

- [ ] **Step 1: 縮短白名單**

將：

```python
if training_model not in {"yolo_detect", "yolo_classify", "yolo_segment", "yolo_pose", "yolo_obb", "yolo_semantic", "cnn_classify"}:
```

改為：

```python
_ALLOWED_TRAINING_MODELS = {
    "yolo_detect",
    "yolo_classify",
    "yolo_semantic",
    "cnn_classify",
}
if training_model not in _ALLOWED_TRAINING_MODELS:
```

（常數可放在該函式上方模組層；detail 訊息可維持 `Unsupported training_model in dataset_config: ...`，或附加中文說明——至少 400。）

下方 job 選擇仍可保留 `yolo_segment` 等分支（死碼可留），因為白名單已擋；**不要刪** trainer job 映射以外的大段邏輯（符合 spec「不刪 trainer」）。

- [ ] **Step 2: prepare API 接住 ValueError**

```python
def post(self, request, pk: int, *args, **kwargs):
    project = _get_project_for_user(request, pk)

    payload = request.data or {}
    train_ratio = float(payload.get("train_ratio", 0.8))
    seed = int(payload.get("seed", 42))
    export_format = payload.get("export_format")

    try:
        meta = prepare_training_dataset_for_project(
            project_id=project.id,
            train_ratio=train_ratio,
            seed=seed,
            export_format=export_format,
        )
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    return Response(meta, status=status.HTTP_200_OK)
```

- [ ] **Step 3: 手動／簡易驗證白名單邏輯（可選單元片段）**

若沒有 Jobs API 既有測試檔，可在 `test_training_datasets.py` 加一個純邏輯備註測試不強制；最低限度：

```bash
cd label_studio
DJANGO_DB=sqlite DJANGO_SETTINGS_MODULE=core.settings.label_studio poetry run pytest tests/test_training_datasets.py -v
```

Expected: PASS

人工檢查：`api.py` 中 `_ALLOWED_TRAINING_MODELS`（或 inline set）不含 `yolo_pose`／`yolo_obb`／`yolo_segment`。

- [ ] **Step 4: Commit**

```bash
git add label_studio/training/api.py
git commit -m "fix: restrict training jobs whitelist and prepare ValueError status"
```

---

### Task 4: Training 頁禁用按鈕

**Files:**
- Modify: `web/apps/labelstudio/src/pages/TrainingPage/TrainingPage.jsx`

- [ ] **Step 1: 狀態與允許集合**

在既有 `useState` 區塊加：

```javascript
const [unsupportedTraining, setUnsupportedTraining] = useState(false);
const [unsupportedTrainingDetail, setUnsupportedTrainingDetail] = useState(null);
```

在檔案上方（import 後）加：

```javascript
const ALLOWED_TRAINING_TASK_TYPES = new Set([
  "classification",
  "detect",
  "semantic_segmentation",
]);

const TRAINING_UNSUPPORTED_HINT =
  "目前僅支援 Classification（Choices）、Bounding Box（RectangleLabels）、Mask Segmentation（Brush/Mask）訓練；pose、OBB、Polygon 等介面尚無法訓練。";
```

- [ ] **Step 2: 進頁呼叫 `trainingInterface`**

在既有 `useEffect`（載入 `project`／`loadModels`／`trainingHistory`，約 206–241 行）內加入：

```javascript
api
  .callApi("trainingInterface", {
    params: { pk: pageParams.id },
    errorFilter: () => true,
  })
  .then((res) => {
    if (cancelled) return;
    if (!res || res?.detail || !ALLOWED_TRAINING_TASK_TYPES.has(res.task_type)) {
      setUnsupportedTraining(true);
      setUnsupportedTrainingDetail(
        typeof res?.detail === "string" ? res.detail : TRAINING_UNSUPPORTED_HINT,
      );
      return;
    }
    setUnsupportedTraining(false);
    setUnsupportedTrainingDetail(null);
    setTrainingSpec((prev) => prev ?? res);
  });
```

（`ApiProvider` 對 400 常見行為：可能回 `null` 或帶 `detail` 的物件——兩邊都要當成 unsupported。）

- [ ] **Step 3: 更新 `canStart` / `disabledReason` / 生成按鈕**

```javascript
const canStart =
  !unsupportedTraining &&
  hasSelectableModel &&
  (trainingState === "idle" || trainingState === "error") &&
  Boolean(selectedBaseWeights) &&
  Boolean(datasetConfigPath) &&
  trainServerReady;

const disabledReason = (() => {
  if (unsupportedTraining) {
    return unsupportedTrainingDetail || TRAINING_UNSUPPORTED_HINT;
  }
  // …既有分支不變…
})();
```

生成資料集按鈕：

```javascript
disabled={!isDefined(pageParams?.id) || unsupportedTraining || preparingDataset}
```

可選：`prepareDatasetFromExport` 開頭若 `unsupportedTraining` 直接 return。

- [ ] **Step 4: 手動驗收**

1. Choices／Rectangle／Brush 專案：兩按鈕可依原邏輯啟用。
2. Polygon 或 KeyPoint pose 專案：兩按鈕 disabled，hint 顯示上述中文。
3. 不跑完整 e2e 也可接受本 Task；若環境可跑：

```bash
cd web
yarn lint
```

Expected: 無新增 Biome 錯誤於 `TrainingPage.jsx`

- [ ] **Step 5: Commit**

```bash
git add web/apps/labelstudio/src/pages/TrainingPage/TrainingPage.jsx
git commit -m "fix: disable Training actions for unsupported label interfaces"
```

---

### Task 5: 端到端對照驗收

- [ ] **Step 1: 後端回歸**

```bash
cd label_studio
DJANGO_DB=sqlite DJANGO_SETTINGS_MODULE=core.settings.label_studio poetry run pytest tests/test_training_datasets.py -v
```

Expected: PASS

- [ ] **Step 2: Spec 逐條勾選**

| Spec 條件 | 對應 |
|-----------|------|
| Choices／detect／mask 可訓練 | Task 2 通過測試 + 前端不設 unsupported |
| pose／obb／polygon 不可準備／開訓 | Task 2 raise + Task 4 禁用 |
| 舊 dataset_config 帶 yolo_pose 開 job 被拒 | Task 3 白名單 |
| 不刪 trainer | Task 3 未刪 job／cnn_trainer |

- [ ] **Step 3: 若有未提交變更則 commit 文件狀態（可選）**

無需再改 code 時跳過。

---

## Spec 覆蓋自檢

| Spec 要求 | Task |
|-----------|------|
| `detect_training_interface` 拒 pose／obb／polygon | 1–2 |
| 錯誤訊息中文統一 | 2（常數）、4（hint） |
| Jobs 白名單縮短 | 3 |
| prepare ValueError → 400 | 3 |
| Training 頁禁用兩按鈕 + reason | 4 |
| 測試改寫／保留三類通過 | 1–2、5 |
| 不改 Playground／不刪 trainer | 明確不動 |

無 TBD／placeholder。
