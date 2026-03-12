# YOLO Detect Trainer（Ultralytics）— 範例模組

這是一個 **獨立的 Python 範例**，放在 Label Studio repo 內方便你後續接到產品流程（例如：從 Export 拿到標註資料、訓練、評估、保存模型供部署）。

## 目標功能

- **模型選擇**：提供 Ultralytics YOLO Detect 預訓練模型清單（YOLO11/YOLOv8），也支援本機 `.pt`
- **Detect 推論**：單張 / 批次資料夾 / 即時攝影機
- **保存模型到本地**
  - `models/original/`：原始（base）模型（預訓練或本機）
  - `models/trained/<run_id>/`：訓練後模型（best/last）與訓練產物
  - `models/metrics/`：評估指標（val）JSON
- **重新訓練**：指定 base 權重 + `data.yaml` 重跑訓練
- **檢視效能/評估頁**：可選用 Streamlit（可視化頁面），或只用 CLI 產出 JSON

## 安裝

請先確保你有安裝：

- `ultralytics`
- `opencv-python`

可選（要 UI 才需要）：

- `streamlit`

## 使用方式（CLI）

在 repo root 執行：

```bash
python scripts/yolo_detect_trainer/use_trained_model.py --help
```

### 1) Detect 推論

```bash
python scripts/yolo_detect_trainer/use_trained_model.py detect --model yolo11n.pt --mode single --source path/to/image.jpg
python scripts/yolo_detect_trainer/use_trained_model.py detect --model yolo11n.pt --mode batch --source path/to/images_folder
python scripts/yolo_detect_trainer/use_trained_model.py detect --model yolo11n.pt --mode realtime --source 0
```

### 2) 訓練 / 重新訓練

```bash
python scripts/yolo_detect_trainer/use_trained_model.py train --base yolo11n.pt --data path/to/data.yaml --epochs 50 --imgsz 640 --batch 16
```

完成後會把訓練輸出保存到 `scripts/yolo_detect_trainer/models/trained/<run_id>/`。

### 3) 評估（val）與保存指標

```bash
python scripts/yolo_detect_trainer/use_trained_model.py eval --model path/to/best.pt --data path/to/data.yaml --imgsz 640
```

## 使用方式（UI）

```bash
streamlit run scripts/yolo_detect_trainer/use_trained_model.py -- ui
```

