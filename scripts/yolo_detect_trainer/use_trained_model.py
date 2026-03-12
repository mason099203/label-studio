"""
YOLO（Ultralytics）Detect 任務工具（CLI + 可選 UI）

此模組目標：
- 提供使用者選擇 Ultralytics YOLO detect 模型（預訓練或本機 .pt）。
- 完成 detect 類型任務：單張、批次、即時（攝影機）推論。
- 保存模型到本地：包含原始模型與訓練過模型（best/last）。
- 支援重新訓練：使用 base 權重與 data.yaml 重新訓練並保存新版本。
- 檢視模型效能：執行 val 取得 mAP 等指標並保存 JSON；可選 Streamlit UI 瀏覽。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Ultralytics Detect 模型選項（可擴充）
# ---------------------------------------------------------------------------

AVAILABLE_YOLO_DETECT_MODELS = [
    ("yolo11n.pt", "YOLO11 Nano（最小最快，適合即時）"),
    ("yolo11s.pt", "YOLO11 Small"),
    ("yolo11m.pt", "YOLO11 Medium"),
    ("yolo11l.pt", "YOLO11 Large"),
    ("yolo11x.pt", "YOLO11 Extra Large（精度最高）"),
    ("yolov8n.pt", "YOLOv8 Nano"),
    ("yolov8s.pt", "YOLOv8 Small"),
    ("yolov8m.pt", "YOLOv8 Medium"),
    ("yolov8l.pt", "YOLOv8 Large"),
    ("yolov8x.pt", "YOLOv8 Extra Large"),
]


def get_pretrained_model_ids() -> List[str]:
    """取得可選的預訓練 Detect 模型 ID 列表。"""

    return [m[0] for m in AVAILABLE_YOLO_DETECT_MODELS]


@dataclass(frozen=True)
class ModelInfo:
    """
    本地模型索引紀錄。

    Args:
        model_id: 識別字（預訓練名稱、檔名或 run_id）
        source: pretrained | local | trained
        weights_path: 權重檔案路徑（可能為 None，代表由 Ultralytics 快取管理）
        created_at: ISO 時間字串
        extra: 其他 metadata（例如 base、data.yaml、metrics）
    """

    model_id: str
    source: str
    weights_path: Optional[str]
    created_at: str
    extra: Dict[str, Any]


class LocalModelStore:
    """
    本地保存資料夾結構：
    - models/original/：base 權重（預訓練或本機）
    - models/trained/<run_id>/：訓練後 best/last 與 artifacts
    - models/metrics/：val 指標 JSON
    - models/index.json：索引
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.original_dir = self.root_dir / "original"
        self.trained_dir = self.root_dir / "trained"
        self.metrics_dir = self.root_dir / "metrics"
        self.index_path = self.root_dir / "index.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"models": []}
        with open(self.index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, data: Dict[str, Any]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(self, info: ModelInfo) -> None:
        data = self._load_index()
        data["models"] = data.get("models", [])
        data["models"].append(
            {
                "model_id": info.model_id,
                "source": info.source,
                "weights_path": info.weights_path,
                "created_at": info.created_at,
                "extra": info.extra,
            },
        )
        self._save_index(data)

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self._load_index().get("models", []))

    def save_original_weights(self, weights: str) -> Optional[Path]:
        """
        保存 base 權重到 `models/original/`。

        - 若 weights 是預訓練名稱，會觸發下載；但不保證能穩定找到快取檔案位置。
        - 若 weights 是本機路徑，直接複製。

        Returns:
            Path | None: 成功複製回傳檔案路徑；若是預訓練但找不到實體快取檔，回傳 None。
        """

        now = datetime.now().isoformat()
        is_pretrained = weights in get_pretrained_model_ids()

        if is_pretrained:
            model = YOLO(weights)

            resolved: Optional[Path] = None
            for attr in ("ckpt_path", "weights", "pt_path"):
                value = getattr(model, attr, None)
                if isinstance(value, (str, os.PathLike)) and Path(value).exists():
                    resolved = Path(value)
                    break

            if resolved is None:
                self.register(
                    ModelInfo(
                        model_id=weights,
                        source="pretrained",
                        weights_path=None,
                        created_at=now,
                        extra={"note": "預訓練模型由 Ultralytics 快取管理；此處未複製出實體檔案。"},
                    ),
                )
                return None

            target = self.original_dir / resolved.name
            if not target.exists():
                shutil.copy2(resolved, target)

            self.register(
                ModelInfo(
                    model_id=weights,
                    source="pretrained",
                    weights_path=str(target),
                    created_at=now,
                    extra={"resolved_from": str(resolved)},
                ),
            )
            return target

        src = Path(weights)
        if not src.exists():
            raise FileNotFoundError(f"找不到權重檔: {weights}")
        target = self.original_dir / src.name
        if not target.exists():
            shutil.copy2(src, target)

        self.register(
            ModelInfo(
                model_id=src.name,
                source="local",
                weights_path=str(target),
                created_at=now,
                extra={"original_path": str(src)},
            ),
        )
        return target

    def save_trained_run(self, run_dir: Path, base_weights: str, data_yaml: str) -> Dict[str, Any]:
        """
        保存訓練結果（best/last 與常見圖表檔）到 `models/trained/<run_id>/`。
        """

        now = datetime.now().isoformat()
        run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        target_dir = self.trained_dir / run_id
        target_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = run_dir / "weights"
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"

        saved: Dict[str, Any] = {
            "run_id": run_id,
            "created_at": now,
            "base_weights": base_weights,
            "data_yaml": data_yaml,
            "source_run_dir": str(run_dir),
            "best_path": None,
            "last_path": None,
            "artifacts": [],
        }

        if best.exists():
            dst = target_dir / "best.pt"
            shutil.copy2(best, dst)
            saved["best_path"] = str(dst)
        if last.exists():
            dst = target_dir / "last.pt"
            shutil.copy2(last, dst)
            saved["last_path"] = str(dst)

        for name in [
            "args.yaml",
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
        ]:
            src = run_dir / name
            if src.exists():
                dst = target_dir / name
                shutil.copy2(src, dst)
                saved["artifacts"].append({"name": name, "path": str(dst)})

        self.register(
            ModelInfo(
                model_id=run_id,
                source="trained",
                weights_path=saved["best_path"] or saved["last_path"],
                created_at=now,
                extra=saved,
            ),
        )
        return saved

    def save_metrics(self, model_id: str, metrics: Dict[str, Any]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.metrics_dir / f"{model_id}_val_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return path


class YOLOPredictor:
    """YOLO Detect 推論器（單張 / 批次 / 即時）。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.class_names = list(self.model.names.values()) if getattr(self.model, "names", None) else []

    def predict_single_image(self, image_path: str, conf_threshold: float = 0.25, save_result: bool = True) -> Dict[str, Any]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"找不到圖片檔案: {image_path}")

        results = self.model.predict(
            source=image_path,
            conf=float(conf_threshold),
            save=bool(save_result),
            project="runs/predict",
            name="single_prediction",
            show_labels=True,
            show_conf=True,
            verbose=False,
        )

        r = results[0]
        preds: List[Dict[str, Any]] = []
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)
                preds.append(
                    {
                        "class_id": cls,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    },
                )

        return {"image_path": image_path, "predictions": preds, "total_objects": len(preds)}

    def predict_batch(self, image_folder: str, conf_threshold: float = 0.25, save_result: bool = True) -> List[Dict[str, Any]]:
        folder = Path(image_folder)
        if not folder.exists():
            raise FileNotFoundError(f"找不到圖片資料夾: {image_folder}")

        results = self.model.predict(
            source=str(folder),
            conf=float(conf_threshold),
            save=bool(save_result),
            project="runs/predict",
            name="batch_prediction",
            show_labels=True,
            show_conf=True,
            verbose=False,
        )

        all_res: List[Dict[str, Any]] = []
        for r in results:
            preds: List[Dict[str, Any]] = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)
                    preds.append(
                        {
                            "class_id": cls,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        },
                    )
            all_res.append({"image_path": str(r.path), "predictions": preds, "total_objects": len(preds)})
        return all_res

    def predict_realtime(self, source: int = 0, conf_threshold: float = 0.25) -> None:
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError("無法開啟攝影機")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                results = self.model.predict(source=frame, conf=float(conf_threshold), verbose=False)
                annotated = results[0].plot()
                cv2.imshow("YOLO Realtime Detect", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def train_detect_model(
    base_weights: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "train",
) -> Path:
    """使用 Ultralytics YOLO 訓練 detect 模型，回傳訓練輸出資料夾。"""

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml}")

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        project=str(project),
        name=str(name),
    )
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    return Path(save_dir) if save_dir else Path(project) / name


def evaluate_detect_model(model_weights: str, data_yaml: str, imgsz: int = 640) -> Dict[str, Any]:
    """執行 val 取得常用指標（mAP 等），回傳 dict。"""

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml}")

    model = YOLO(model_weights)
    res = model.val(data=str(data_yaml), imgsz=int(imgsz))

    metrics: Dict[str, Any] = {
        "model": model_weights,
        "data": str(data_yaml),
        "imgsz": int(imgsz),
        "timestamp": datetime.now().isoformat(),
    }

    box = getattr(res, "box", None) or getattr(res, "metrics", None)
    for key in ("map50", "map", "map75", "mp", "mr"):
        val = getattr(box, key, None) if box is not None else getattr(res, key, None)
        if val is None:
            continue
        try:
            metrics[key] = float(val)
        except Exception:
            metrics[key] = val

    maps = getattr(box, "maps", None) if box is not None else None
    if maps is not None:
        try:
            metrics["maps_per_class"] = [float(x) for x in maps]
        except Exception:
            metrics["maps_per_class"] = maps

    return metrics


def run_streamlit_ui(store: LocalModelStore) -> None:
    """啟動 Streamlit UI（需先安裝 streamlit）。"""

    try:
        import streamlit as st  # type: ignore
    except Exception as e:
        raise RuntimeError("尚未安裝 streamlit。請先執行：pip install streamlit") from e

    st.set_page_config(page_title="YOLO Detect Trainer", layout="wide")
    st.title("YOLO Detect 訓練 / 推論 / 評估（範例）")

    page = st.sidebar.radio("功能", ["Models", "Detect", "Train", "Evaluate"])

    if page == "Models":
        st.subheader("本地模型清單（index.json）")
        st.dataframe(store.list_models(), use_container_width=True)

        st.divider()
        st.subheader("保存原始模型（base weights）")
        base = st.selectbox("預訓練模型", options=get_pretrained_model_ids())
        if st.button("保存 base 到 models/original"):
            saved = store.save_original_weights(base)
            st.success(f"完成：{saved if saved else '已建立索引（快取由 Ultralytics 管理）'}")

    elif page == "Detect":
        st.subheader("Detect 推論")
        weights = st.selectbox("模型（預訓練）", options=get_pretrained_model_ids())
        conf = st.slider("conf", 0.0, 1.0, 0.25, 0.01)
        mode = st.radio("模式", ["single", "batch"])
        if mode == "single":
            img = st.text_input("圖片路徑")
            if st.button("推論") and img:
                pred = YOLOPredictor(weights).predict_single_image(img, conf_threshold=float(conf), save_result=True)
                st.json(pred)
        else:
            folder = st.text_input("資料夾路徑")
            if st.button("批次推論") and folder:
                res = YOLOPredictor(weights).predict_batch(folder, conf_threshold=float(conf), save_result=True)
                st.write(f"完成：{len(res)} 張")

    elif page == "Train":
        st.subheader("訓練 / 重新訓練")
        base = st.selectbox("Base（預訓練）", options=get_pretrained_model_ids())
        data_yaml = st.text_input("data.yaml 路徑")
        epochs = st.number_input("epochs", 1, 500, 50)
        imgsz = st.number_input("imgsz", 32, 2048, 640, step=32)
        batch = st.number_input("batch", 1, 256, 16)
        name = st.text_input("run name", "train")
        if st.button("開始訓練"):
            store.save_original_weights(base)
            run_dir = train_detect_model(base, data_yaml, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), name=name)
            saved = store.save_trained_run(run_dir, base, data_yaml)
            st.success("訓練完成並已保存")
            st.json(saved)

    else:
        st.subheader("評估（val）")
        weights = st.selectbox("模型（預訓練）", options=get_pretrained_model_ids())
        data_yaml = st.text_input("data.yaml 路徑")
        imgsz = st.number_input("imgsz", 32, 2048, 640, step=32)
        if st.button("開始評估"):
            metrics = evaluate_detect_model(weights, data_yaml, imgsz=int(imgsz))
            out = store.save_metrics(Path(weights).stem, metrics)
            st.success(f"已保存：{out}")
            st.json(metrics)

        st.divider()
        st.subheader("歷史評估 JSON")
        items = sorted(store.metrics_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if items:
            pick = st.selectbox("選擇檔案", options=[str(p) for p in items])
            with open(pick, "r", encoding="utf-8") as f:
                st.json(json.load(f))
        else:
            st.info("尚無評估紀錄。")


def build_arg_parser() -> argparse.ArgumentParser:
    """建立 CLI 參數。"""

    parser = argparse.ArgumentParser(description="YOLO Detect：推論 / 訓練 / 評估 / 模型保存（範例）")
    sub = parser.add_subparsers(dest="cmd")

    p_detect = sub.add_parser("detect", help="推論（單張/批次/即時）")
    p_detect.add_argument("--model", type=str, default="yolo11n.pt", help="模型路徑或預訓練名稱")
    p_detect.add_argument("--mode", type=str, choices=["single", "batch", "realtime"], default="single")
    p_detect.add_argument("--source", type=str, help="圖片路徑、資料夾路徑或攝影機索引（realtime）")
    p_detect.add_argument("--conf", type=float, default=0.25)
    p_detect.add_argument("--save-original", action="store_true", help="嘗試保存 base 到 models/original")
    p_detect.add_argument("--no-save", action="store_true", help="不保存 YOLO 輸出圖")

    p_train = sub.add_parser("train", help="訓練/重新訓練 detect 模型")
    p_train.add_argument("--base", type=str, default="yolo11n.pt", help="base 權重（預訓練名稱或本機 .pt）")
    p_train.add_argument("--data", type=str, required=True, help="data.yaml 路徑")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--name", type=str, default="train")

    p_eval = sub.add_parser("eval", help="評估（val）並保存指標 JSON")
    p_eval.add_argument("--model", type=str, required=True, help="模型權重路徑或預訓練名稱")
    p_eval.add_argument("--data", type=str, required=True, help="data.yaml 路徑")
    p_eval.add_argument("--imgsz", type=int, default=640)

    sub.add_parser("ui", help="啟動 UI（需要 streamlit）")
    return parser


def main() -> None:
    store = LocalModelStore(Path(__file__).parent / "models")
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "ui":
        run_streamlit_ui(store)
        return

    if args.cmd == "train":
        store.save_original_weights(args.base)
        run_dir = train_detect_model(
            base_weights=args.base,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
        )
        saved = store.save_trained_run(run_dir, args.base, args.data)
        print(json.dumps(saved, ensure_ascii=False, indent=2))
        return

    if args.cmd == "eval":
        metrics = evaluate_detect_model(args.model, args.data, imgsz=args.imgsz)
        out = store.save_metrics(Path(args.model).stem, metrics)
        print(f"metrics_saved_to={out}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    # 預設 detect
    if args.cmd in (None, "detect"):
        predictor = YOLOPredictor(args.model)
        if args.save_original:
            store.save_original_weights(args.model)

        if args.mode == "single":
            if not args.source:
                raise ValueError("single 模式需要 --source（圖片路徑）")
            res = predictor.predict_single_image(args.source, conf_threshold=args.conf, save_result=not args.no_save)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return

        if args.mode == "batch":
            if not args.source:
                raise ValueError("batch 模式需要 --source（資料夾路徑）")
            res = predictor.predict_batch(args.source, conf_threshold=args.conf, save_result=not args.no_save)
            print(json.dumps({"total_images": len(res)}, ensure_ascii=False, indent=2))
            return

        if args.mode == "realtime":
            cam = int(args.source) if args.source is not None else 0
            predictor.predict_realtime(source=cam, conf_threshold=args.conf)
            return

    parser.print_help()


if __name__ == "__main__":
    main()

