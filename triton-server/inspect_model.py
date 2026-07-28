"""
inspect_model.py ── 讀取並分析 PyTorch / TorchScript / ONNX 模型的參數與輸入輸出格式

支援三種模型格式：
  1. TorchScript (.pt) ─ torch.jit.load()   ← Triton 常用
  2. PyTorch 權重 (.pt / .pth) ─ torch.load()  ← 訓練後的 state_dict
  3. ONNX (.onnx) ─ onnx + onnxruntime    ← 轉換後的推論格式

用法：
  python inspect_model.py --model path/to/model.pt
  python inspect_model.py --model path/to/model.onnx --type onnx
  python inspect_model.py --model path/to/model.pt --dummy-input "1,3,640,640" --run

選項：
  --model         模型檔案路徑（必填）
  --type          強制指定格式：torchscript | weights | onnx（預設自動偵測）
  --dummy-input   虛擬輸入尺寸，格式 B,C,H,W，用於推論測試（預設 1,3,640,640）
  --run           執行一次推論並印出輸出 shape
  --device        cpu 或 cuda（預設自動偵測）
  --netron        使用 Netron 開啟視覺化介面（需安裝 netron）
"""

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────────────────────

def _bytes_to_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MB"


def _dtype_str(dtype) -> str:
    """將 torch.dtype 轉成易讀字串"""
    return str(dtype).replace("torch.", "")


def _print_section(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. TorchScript 模型檢測
# ──────────────────────────────────────────────────────────────────────────────

def inspect_torchscript(model_path: Path, device: str, dummy_shape: tuple, run: bool) -> None:
    """
    @param {Path} model_path - TorchScript 模型路徑
    @param {str} device - 執行裝置 (cpu / cuda)
    @param {tuple} dummy_shape - 虛擬輸入尺寸 (B, C, H, W)
    @param {bool} run - 是否執行推論
    """
    import torch

    _print_section("TorchScript 模型分析")
    print(f"  路徑：{model_path}")
    print(f"  大小：{_bytes_to_mb(model_path.stat().st_size)}")
    print(f"  裝置：{device}")

    print("\n[載入模型...]")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    print("  ✓ 載入成功")

    # ── 圖形結構 ───────────────────────────────────────────────────────────────
    _print_section("計算圖（Graph）")
    try:
        graph = model.graph
        # 輸出精簡版圖形（前 60 行）
        graph_str = str(graph)
        lines = graph_str.splitlines()
        for i, line in enumerate(lines[:60]):
            print(f"  {line}")
        if len(lines) > 60:
            print(f"  ... （共 {len(lines)} 行，已截斷）")
    except Exception as e:
        print(f"  [警告] 無法取得圖形：{e}")

    # ── 輸入節點 ───────────────────────────────────────────────────────────────
    _print_section("輸入節點（Inputs）")
    try:
        inputs = list(model.graph.inputs())
        # 跳過第一個 self 節點
        data_inputs = [i for i in inputs if i.debugName() != "self"]
        if not data_inputs:
            data_inputs = inputs[1:]  # fallback
        for node in data_inputs:
            t = node.type()
            print(f"  名稱：{node.debugName()}")
            print(f"  型別：{t}")
            try:
                print(f"  Shape：{t.sizes()}")
                print(f"  Dtype：{t.scalarType()}")
            except Exception:
                pass
            print()
    except Exception as e:
        print(f"  [警告] 無法解析輸入：{e}")

    # ── 輸出節點 ───────────────────────────────────────────────────────────────
    _print_section("輸出節點（Outputs）")
    try:
        outputs = list(model.graph.outputs())
        for node in outputs:
            t = node.type()
            print(f"  名稱：{node.debugName()}")
            print(f"  型別：{t}")
            try:
                print(f"  Shape：{t.sizes()}")
            except Exception:
                pass
            print()
    except Exception as e:
        print(f"  [警告] 無法解析輸出：{e}")

    # ── 模型內嵌屬性（metadata） ───────────────────────────────────────────────
    _print_section("模型內嵌屬性（Extra Files / Metadata）")
    try:
        extra_files = {"config.txt": "", "meta.json": ""}
        torch.jit.load(str(model_path), map_location="cpu", _extra_files=extra_files)
        for k, v in extra_files.items():
            if v:
                print(f"  [{k}]\n  {v[:500]}\n")
    except Exception as e:
        print(f"  [提示] 無內嵌附加檔案（{e}）")

    # ── 實際推論 ───────────────────────────────────────────────────────────────
    if run:
        _print_section("推論測試")
        import torch
        dummy = torch.zeros(*dummy_shape, dtype=torch.float32).to(device)
        print(f"  虛擬輸入 shape：{list(dummy.shape)}, dtype：{_dtype_str(dummy.dtype)}")
        with torch.no_grad():
            outputs = model(dummy)
        # 處理單一或多輸出
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        elif isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
        else:
            outputs = [outputs]
        print(f"  輸出數量：{len(outputs)}")
        for idx, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"  [輸出 {idx}] shape={list(out.shape)}, dtype={_dtype_str(out.dtype)}, "
                      f"min={out.min().item():.4f}, max={out.max().item():.4f}")
            else:
                print(f"  [輸出 {idx}] type={type(out)}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. PyTorch 權重（state_dict）檢測
# ──────────────────────────────────────────────────────────────────────────────

def inspect_weights(model_path: Path) -> None:
    """
    @param {Path} model_path - PyTorch 權重檔路徑
    """
    import torch

    _print_section("PyTorch 權重（state_dict）分析")
    print(f"  路徑：{model_path}")
    print(f"  大小：{_bytes_to_mb(model_path.stat().st_size)}")

    print("\n[載入權重...]")
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    print(f"  物件型別：{type(ckpt)}")

    # ── 判斷是否為 dict（checkpoint / state_dict） ─────────────────────────────
    if isinstance(ckpt, dict):
        print(f"\n  Dict 鍵值：{list(ckpt.keys())[:20]}")

        # 取得 state_dict
        state = None
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in ckpt:
                candidate = ckpt[key]
                if hasattr(candidate, "state_dict"):
                    state = candidate.state_dict()
                elif isinstance(candidate, dict):
                    state = candidate
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt

        if state:
            _print_section("層參數（Layer Parameters）")
            total_params = 0
            for name, tensor in list(state.items())[:50]:
                params = tensor.numel()
                total_params += params
                print(f"  {name:<50s} shape={list(tensor.shape)}, "
                      f"dtype={_dtype_str(tensor.dtype)}, params={params:,}")
            if len(state) > 50:
                print(f"  ... （共 {len(state)} 層，已截斷前 50 層）")
            print(f"\n  ✓ 總參數量：{total_params:,} ({total_params / 1e6:.2f}M)")

        # ── 其他 metadata ──────────────────────────────────────────────────────
        for key, val in ckpt.items():
            if not isinstance(val, (dict, torch.Tensor)):
                print(f"  [{key}] = {val}")
    else:
        # 可能是直接的 nn.Module
        if hasattr(ckpt, "parameters"):
            total = sum(p.numel() for p in ckpt.parameters())
            trainable = sum(p.numel() for p in ckpt.parameters() if p.requires_grad)
            print(f"\n  總參數量：{total:,} ({total / 1e6:.2f}M)")
            print(f"  可訓練參數：{trainable:,}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ONNX 模型檢測
# ──────────────────────────────────────────────────────────────────────────────

def inspect_onnx(model_path: Path, run: bool, dummy_shape: tuple) -> None:
    """
    @param {Path} model_path - ONNX 模型路徑
    @param {bool} run - 是否執行推論
    @param {tuple} dummy_shape - 虛擬輸入尺寸
    """
    try:
        import onnx
    except ImportError:
        print("  [錯誤] 請先安裝 onnx：pip install onnx")
        return

    _print_section("ONNX 模型分析")
    print(f"  路徑：{model_path}")
    print(f"  大小：{_bytes_to_mb(model_path.stat().st_size)}")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    print("  ✓ 模型驗證通過")

    # ── 基本資訊 ───────────────────────────────────────────────────────────────
    _print_section("基本資訊（Metadata）")
    print(f"  IR version  : {model.ir_version}")
    print(f"  Opset       : {[op.version for op in model.opset_import]}")
    print(f"  Producer    : {model.producer_name} {model.producer_version}")
    print(f"  Domain      : {model.domain}")
    print(f"  Model ver.  : {model.model_version}")
    if model.doc_string:
        print(f"  Doc         : {model.doc_string}")

    # ── 輸入張量 ───────────────────────────────────────────────────────────────
    _print_section("輸入張量（Inputs）")
    _onnx_dtype_map = {
        1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
        6: "int32",   7: "int64", 8: "string", 9: "bool",
        10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
    }
    for inp in model.graph.input:
        t = inp.type.tensor_type
        dtype = _onnx_dtype_map.get(t.elem_type, f"type_{t.elem_type}")
        dims = []
        for d in t.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        print(f"  名稱：{inp.name}")
        print(f"  Dtype：{dtype}")
        print(f"  Shape：{dims}")
        print()

    # ── 輸出張量 ───────────────────────────────────────────────────────────────
    _print_section("輸出張量（Outputs）")
    for out in model.graph.output:
        t = out.type.tensor_type
        dtype = _onnx_dtype_map.get(t.elem_type, f"type_{t.elem_type}")
        dims = []
        for d in t.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        print(f"  名稱：{out.name}")
        print(f"  Dtype：{dtype}")
        print(f"  Shape：{dims}")
        print()

    # ── 中間節點統計 ───────────────────────────────────────────────────────────
    _print_section("算子統計（Op Types）")
    from collections import Counter
    op_counts = Counter(node.op_type for node in model.graph.node)
    for op, count in op_counts.most_common(20):
        print(f"  {op:<30s} × {count}")
    print(f"\n  ✓ 總節點數：{len(model.graph.node)}")

    # ── OnnxRuntime 推論 ───────────────────────────────────────────────────────
    if run:
        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError:
            print("\n  [警告] 請安裝 onnxruntime 以執行推論：pip install onnxruntime-gpu")
            return

        _print_section("推論測試（OnnxRuntime）")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(model_path), providers=providers)
        active = sess.get_providers()[0]
        print(f"  執行後端：{active}")

        # 建立虛擬輸入
        input_meta = sess.get_inputs()
        feeds = {}
        for meta in input_meta:
            shape = [dummy_shape[i] if i < len(dummy_shape) else 1
                     for i, d in enumerate(meta.shape)]
            # 使用 meta.shape 中的靜態維度（若非動態）
            resolved = []
            for i, d in enumerate(meta.shape):
                if isinstance(d, int) and d > 0:
                    resolved.append(d)
                elif i < len(dummy_shape):
                    resolved.append(dummy_shape[i])
                else:
                    resolved.append(1)
            arr = np.zeros(resolved, dtype=np.float32)
            feeds[meta.name] = arr
            print(f"  輸入 [{meta.name}]：shape={resolved}, dtype=float32")

        results = sess.run(None, feeds)
        print(f"\n  輸出數量：{len(results)}")
        output_meta = sess.get_outputs()
        for i, (meta, arr) in enumerate(zip(output_meta, results)):
            print(f"  [輸出 {i}] 名稱={meta.name}, shape={list(arr.shape)}, "
                  f"dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. 自動偵測格式
# ──────────────────────────────────────────────────────────────────────────────

def auto_detect_type(model_path: Path) -> str:
    """
    自動判斷 .pt 是 TorchScript 還是 state_dict

    @param {Path} model_path - 模型檔案路徑
    @returns {str} 格式類型：torchscript | weights | onnx
    """
    if model_path.suffix == ".onnx":
        return "onnx"

    # 嘗試以 TorchScript 載入
    try:
        import torch
        torch.jit.load(str(model_path), map_location="cpu")
        return "torchscript"
    except Exception:
        pass

    return "weights"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Netron 視覺化
# ──────────────────────────────────────────────────────────────────────────────

def open_netron(model_path: Path) -> None:
    """
    @param {Path} model_path - 模型檔案路徑
    """
    try:
        import netron
        print(f"\n[Netron] 開啟視覺化介面：{model_path}")
        netron.start(str(model_path))
    except ImportError:
        print("\n  [提示] 請先安裝 netron：pip install netron")
        print("  安裝後執行：python -c \"import netron; netron.start('model.pt')\"")


# ──────────────────────────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 PyTorch / TorchScript / ONNX 模型的參數與輸入輸出格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",       required=True,          help="模型檔案路徑")
    parser.add_argument("--type",        default="auto",
                        choices=["auto", "torchscript", "weights", "onnx"],
                        help="強制指定格式（預設自動偵測）")
    parser.add_argument("--dummy-input", default="1,3,640,640",  help="虛擬輸入尺寸 B,C,H,W")
    parser.add_argument("--run",         action="store_true",    help="執行推論並顯示輸出 shape")
    parser.add_argument("--device",      default="auto",         help="cpu 或 cuda（預設自動偵測）")
    parser.add_argument("--netron",      action="store_true",    help="使用 Netron 視覺化")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[錯誤] 找不到模型檔案：{model_path}")
        sys.exit(1)

    # 解析虛擬輸入尺寸
    dummy_shape = tuple(int(x) for x in args.dummy_input.split(","))

    # 裝置選擇
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    # 自動偵測格式
    fmt = args.type if args.type != "auto" else auto_detect_type(model_path)
    print(f"\n偵測格式：{fmt.upper()}")

    if args.netron:
        open_netron(model_path)
        return

    if fmt == "torchscript":
        inspect_torchscript(model_path, device, dummy_shape, args.run)
    elif fmt == "weights":
        inspect_weights(model_path)
    elif fmt == "onnx":
        inspect_onnx(model_path, args.run, dummy_shape)
    else:
        print(f"[錯誤] 不支援的格式：{fmt}")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print("  ✓ 分析完成")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
