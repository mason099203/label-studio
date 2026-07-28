"""
Triton Model Upload Server

提供 REST API 讓使用者上傳資料或模型檔案到 Triton Server 的模型倉庫路徑。
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Triton Upload Server",
    description="上傳模型或資料到 Triton Inference Server 的模型倉庫",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Triton 模型倉庫掛載路徑（與 triton container 共享同一個 volume）
MODEL_REPO: Path = Path(os.getenv("MODEL_REPO", "/models"))


def _resolve_safe(base: Path, rel: str) -> Path:
    """
    解析目標路徑並確保不超出基礎目錄（防止路徑穿越攻擊）。
    @param {Path} base - 基礎目錄
    @param {str} rel - 相對路徑字串
    @returns {Path} 安全的絕對路徑
    @raises {HTTPException} 若路徑穿越基礎目錄則拋出 400
    """
    target = (base / rel).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="非法路徑：不允許超出模型倉庫目錄")
    return target


# ── 上傳端點 ────────────────────────────────────────────────────────────────────

@app.post(
    "/upload",
    summary="上傳單一檔案",
    description=(
        "將檔案上傳到 Triton 模型倉庫的指定子路徑。\n\n"
        "**sub_path 範例**：\n"
        "- `my_model/1/model.plan`\n"
        "- `my_model/config.pbtxt`\n\n"
        "若目錄不存在會自動建立。"
    ),
)
async def upload_file(
    file: UploadFile = File(..., description="要上傳的檔案"),
    sub_path: Optional[str] = Query(
        None,
        description="模型倉庫內的相對目標路徑（含檔名），留空則放在根目錄",
    ),
):
    """
    上傳單一檔案到 Triton 模型倉庫。
    @param {UploadFile} file - 上傳的檔案物件
    @param {Optional[str]} sub_path - 相對於模型倉庫根目錄的目標路徑
    @returns {JSONResponse} 儲存結果資訊
    """
    filename = file.filename or "upload"
    rel = sub_path if sub_path else filename
    dest: Path = _resolve_safe(MODEL_REPO, rel)

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    return JSONResponse(
        status_code=201,
        content={
            "message": "上傳成功",
            "saved_to": str(dest.relative_to(MODEL_REPO)),
            "size_bytes": dest.stat().st_size,
        },
    )


def _triton_model_filename(original_name: str) -> str:
    """
    將上傳的原始檔名轉換為 Triton 標準模型檔名。

    Triton 各 backend 對版本目錄內的檔名有固定要求：
    - pytorch_libtorch  → model.pt   (.pt / .pth / .torchscript)
    - onnxruntime_onnx  → model.onnx (.onnx)
    - tensorrt_plan     → model.plan (.plan / .engine)
    - 其他副檔名        → model.<ext>（保留副檔名，僅替換主檔名）

    @param {str} original_name - 使用者上傳的原始檔名
    @returns {str} 符合 Triton 規範的標準檔名
    """
    suffix = Path(original_name).suffix.lower()

    # PyTorch / TorchScript → model.pt
    if suffix in {".pt", ".pth", ".torchscript"}:
        return "model.pt"

    # ONNX → model.onnx
    if suffix == ".onnx":
        return "model.onnx"

    # TensorRT → model.plan
    if suffix in {".plan", ".engine"}:
        return "model.plan"

    # 其他副檔名：保留副檔名，主檔名改為 model
    return f"model{suffix}" if suffix else "model.bin"


@app.post(
    "/upload/model",
    summary="建立模型目錄結構並上傳檔案",
    description=(
        "自動在 `<model_name>/<version>/` 下建立目錄，並上傳模型檔案。\n\n"
        "檔名會自動依 Triton backend 規範重新命名（例如 `best.torchscript` → `model.pt`）。\n\n"
        "**適用情境**：上傳 ONNX / TensorRT plan / PyTorch 等模型主檔。"
    ),
)
async def upload_model_file(
    file: UploadFile = File(..., description="模型檔案"),
    model_name: str = Query(..., description="模型名稱（對應模型倉庫的子目錄名）"),
    version: int = Query(1, ge=1, description="模型版本號（正整數）"),
):
    """
    將模型檔案上傳到標準的 Triton 版本目錄結構。
    原始檔名會自動轉換為 Triton 要求的標準檔名（model.pt / model.onnx / model.plan）。
    @param {UploadFile} file - 模型檔案
    @param {str} model_name - Triton 模型名稱
    @param {int} version - 模型版本號
    @returns {JSONResponse} 儲存結果資訊（含原始檔名與轉換後檔名）
    """
    original_filename = file.filename or "model.bin"
    triton_filename = _triton_model_filename(original_filename)
    rel = f"{model_name}/{version}/{triton_filename}"
    dest: Path = _resolve_safe(MODEL_REPO, rel)

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    return JSONResponse(
        status_code=201,
        content={
            "message": "模型上傳成功",
            "model_name": model_name,
            "version": version,
            "original_filename": original_filename,
            "triton_filename": triton_filename,
            "saved_to": str(dest.relative_to(MODEL_REPO)),
            "size_bytes": dest.stat().st_size,
        },
    )


@app.post(
    "/upload/config",
    summary="上傳 config.pbtxt",
    description="將 `config.pbtxt` 上傳到指定模型的根目錄（`<model_name>/config.pbtxt`）。",
)
async def upload_config(
    file: UploadFile = File(..., description="config.pbtxt 檔案"),
    model_name: str = Query(..., description="目標模型名稱"),
):
    """
    上傳 Triton 模型設定檔 config.pbtxt。
    @param {UploadFile} file - config.pbtxt 檔案
    @param {str} model_name - Triton 模型名稱
    @returns {JSONResponse} 儲存結果資訊
    """
    rel = f"{model_name}/config.pbtxt"
    dest: Path = _resolve_safe(MODEL_REPO, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    return JSONResponse(
        status_code=201,
        content={
            "message": "config.pbtxt 上傳成功",
            "model_name": model_name,
            "saved_to": str(dest.relative_to(MODEL_REPO)),
        },
    )


# ── 查詢端點 ────────────────────────────────────────────────────────────────────

@app.get("/models", summary="列出所有模型目錄")
def list_models():
    """
    列出模型倉庫中所有頂層模型目錄名稱。
    @returns {dict} 包含模型名稱清單的字典
    """
    if not MODEL_REPO.exists():
        return {"models": []}

    models = [
        d.name
        for d in sorted(MODEL_REPO.iterdir())
        if d.is_dir()
    ]
    return {"models": models, "count": len(models)}


@app.get("/models/{model_name}", summary="列出模型版本與檔案")
def list_model_files(model_name: str):
    """
    列出指定模型目錄下的所有版本與檔案。
    @param {str} model_name - 模型名稱
    @returns {dict} 模型版本及檔案樹狀結構
    """
    model_dir = _resolve_safe(MODEL_REPO, model_name)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 不存在")

    def _walk(path: Path) -> dict:
        """遞迴建立目錄樹。"""
        if path.is_file():
            return {"type": "file", "size_bytes": path.stat().st_size}
        return {
            "type": "directory",
            "children": {
                child.name: _walk(child)
                for child in sorted(path.iterdir())
            },
        }

    return {"model_name": model_name, "tree": _walk(model_dir)}


# ── 刪除端點 ────────────────────────────────────────────────────────────────────

@app.delete("/models/{model_name}", summary="刪除整個模型目錄")
def delete_model(model_name: str):
    """
    刪除指定的模型目錄（包含所有版本與檔案）。
    @param {str} model_name - 要刪除的模型名稱
    @returns {dict} 操作結果訊息
    """
    model_dir = _resolve_safe(MODEL_REPO, model_name)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 不存在")

    shutil.rmtree(model_dir)
    return {"message": f"模型 '{model_name}' 已刪除"}


@app.delete("/models/{model_name}/{version}", summary="刪除特定版本")
def delete_model_version(model_name: str, version: int):
    """
    刪除模型的特定版本目錄。
    @param {str} model_name - 模型名稱
    @param {int} version - 版本號
    @returns {dict} 操作結果訊息
    """
    version_dir = _resolve_safe(MODEL_REPO, f"{model_name}/{version}")
    if not version_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"模型 '{model_name}' 版本 {version} 不存在",
        )

    shutil.rmtree(version_dir)
    return {"message": f"模型 '{model_name}' 版本 {version} 已刪除"}


# ── 健康檢查 ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="健康檢查")
def health():
    """
    確認服務與模型倉庫掛載狀態。
    @returns {dict} 服務狀態資訊
    """
    return {
        "status": "ok",
        "model_repo": str(MODEL_REPO),
        "model_repo_exists": MODEL_REPO.exists(),
    }
