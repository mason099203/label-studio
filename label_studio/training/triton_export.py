"""
Utilities for exporting training artifacts into a Triton model repository.

The deployment flow converts a trained YOLO checkpoint into a Triton-loadable
TorchScript model, writes the Triton config files, and stores a small metadata
file so the frontend Playground can discover project-specific deployments.

Remote deployment flow
----------------------
When the target Triton is on a different host (e.g. http://10.214.57.20:18000),
the Django backend cannot write directly to its model repository.  Instead the
exported files are HTTP-POSTed to the companion ``upload_server`` service
(same host, port ``TRITON_UPLOAD_SERVER_PORT``, default 8003), which shares the
``triton_models`` Docker volume with Triton.

  Django ──export──► temp model.pt + config.pbtxt
                            │  HTTP POST
                            ▼
                   upload_server :8003
                            │  shared volume
                            ▼
                   Triton /models ← model loaded
"""

from __future__ import annotations

import logging
import os
import json
import re
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests
import shutil

logger = logging.getLogger(__name__)

# Upload Server 對外埠號（與 docker-compose upload service 一致）
TRITON_UPLOAD_SERVER_PORT: int = int(os.environ.get("TRITON_UPLOAD_SERVER_PORT", "8003"))


def get_triton_model_repository_root() -> Path:
    """
    Root path for Triton model repository.
    Uses TRITON_MODEL_REPOSITORY env, else default under repo data/triton_models.
    """
    env_path = os.environ.get("TRITON_MODEL_REPOSITORY")
    if env_path:
        return Path(env_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "triton_models"


def get_triton_server_url() -> str:
    """
    Base HTTP URL for Triton.
    """
    return os.environ.get("TRITON_SERVER_URL", "http://localhost:8000").rstrip("/")


def _is_remote_triton(triton_url: str) -> bool:
    """
    判斷 Triton URL 是否指向遠端主機（非 localhost / 127.0.0.1）。

    本機部署時 Django 可直接寫入共享磁碟；遠端時需透過 Upload Server 轉送。

    @param {str} triton_url - Triton HTTP 基底 URL
    @returns {bool} 是否為遠端主機
    """
    try:
        host = urlparse(triton_url).hostname or ""
        return host not in ("localhost", "127.0.0.1", "::1", "")
    except Exception:
        return False


def derive_upload_server_url(triton_url: str) -> str:
    """
    從 Triton 基底 URL 推導 Upload Server URL（同主機、埠 TRITON_UPLOAD_SERVER_PORT）。

    @example
    derive_upload_server_url("http://10.214.57.20:18000") → "http://10.214.57.20:8003"

    @param {str} triton_url - Triton HTTP 基底 URL
    @returns {str} Upload Server 基底 URL
    """
    parsed = urlparse(triton_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    return f"{scheme}://{host}:{TRITON_UPLOAD_SERVER_PORT}"


def upload_model_to_remote_server(
    upload_base_url: str,
    model_name: str,
    model_pt_path: Path,
    config_content: str,
    timeout: float = 60.0,
    version: int = 1,
) -> dict:
    """
    透過 Upload Server REST API 將模型檔案與設定檔上傳到遠端 Triton 模型倉庫。

    上傳路徑（對應 upload_server.py 端點）：
    - ``POST {upload_base_url}/upload/model?model_name={model_name}&version={version}``
    - ``POST {upload_base_url}/upload/config?model_name={model_name}``

    @param {str} upload_base_url - Upload Server 基底 URL，例如 ``http://10.214.57.20:8003``
    @param {str} model_name      - Triton 模型名稱
    @param {Path} model_pt_path  - 本機已匯出的 model.pt 路徑
    @param {str} config_content  - config.pbtxt 文字內容
    @param {float} timeout       - HTTP 逾時秒數（預設 60）
    @param {int} version         - Triton 版本號，對應倉庫中的版本子目錄（1/ 2/ ...），預設為 1
    @returns {dict} 上傳結果；包含 ``error`` 鍵時表示失敗
    """
    base = upload_base_url.rstrip("/")

    # ── 上傳 model.pt ───────────────────────────────────────────────────────
    try:
        with model_pt_path.open("rb") as f:
            resp = requests.post(
                f"{base}/upload/model",
                params={"model_name": model_name, "version": version},
                files={"file": (model_pt_path.name, f, "application/octet-stream")},
                timeout=timeout,
            )
        if not resp.ok:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
            return {"error": f"Upload Server 拒絕模型檔案（{resp.status_code}）：{detail}"}
        model_result = resp.json()
    except requests.RequestException as exc:
        return {"error": f"無法連線至 Upload Server（{base}）：{exc}"}

    # ── 上傳 config.pbtxt ────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{base}/upload/config",
            params={"model_name": model_name},
            files={"file": ("config.pbtxt", config_content.encode("utf-8"), "text/plain")},
            timeout=timeout,
        )
        if not resp.ok:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
            return {"error": f"Upload Server 拒絕 config.pbtxt（{resp.status_code}）：{detail}"}
        config_result = resp.json()
    except requests.RequestException as exc:
        return {"error": f"上傳 config.pbtxt 失敗：{exc}"}

    return {
        "upload_server_url": base,
        "model_saved_to": model_result.get("saved_to"),
        "config_saved_to": config_result.get("saved_to"),
    }


def sanitize_triton_model_name(raw_name: str, fallback: str = "triton_model") -> str:
    """
    Convert an arbitrary string into a Triton-safe model name.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", (raw_name or "").strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or fallback


def _build_triton_pbtxt(
    model_name: str,
    imgsz: int,
    instance_kind: str = "AUTO",
    gpu_ids: Optional[list[int]] = None,
    instance_count: int = 1,
    always_in_memory: bool = True,
) -> str:
    """
    Build the Triton config.pbtxt content for a model deployment.

    @param {str} model_name         - Triton 模型名稱
    @param {int} imgsz              - 輸入圖片尺寸（正方形）
    @param {str} instance_kind      - 推論裝置："GPU" | "CPU" | "AUTO"（預設 AUTO）
    @param {Optional[list[int]]} gpu_ids - GPU 裝置 ID 清單，僅 instance_kind=="GPU" 時有效
    @param {int} instance_count     - 要建立的推論實例數量（預設 1）
    @param {bool} always_in_memory  - True：加入 model_warmup 預熱區塊，確保模型啟動時即載入並常駐記憶體；
                                      False：不預熱，模型於首次推論請求時才完整初始化（即時載入）
    @returns {str} 格式化後的 config.pbtxt 字串
    """
    # ── instance_group（裝置 / 實例數量）──────────────────────────────────
    _kind_map = {"GPU": "KIND_GPU", "CPU": "KIND_CPU", "AUTO": "KIND_AUTO"}
    kind_str = _kind_map.get((instance_kind or "AUTO").upper(), "KIND_AUTO")

    gpu_line = ""
    if kind_str == "KIND_GPU" and gpu_ids:
        gpu_list_str = ", ".join(str(g) for g in gpu_ids)
        gpu_line = f"\n    gpus: [{gpu_list_str}]"

    instance_group_block = f"""
instance_group [
  {{
    kind: {kind_str}
    count: {max(1, instance_count)}{gpu_line}
  }}
]"""

    # ── model_warmup（常駐記憶體）────────────────────────────────────────
    warmup_block = ""
    if always_in_memory:
        warmup_block = f"""
model_warmup [
  {{
    name: "warmup"
    batch_size: 1
    inputs {{
      key: "images"
      value {{
        dims: 3
        dims: {imgsz}
        dims: {imgsz}
        data_type: TYPE_FP32
        zero_data: true
      }}
    }}
  }}
]"""

    output_dims = "[-1, -1]" if "yolo" in model_name.lower() else "[-1]"

    return f'''name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 1

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, {imgsz}, {imgsz}]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]
{instance_group_block}
{warmup_block}
'''


def export_torchscript_pt_to_triton(
    best_pt_path: str | Path,
    model_name: str,
    project_id: Optional[int] = None,
    run_id: Optional[str] = None,
    triton_repo_root: Optional[str | Path] = None,
    imgsz: int = 224,
    deployed_by_user_id: Optional[int] = None,
    deployed_by_username: Optional[str] = None,
    public_triton_base_url: Optional[str] = None,
    upload_server_url: Optional[str] = None,
    instance_kind: str = "AUTO",
    gpu_ids: Optional[list[int]] = None,
    instance_count: int = 1,
    always_in_memory: bool = True,
    export_device: Optional[str] = None,
    target_version: int = 1,
    custom_pbtxt: Optional[str] = None,
) -> dict:
    """
    Export a general TorchScript model (.pt) into Triton's libtorch layout.

    若 ``upload_server_url`` 不為空，匯出的 model.pt 與 config.pbtxt 會透過
    Upload Server HTTP API 推送到遠端 Triton 模型倉庫；否則直接複製到本機磁碟。

    **GPU 部署流程**：
    當 ``instance_kind == "GPU"`` 或 ``export_device == "cuda"`` 且本機 CUDA 可用時，
    本函式會先以 ``torch.jit.load(..., map_location="cuda")`` 將 TorchScript 載入 GPU，
    再以 ``torch.jit.save`` 重新儲存為 GPU-on-disk 版本，確保 Triton libtorch backend
    以 ``KIND_GPU`` 載入時使用 CUDA operator path，而非 fallback 至 CPU 再搬移。

    @param {str | Path} best_pt_path        - 訓練產出的 .pt 檔案路徑
    @param {str} model_name                 - Triton 模型名稱
    @param {Optional[int]} project_id       - Label Studio 專案 ID（寫入後設資料）
    @param {Optional[str]} run_id           - 訓練 run ID（寫入後設資料）
    @param {Optional[str | Path]} triton_repo_root - 本機模型倉庫根目錄（本機部署用）
    @param {int} imgsz                      - 輸入圖片尺寸（用於 config.pbtxt）
    @param {Optional[int]} deployed_by_user_id    - 部署者 user ID
    @param {Optional[str]} deployed_by_username   - 部署者帳號
    @param {Optional[str]} public_triton_base_url - 外部 Triton URL（寫入後設資料）
    @param {Optional[str]} upload_server_url      - Upload Server 基底 URL；非空時走遠端上傳
    @param {str} instance_kind              - 推論裝置："GPU" | "CPU" | "AUTO"（預設 AUTO）
    @param {Optional[list[int]]} gpu_ids   - GPU 裝置 ID（instance_kind=="GPU" 時有效）
    @param {int} instance_count            - 推論實例數（預設 1）
    @param {bool} always_in_memory         - True：加入 model_warmup，常駐記憶體；False：即時載入
    @param {Optional[str]} export_device   - TorchScript 儲存裝置："cuda" | "cpu" | None；
                                             None 時若 instance_kind=="GPU" 且 CUDA 可用則自動選 "cuda"
    @param {int} target_version            - 寫入 Triton 倉庫的版本號（對應版本子目錄 1/ 2/ ...），預設為 1
    @param {Optional[str]} custom_pbtxt   - 自訂 config.pbtxt 內容；非 None 時直接使用，
                                            跳過 _build_triton_pbtxt() 自動生成
    @returns {dict} 部署結果；包含 ``error`` 鍵時表示失敗
    """
    import torch as _torch

    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        return {"error": f"TorchScript file not found: {best_pt_path}"}

    target_version = max(1, int(target_version))

    # ── GPU 可用性檢查：KIND_GPU 但本機無 CUDA → 自動降級為 KIND_AUTO ──────
    # Triton 以 KIND_GPU 啟動時會掃描本機 GPU；若容器沒有 GPU 直通
    # （例如 docker run 未加 --gpus all），Triton 會報
    # "specifies invalid or unsupported gpu id 0"。
    # 此處提前偵測並降級，避免部署後模型無法載入。
    _effective_instance_kind: str = instance_kind.upper()
    _gpu_warning: Optional[str] = None

    if _effective_instance_kind == "GPU" and not _torch.cuda.is_available():
        _effective_instance_kind = "AUTO"
        _gpu_warning = (
            "instance_kind=GPU 已請求，但本機（Django/Export 主機）未偵測到 CUDA GPU。"
            " Triton 部署設定已自動降級為 KIND_AUTO，以避免 Triton 啟動時回報"
            " 'invalid or unsupported gpu id' 錯誤。"
            " 若 Triton 容器確實擁有 GPU，請確認 Docker 以 --gpus all 啟動，"
            " 並在部署頁面重新選擇 KIND_GPU。"
        )
        logger.warning(
            "GPU deployment requested but no CUDA available on export host; "
            "downgrading Triton instance_kind from GPU → AUTO. "
            "To use GPU: start Triton container with --gpus all."
        )

    # ── 決定 TorchScript 儲存裝置 ────────────────────────────────────────────
    # 優先順序：明確指定 export_device > instance_kind 推導 > 保持原樣
    if export_device:
        _export_device: Optional[str] = export_device.lower()
    elif _effective_instance_kind == "GPU" and _torch.cuda.is_available():
        _export_device = "cuda"
    else:
        _export_device = None  # 不重新儲存，直接使用原始 .pt

    # ── 若需要特定裝置，載入 TorchScript 並以目標裝置重新儲存 ──────────────
    # 此步驟確保 Triton 載入 KIND_GPU 時拿到的是 GPU-on-disk 版本，
    # CUDA operator path 已被記錄在 TorchScript 圖中。
    _tmp_pt_path: Optional[Path] = None
    if _export_device in ("cuda", "cpu"):
        if _export_device == "cuda" and not _torch.cuda.is_available():
            logger.warning(
                "export_device='cuda' requested but CUDA is not available; "
                "using original .pt (CPU-traced) for Triton deployment."
            )
        else:
            try:
                logger.info(
                    "Re-saving TorchScript on device=%s for Triton KIND_%s deployment → %s",
                    _export_device, _effective_instance_kind, best_pt_path,
                )
                _jit_model = _torch.jit.load(
                    str(best_pt_path), map_location=_export_device
                )
                _jit_model = _jit_model.to(_export_device)
                # 暫存到同目錄，避免跨磁碟 rename 問題
                _tmp_pt_path = best_pt_path.parent / f"_export_{_export_device}_tmp.pt"
                _torch.jit.save(_jit_model, str(_tmp_pt_path))
                best_pt_path = _tmp_pt_path
                logger.info("TorchScript re-saved on %s → %s", _export_device, best_pt_path)
            except Exception as _exc:
                logger.warning(
                    "Failed to re-save TorchScript on %s: %s; "
                    "falling back to original .pt",
                    _export_device, _exc,
                )

    model_name = sanitize_triton_model_name(model_name)
    # 若呼叫端傳入 custom_pbtxt，直接使用；否則自動生成
    if custom_pbtxt and custom_pbtxt.strip():
        config_content = custom_pbtxt.strip()
        logger.info("Using custom config.pbtxt for model '%s'", model_name)
    else:
        config_content = _build_triton_pbtxt(
            model_name=model_name,
            imgsz=imgsz,
            instance_kind=_effective_instance_kind,
            gpu_ids=gpu_ids if _effective_instance_kind == "GPU" else None,
            instance_count=instance_count,
            always_in_memory=always_in_memory,
        )
    infer_base = (public_triton_base_url or get_triton_server_url()).rstrip("/")

    try:
        if upload_server_url:
            # ── 遠端模式：透過 Upload Server HTTP API 上傳 ──────────────────────
            upload_result = upload_model_to_remote_server(
                upload_base_url=upload_server_url,
                model_name=model_name,
                model_pt_path=best_pt_path,
                config_content=config_content,
                version=target_version,
            )
            if upload_result.get("error"):
                return upload_result

            # 後設資料仍寫到本機模型倉庫（供 list_triton_model_deployments 查詢）
            repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
            repo_root.mkdir(parents=True, exist_ok=True)
            model_dir = repo_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = model_dir / "deployment_meta.json"
            _deployed_at = datetime.now(timezone.utc).isoformat()
            _new_server_url = (public_triton_base_url or "").rstrip("/") or None
            metadata = {
                "model_name": model_name,
                "project_id": project_id,
                "run_id": run_id,
                "imgsz": imgsz,
                "deployed_at": _deployed_at,
                "model_type": "torchscript_cnn",
                "export_device": _export_device or "original",
                "deployed_by_user_id": deployed_by_user_id,
                "deployed_by_username": deployed_by_username,
                "upload_server_url": upload_server_url,
                "triton_public_base_url": _new_server_url,
                # 實際寫入 config.pbtxt 的裝置（可能與使用者請求不同）
                "instance_kind": _effective_instance_kind,
                "instance_kind_requested": instance_kind,
                "gpu_ids": gpu_ids if _effective_instance_kind == "GPU" else [],
                "instance_count": instance_count,
                "always_in_memory": always_in_memory,
                # 累積多台 Triton 伺服器記錄（同 URL 去重；遷移舊格式自動合併）
                "triton_servers": _merge_triton_servers(
                    metadata_path, _new_server_url, _deployed_at,
                    upload_server_url=upload_server_url,
                ),
                # 追蹤已部署的版本號（遠端模式本機無版本子目錄，靠此欄位供 auto_version 使用）
                "deployed_versions": _merge_deployed_versions(metadata_path, target_version),
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

            resp = {
                "model_name": model_name,
                "repo_path": str(repo_root),
                "model_dir": str(model_dir),
                "metadata_path": str(metadata_path),
                "infer_url": f"{infer_base}/v2/models/{model_name}/infer",
                **upload_result,
            }
            if _gpu_warning:
                resp["warning"] = _gpu_warning
            return resp

        # ── 本機模式：直接複製到磁碟 ─────────────────────────────────────────────
        repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
        repo_root.mkdir(parents=True, exist_ok=True)

        model_dir = repo_root / model_name
        version_dir = model_dir / str(target_version)
        version_dir.mkdir(parents=True, exist_ok=True)

        model_pt_path = version_dir / "model.pt"
        config_path = model_dir / "config.pbtxt"
        metadata_path = model_dir / "deployment_meta.json"

        try:
            shutil.copy2(best_pt_path, model_pt_path)
        except Exception as exc:
            return {"error": f"Failed to copy model: {exc}"}

        config_path.write_text(config_content, encoding="utf-8")

        _deployed_at = datetime.now(timezone.utc).isoformat()
        _new_server_url = public_triton_base_url.rstrip("/") if public_triton_base_url else None
        metadata = {
            "model_name": model_name,
            "project_id": project_id,
            "run_id": run_id,
            "imgsz": imgsz,
            "model_pt_path": str(model_pt_path.resolve()),
            "deployed_at": _deployed_at,
            "model_type": "torchscript_cnn",
            "export_device": _export_device or "original",
            "deployed_by_user_id": deployed_by_user_id,
            "deployed_by_username": deployed_by_username,
            # 實際寫入 config.pbtxt 的裝置（可能與使用者請求不同）
            "instance_kind": _effective_instance_kind,
            "instance_kind_requested": instance_kind,
            "gpu_ids": gpu_ids if _effective_instance_kind == "GPU" else [],
            "instance_count": instance_count,
            "always_in_memory": always_in_memory,
            # 累積多台 Triton 伺服器記錄（同 URL 去重；遷移舊格式自動合併；本機模式無 upload_server_url）
            "triton_servers": _merge_triton_servers(
                metadata_path, _new_server_url, _deployed_at, upload_server_url=None,
            ),
            # 追蹤已部署的版本號（與本機版本子目錄互補，確保 auto_version 正確計算）
            "deployed_versions": _merge_deployed_versions(metadata_path, target_version),
        }
        if _new_server_url:
            metadata["triton_public_base_url"] = _new_server_url
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

        resp = {
            "model_name": model_name,
            "repo_path": str(repo_root),
            "model_dir": str(model_dir),
            "metadata_path": str(metadata_path),
            "infer_url": f"{infer_base}/v2/models/{model_name}/infer",
        }
        if _gpu_warning:
            resp["warning"] = _gpu_warning
        return resp
    finally:
        # 清除 GPU re-save 產生的暫存 .pt，避免佔用磁碟空間
        if _tmp_pt_path and _tmp_pt_path.exists():
            try:
                _tmp_pt_path.unlink()
            except Exception:
                pass


def export_yolo_pt_to_triton(
    best_pt_path: str | Path,
    model_name: str,
    project_id: Optional[int] = None,
    run_id: Optional[str] = None,
    triton_repo_root: Optional[str | Path] = None,
    imgsz: int = 640,
    deployed_by_user_id: Optional[int] = None,
    deployed_by_username: Optional[str] = None,
    public_triton_base_url: Optional[str] = None,
    upload_server_url: Optional[str] = None,
    instance_kind: str = "AUTO",
    gpu_ids: Optional[list[int]] = None,
    instance_count: int = 1,
    always_in_memory: bool = True,
    export_device: Optional[str] = None,
    target_version: int = 1,
    custom_pbtxt: Optional[str] = None,
) -> dict:
    """
    Export a trained YOLO checkpoint into Triton's libtorch model layout.

    若 ``upload_server_url`` 不為空，匯出的 model.pt 與 config.pbtxt 會透過
    Upload Server HTTP API 推送到遠端 Triton 模型倉庫；否則直接複製到本機磁碟。

    @param {str | Path} best_pt_path        - 訓練產出的 best.pt 檔案路徑
    @param {str} model_name                 - Triton 模型名稱
    @param {Optional[int]} project_id       - Label Studio 專案 ID（寫入後設資料）
    @param {Optional[str]} run_id           - 訓練 run ID（寫入後設資料）
    @param {Optional[str | Path]} triton_repo_root - 本機模型倉庫根目錄（本機部署用）
    @param {int} imgsz                      - 輸入圖片尺寸（用於 config.pbtxt）
    @param {Optional[int]} deployed_by_user_id    - 部署者 user ID
    @param {Optional[str]} deployed_by_username   - 部署者帳號
    @param {Optional[str]} public_triton_base_url - 外部 Triton URL（寫入後設資料）
    @param {Optional[str]} upload_server_url      - Upload Server 基底 URL；非空時走遠端上傳
    @param {str} instance_kind              - 推論裝置："GPU" | "CPU" | "AUTO"（預設 AUTO）
    @param {Optional[list[int]]} gpu_ids   - GPU 裝置 ID（instance_kind=="GPU" 時有效）
    @param {int} instance_count            - 推論實例數（預設 1）
    @param {bool} always_in_memory         - True：加入 model_warmup，常駐記憶體；False：即時載入
    @param {Optional[str]} export_device   - TorchScript 匯出裝置："cuda" | "cpu" | None（None 時自動偵測 CUDA）
    @param {int} target_version            - 寫入 Triton 倉庫的版本號（對應版本子目錄 1/ 2/ ...），預設為 1
    @param {Optional[str]} custom_pbtxt   - 自訂 config.pbtxt 內容；非 None 時直接使用，
                                            跳過 _build_triton_pbtxt() 自動生成
    @returns {dict} 部署結果；包含 ``error`` 鍵時表示失敗
    """
    import torch as _torch

    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        return {"error": f"best.pt not found: {best_pt_path}"}

    target_version = max(1, int(target_version))

    model_name = sanitize_triton_model_name(model_name)
    infer_base = (public_triton_base_url or get_triton_server_url()).rstrip("/")

    # 決定 TorchScript export 所用的裝置：優先 CUDA，可由呼叫端覆寫
    if export_device:
        _export_device = export_device
    elif _torch.cuda.is_available():
        _export_device = "cuda"
    else:
        _export_device = "cpu"
    logger.info("YOLO TorchScript export device: %s", _export_device)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return {"error": f"Ultralytics not installed: {exc}"}

    # ── YOLO → TorchScript 匯出（不論本機/遠端皆需此步驟）─────────────────
    # device 參數控制 trace 時使用的運算裝置，確保 GPU operator path 被正確記錄
    try:
        model = YOLO(str(best_pt_path))
        model.to(_export_device)
        exported = model.export(format="torchscript", imgsz=imgsz, device=_export_device)
        src_torchscript = (
            Path(exported).resolve()
            if exported
            else (best_pt_path.parent / (best_pt_path.stem + ".torchscript"))
        )
        if not src_torchscript.exists():
            return {"error": f"TorchScript export did not produce {src_torchscript}"}
    except Exception as exc:
        logger.exception("Failed to export trained model to TorchScript for Triton")
        return {"error": str(exc)}

    # 若呼叫端傳入 custom_pbtxt，直接使用；否則自動生成
    if custom_pbtxt and custom_pbtxt.strip():
        config_content = custom_pbtxt.strip()
        logger.info("Using custom config.pbtxt for model '%s'", model_name)
    else:
        config_content = _build_triton_pbtxt(
            model_name=model_name,
            imgsz=imgsz,
            instance_kind=instance_kind,
            gpu_ids=gpu_ids,
            instance_count=instance_count,
            always_in_memory=always_in_memory,
        )

    if upload_server_url:
        # ── 遠端模式：透過 Upload Server HTTP API 上傳 ──────────────────────
        upload_result = upload_model_to_remote_server(
            upload_base_url=upload_server_url,
            model_name=model_name,
            model_pt_path=src_torchscript,
            config_content=config_content,
            version=target_version,
        )
        if upload_result.get("error"):
            return upload_result

        # 後設資料寫到本機模型倉庫（供 list_triton_model_deployments 查詢）
        repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
        repo_root.mkdir(parents=True, exist_ok=True)
        model_dir = repo_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / "deployment_meta.json"
        _deployed_at = datetime.now(timezone.utc).isoformat()
        _new_server_url = (public_triton_base_url or "").rstrip("/") or None
        metadata = {
            "model_name": model_name,
            "project_id": project_id,
            "run_id": run_id,
            "imgsz": imgsz,
            "source_pt_path": str(best_pt_path.resolve()),
            "exported_torchscript_path": str(src_torchscript.resolve()),
            "deployed_at": _deployed_at,
            "deployed_by_user_id": deployed_by_user_id,
            "deployed_by_username": deployed_by_username,
            "upload_server_url": upload_server_url,
            "triton_public_base_url": _new_server_url,
            "instance_kind": instance_kind,
            "gpu_ids": gpu_ids or [],
            "instance_count": instance_count,
            "always_in_memory": always_in_memory,
            # 累積多台 Triton 伺服器記錄（同 URL 去重；遷移舊格式自動合併）
            "triton_servers": _merge_triton_servers(
                metadata_path, _new_server_url, _deployed_at,
                upload_server_url=upload_server_url,
            ),
            # 追蹤已部署的版本號（遠端模式本機無版本子目錄，靠此欄位供 auto_version 使用）
            "deployed_versions": _merge_deployed_versions(metadata_path, target_version),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

        return {
            "model_name": model_name,
            "repo_path": str(repo_root),
            "model_dir": str(model_dir),
            "metadata_path": str(metadata_path),
            "infer_url": f"{infer_base}/v2/models/{model_name}/infer",
            **upload_result,
        }

    # ── 本機模式：直接複製到磁碟 ─────────────────────────────────────────────
    repo_root = Path(triton_repo_root) if triton_repo_root else get_triton_model_repository_root()
    repo_root.mkdir(parents=True, exist_ok=True)

    model_dir = repo_root / model_name
    version_dir = model_dir / str(target_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    model_pt_path = version_dir / "model.pt"
    config_path = model_dir / "config.pbtxt"
    legacy_bptxt_path = model_dir / f"{model_name}.bptxt"
    metadata_path = model_dir / "deployment_meta.json"

    try:
        shutil.copy2(src_torchscript, model_pt_path)
    except Exception as exc:
        return {"error": f"Failed to copy TorchScript to repo: {exc}"}

    config_path.write_text(config_content, encoding="utf-8")
    legacy_bptxt_path.write_text(config_content, encoding="utf-8")

    _deployed_at = datetime.now(timezone.utc).isoformat()
    _new_server_url = public_triton_base_url.rstrip("/") if public_triton_base_url else None
    metadata = {
        "model_name": model_name,
        "project_id": project_id,
        "run_id": run_id,
        "imgsz": imgsz,
        "source_pt_path": str(best_pt_path.resolve()),
        "exported_torchscript_path": str(src_torchscript.resolve()),
        "model_pt_path": str(model_pt_path.resolve()),
        "config_path": str(config_path.resolve()),
        "legacy_bptxt_path": str(legacy_bptxt_path.resolve()),
        "deployed_at": _deployed_at,
        "deployed_by_user_id": deployed_by_user_id,
        "deployed_by_username": deployed_by_username,
        "instance_kind": instance_kind,
        "gpu_ids": gpu_ids or [],
        "instance_count": instance_count,
        "always_in_memory": always_in_memory,
        # 累積多台 Triton 伺服器記錄（同 URL 去重；遷移舊格式自動合併；本機模式無 upload_server_url）
        "triton_servers": _merge_triton_servers(
            metadata_path, _new_server_url, _deployed_at, upload_server_url=None,
        ),
        # 追蹤已部署的版本號（與本機版本子目錄互補，確保 auto_version 正確計算）
        "deployed_versions": _merge_deployed_versions(metadata_path, target_version),
    }
    if _new_server_url:
        metadata["triton_public_base_url"] = _new_server_url
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "model_name": model_name,
        "repo_path": str(repo_root),
        "model_dir": str(model_dir),
        "version_dir": str(version_dir),
        "model_pt_path": str(model_pt_path),
        "config_path": str(config_path),
        "legacy_bptxt_path": str(legacy_bptxt_path),
        "metadata_path": str(metadata_path),
        "infer_url": f"{infer_base}/v2/models/{model_name}/infer",
    }


def _merge_deployed_versions(metadata_path: Path, new_version: int) -> list[int]:
    """
    讀取現有 ``deployment_meta.json`` 中的 ``deployed_versions`` 陣列，
    加入 ``new_version`` 後去重排序並回傳。

    遠端部署（Upload Server）模式下本機不建立版本子目錄，
    靠此函式在 metadata 中追蹤已部署的版本號，供 auto_version 計算使用。

    @param {Path} metadata_path - deployment_meta.json 的路徑（可能尚不存在）
    @param {int} new_version    - 本次部署的版本號
    @returns {list[int]} 排序後的已部署版本號清單
    """
    existing: list[int] = []
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing = [int(v) for v in meta.get("deployed_versions", []) if str(v).isdigit()]
        except Exception:
            pass
    return sorted(set(existing) | {new_version})


def _merge_triton_servers(
    metadata_path: Path,
    new_server_url: Optional[str],
    deployed_at: str,
    upload_server_url: Optional[str] = None,
) -> list:
    """
    讀取現有 ``deployment_meta.json`` 中的 ``triton_servers`` 陣列，將新伺服器合併後回傳。

    - 以正規化後的 URL 進行去重（同 URL 視為同一台伺服器，更新部署時間）。
    - 自動遷移舊格式：若現有檔案只有 ``triton_public_base_url`` 而無 ``triton_servers``，
      先將舊值轉為清單的第一筆再追加。
    - ``upload_server_url`` 記錄於每筆伺服器項目，供單台刪除時呼叫對應的 Upload Server。

    :param metadata_path: ``deployment_meta.json`` 的 Path（可能尚不存在）。
    :param new_server_url: 新部署目標 Triton 基底 URL，例如 ``http://10.214.57.66:18000``。
    :param deployed_at: 本次部署的 ISO 8601 時間字串。
    :param upload_server_url: 對應的 Upload Server URL（遠端部署時傳入，本機部署為 None）。
    :returns: 更新後的 triton_servers 清單，每項格式為::

        {
            "url": str,                # Triton HTTP 基底 URL
            "deployed_at": str,        # ISO 8601
            "upload_server_url": str   # Upload Server URL（可為空字串）
        }
    """
    servers: list = []
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing_servers = existing.get("triton_servers")
            if existing_servers and isinstance(existing_servers, list):
                servers = list(existing_servers)
            else:
                # 遷移舊格式：triton_public_base_url 單一字串 → 清單第一項
                legacy_url = (existing.get("triton_public_base_url") or "").strip().rstrip("/")
                if legacy_url:
                    servers = [{
                        "url": legacy_url,
                        "deployed_at": existing.get("deployed_at", ""),
                        "upload_server_url": existing.get("upload_server_url") or "",
                    }]
        except Exception:
            servers = []

    new_url_norm = (new_server_url or "").strip().rstrip("/")
    if new_url_norm:
        # 去重：移除相同 URL 的舊紀錄，再附加最新記錄
        servers = [s for s in servers if (s.get("url") or "").rstrip("/") != new_url_norm]
        servers.append({
            "url": new_url_norm,
            "deployed_at": deployed_at,
            "upload_server_url": (upload_server_url or "").strip().rstrip("/"),
        })

    return servers


def list_triton_model_deployments(
    project_ids: Optional[Iterable[int]] = None,
    deployed_by_user_ids: Optional[Iterable[int]] = None,
) -> list[dict]:
    """
    List Triton deployments discovered from repository metadata files.
    同時掃描模型目錄下所有數字版本子目錄（1/ 2/ 3/ ...），
    並於回傳結果中附帶 ``available_versions`` 清單（僅含實際存在的目錄）。
    """
    repo_root = get_triton_model_repository_root()
    if not repo_root.exists():
        return []

    allowed_projects = {int(project_id) for project_id in project_ids or []}
    allowed_users = {int(user_id) for user_id in deployed_by_user_ids or []}
    items = []

    for metadata_path in repo_root.glob("*/deployment_meta.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Invalid Triton deployment metadata: %s", metadata_path)
            continue

        project_id = metadata.get("project_id")
        if allowed_projects and project_id not in allowed_projects:
            continue
        deployed_by_user_id = metadata.get("deployed_by_user_id")
        if allowed_users and deployed_by_user_id not in allowed_users:
            continue

        model_name = metadata.get("model_name")
        model_dir = metadata_path.parent
        public = (metadata.get("triton_public_base_url") or "").strip().rstrip("/")
        infer_base = public or get_triton_server_url().rstrip("/")

        # 掃描本機版本子目錄（本機部署）
        _local_versions: list[int] = [
            int(d.name)
            for d in model_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        # 合併 metadata 中的 deployed_versions（遠端部署本機無子目錄，靠此欄位追蹤）
        _meta_versions: list[int] = [
            int(v) for v in metadata.get("deployed_versions", [])
            if str(v).isdigit()
        ]
        available_versions: list[int] = sorted(set(_local_versions) | set(_meta_versions))
        # 相容舊版：至少確保 version_dir 指向最新已存在版本（或版本 1）
        latest_version = available_versions[-1] if available_versions else 1
        version_dir = model_dir / str(latest_version)

        items.append(
            {
                **metadata,
                "model_dir": str(model_dir),
                "version_dir": str(version_dir),
                "latest_version": latest_version,
                "available_versions": available_versions,
                "exists": version_dir.exists(),
                "infer_url": f"{infer_base}/v2/models/{model_name}/infer",
            }
        )

    return sorted(items, key=lambda item: item.get("deployed_at") or "", reverse=True)
