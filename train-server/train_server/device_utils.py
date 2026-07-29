"""Train Server 執行環境解析：裝置、DataLoader workers 等。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def resolve_training_device(explicit: Optional[str] = None) -> str:
    """
    決定 YOLO 訓練裝置。

    - 環境變數 TRAIN_SERVER_DEVICE（如 cpu / 0）優先於自動偵測（仍可被請求 explicit 覆寫）
    - 未指定時：CUDA 可用且能跑 kernel → "0"，否則 "cpu"
    - 明確指定 "cuda" / "gpu"：有 GPU 用 "0"，否則 "cpu"
    - 其他值（如 "cpu"、"0"、"1"）原樣傳給 Ultralytics
    """
    if explicit:
        raw = str(explicit).strip()
        lowered = raw.lower()
        if lowered in ("cuda", "gpu"):
            return _default_device()
        return raw
    env_device = (os.environ.get("TRAIN_SERVER_DEVICE") or "").strip()
    if env_device:
        lowered = env_device.lower()
        if lowered in ("cuda", "gpu"):
            return _default_device()
        logger.info("Using TRAIN_SERVER_DEVICE=%s", env_device)
        return env_device
    return _default_device()


def _cuda_kernel_usable() -> bool:
    """torch.cuda.is_available() 在架構不相容時仍可能為 True；用極小運算探測。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        logger.warning(
            "CUDA 可見但無法執行 kernel（常見於 GPU 架構與 PyTorch wheel 不符，"
            "例如 RTX 50 / sm_120 配 cu124）。將改用 CPU。原因：%s",
            exc,
        )
        return False


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available() and _cuda_kernel_usable():
            name = torch.cuda.get_device_name(0)
            logger.info("Using GPU device 0: %s", name)
            return "0"
    except Exception as exc:
        logger.debug("CUDA unavailable: %s", exc)
    logger.info("No usable CUDA GPU; using CPU")
    return "cpu"


def resolve_dataloader_workers(explicit: Optional[str | int] = None, *, user_specified: bool = False) -> int:
    """
    DataLoader workers 數量。

    Docker 預設 /dev/shm 僅 64MB，workers>0 常觸發 bus error / worker exited unexpectedly。
    預設 0（主進程載入）；可透過 TRAIN_SERVER_DATALOADER_WORKERS 或請求 param_overrides 覆寫。
    """
    if user_specified and explicit is not None:
        return max(0, int(explicit))
    raw = os.environ.get("TRAIN_SERVER_DATALOADER_WORKERS", "0").strip()
    try:
        workers = max(0, int(raw))
    except ValueError:
        workers = 0
    if not user_specified and explicit is not None and int(explicit) != workers:
        logger.info(
            "Using DataLoader workers=%s (catalog default=%s); set TRAIN_SERVER_DATALOADER_WORKERS or train_params.workers to override",
            workers,
            explicit,
        )
    return workers


def get_torch_device_info() -> Dict[str, Any]:
    """供 /health 回報目前 torch 偵測到的裝置能力。"""
    info: Dict[str, Any] = {
        "default_device": resolve_training_device(None),
        "dataloader_workers": resolve_dataloader_workers(),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_kernel_usable"] = _cuda_kernel_usable() if info["cuda_available"] else False
        if info["cuda_available"]:
            count = torch.cuda.device_count()
            info["cuda_device_count"] = count
            info["cuda_devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i)} for i in range(count)
            ]
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info
