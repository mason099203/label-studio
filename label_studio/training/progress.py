"""Parse Ultralytics training progress from results.csv and preview artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def parse_results_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed: Dict[str, Any] = {"epoch": _safe_float(row.get("epoch"))}
                for key, val in row.items():
                    if key == "epoch":
                        continue
                    parsed[key.strip()] = _safe_float(val) if val not in (None, "") else val
                if parsed.get("epoch") is not None:
                    rows.append(parsed)
    except Exception:
        return []
    return rows


def find_train_run_dir(job_run_dir: Path) -> Optional[Path]:
    direct = job_run_dir / "train"
    if direct.is_dir():
        return direct
    for candidate in job_run_dir.glob("**/results.csv"):
        return candidate.parent
    return None


def collect_preview_images(run_dir: Path, dest_dir: Path | None = None) -> List[str]:
    names: List[str] = []
    if not run_dir or not run_dir.exists():
        return names
    patterns = [
        "results.png",
        "train_batch*.jpg",
        "train_batch*.png",
        "val_batch*.jpg",
        "val_batch*.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    for pattern in patterns:
        for p in sorted(run_dir.glob(pattern)):
            if p.is_file():
                if dest_dir:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    target = dest_dir / p.name
                    if not target.exists() or p.stat().st_mtime > target.stat().st_mtime:
                        try:
                            import shutil

                            shutil.copy2(p, target)
                        except Exception:
                            pass
                names.append(p.name)
    return sorted(set(names))


def build_progress_snapshot(
    *,
    run_dir: Path,
    status: str,
    message: str | None = None,
    total_epochs: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    progress_path = run_dir / "progress.json"
    saved: Dict[str, Any] = {}
    if progress_path.exists():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            saved = {}

    train_dir = find_train_run_dir(run_dir) or run_dir
    history = parse_results_csv(train_dir / "results.csv")
    epoch = saved.get("epoch")
    if epoch is None and history:
        epoch = history[-1].get("epoch")

    te = total_epochs or saved.get("total_epochs")
    if te is None and train_dir.joinpath("args.yaml").exists():
        try:
            import yaml

            args = yaml.safe_load(train_dir.joinpath("args.yaml").read_text(encoding="utf-8"))
            te = int(args.get("epochs") or 0) or None
        except Exception:
            te = None

    progress_pct = saved.get("progress_pct")
    if progress_pct is None and epoch is not None and te:
        progress_pct = round(float(epoch) / float(te) * 100, 1)

    live_dir = run_dir / "artifacts" / "live"
    preview_names = collect_preview_images(train_dir, dest_dir=live_dir)

    latest_metrics: Dict[str, Any] = {}
    if history:
        latest_metrics = {k: v for k, v in history[-1].items() if k != "epoch"}

    payload: Dict[str, Any] = {
        "status": status,
        "message": message or saved.get("message"),
        "epoch": epoch,
        "total_epochs": te,
        "progress_pct": progress_pct,
        "latest_metrics": latest_metrics or saved.get("latest_metrics") or {},
        "history": history,
        "preview_images": preview_names,
        "run_dir": str(run_dir),
    }
    if extra:
        payload.update(extra)
    return payload
