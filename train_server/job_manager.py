"""In-process job registry for Train Server."""

from __future__ import annotations

import json
import logging
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .metrics_utils import json_safe
from .trainer import run_yolo_training

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, output_root: Path, max_workers: int = 1):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yolo-train")

    def _save_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        path = self.output_root / f"project_{job['project_id']}" / job_id / "job_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(job), ensure_ascii=False, indent=2), encoding="utf-8")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])
        for state_file in self.output_root.glob(f"project_*/{job_id}/job_state.json"):
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def list_runs(self, project_id: int) -> list[Dict[str, Any]]:
        runs = []
        root = self.output_root / f"project_{project_id}"
        if not root.exists():
            return runs
        for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not meta.get("job_id"):
                        meta["job_id"] = d.name
                    if not meta.get("status") or str(meta.get("status")).strip() in ("", "unknown"):
                        best_pt = d / "train" / "weights" / "best.pt"
                        last_pt = d / "train" / "weights" / "last.pt"
                        if best_pt.is_file() or last_pt.is_file():
                            meta["status"] = "finished"
                            meta.setdefault("message", "Training finished")
                    runs.append(meta)
                    continue
                except Exception:
                    pass
            best_pt = d / "train" / "weights" / "best.pt"
            if best_pt.is_file():
                runs.append(
                    {
                        "job_id": d.name,
                        "project_id": project_id,
                        "status": "finished",
                        "message": "Training finished",
                        "run_dir": str(d),
                        "best_path": str(best_pt),
                        "finished_at": datetime.fromtimestamp(best_pt.stat().st_mtime).isoformat(),
                    }
                )
        return runs

    def create_job(
        self,
        *,
        project_id: int,
        base_weights: str,
        dataset_zip_path: Path | None,
        dataset_path: str | None,
        dataset_meta: Dict[str, Any],
        param_overrides: Dict[str, Any] | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        job: Dict[str, Any] = {
            "job_id": job_id,
            "project_id": project_id,
            "status": "queued",
            "message": "Job queued",
            "created_at": now,
            "base_weights": base_weights,
            "params": param_overrides or {},
            "meta": {},
        }
        with self._lock:
            self._jobs[job_id] = job
        self._save_job(job_id)

        run_dir = self.output_root / f"project_{project_id}" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        initial_meta: Dict[str, Any] = {
            "job_id": job_id,
            "project_id": project_id,
            "status": "queued",
            "message": "Job queued",
            "created_at": now,
            "params": param_overrides or {},
        }
        if param_overrides and param_overrides.get("run_name"):
            initial_meta["name"] = str(param_overrides["run_name"])
        (run_dir / "run_meta.json").write_text(json.dumps(initial_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._executor.submit(
            self._run_job,
            job_id,
            project_id,
            base_weights,
            dataset_zip_path,
            dataset_path,
            dataset_meta,
            param_overrides,
        )
        return job_id

    def _prepare_dataset(
        self,
        job_id: str,
        project_id: int,
        dataset_zip_path: Path | None,
        dataset_path: str | None,
        dataset_meta: Dict[str, Any],
    ) -> Path:
        if dataset_path:
            root = Path(dataset_path)
            if not root.exists():
                raise FileNotFoundError(f"dataset_path not found: {dataset_path}")
            return root

        if not dataset_zip_path or not dataset_zip_path.exists():
            raise FileNotFoundError("dataset_zip or dataset_path is required")

        extract_root = self.output_root / f"project_{project_id}" / job_id / "dataset"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_zip_path, "r") as zf:
            names = zf.namelist()
            image_entries = [n for n in names if n.lower().startswith("images/") and not n.endswith("/")]
            logger.info(
                "Extracting dataset zip for job %s: %s files (%s under images/)",
                job_id,
                len(names),
                len(image_entries),
            )
            if not image_entries:
                raise FileNotFoundError(
                    "Uploaded dataset zip contains no files under images/. "
                    "Regenerate the training dataset in Label Studio and resubmit."
                )
            zf.extractall(extract_root)

        # 若 zip 只有一層目錄，進入該目錄
        children = [p for p in extract_root.iterdir() if p.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return extract_root

    def _run_job(
        self,
        job_id: str,
        project_id: int,
        base_weights: str,
        dataset_zip_path: Path | None,
        dataset_path: str | None,
        dataset_meta: Dict[str, Any],
        param_overrides: Dict[str, Any] | None,
    ) -> None:
        def on_status(payload: Dict[str, Any]) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job["status"] = payload.get("status", job.get("status"))
                job["message"] = payload.get("message", job.get("message"))
                safe_payload = json_safe(payload)
                if isinstance(safe_payload, dict):
                    job["meta"] = {**job.get("meta", {}), **safe_payload}
            self._save_job(job_id)

        try:
            on_status({"status": "preparing", "message": "Preparing dataset"})
            dataset_root = self._prepare_dataset(
                job_id, project_id, dataset_zip_path, dataset_path, dataset_meta
            )
            run_yolo_training(
                project_id=project_id,
                job_id=job_id,
                base_weights=base_weights,
                dataset_root=dataset_root,
                dataset_meta=dict(dataset_meta),
                output_root=self.output_root,
                param_overrides=param_overrides,
                on_status=on_status,
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["status"] = "finished"
                    job["message"] = "Training finished"
            self._save_job(job_id)
        except Exception as exc:
            run_dir = self.output_root / f"project_{project_id}" / job_id
            best_pt = run_dir / "artifacts" / "best.pt"
            last_pt = run_dir / "artifacts" / "last.pt"
            weights_done = best_pt.is_file() or last_pt.is_file()
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    if weights_done:
                        job["status"] = "finished"
                        job["message"] = "Training finished"
                        job["warning"] = str(exc)
                    else:
                        job["status"] = "failed"
                        job["message"] = "Training failed"
                        job["error"] = str(exc)
            self._save_job(job_id)
            run_dir = self.output_root / f"project_{project_id}" / job_id
            run_meta_path = run_dir / "run_meta.json"
            failed_meta = {
                "job_id": job_id,
                "project_id": project_id,
                "status": "finished" if weights_done else "failed",
                "message": "Training finished" if weights_done else "Training failed",
                "error": None if weights_done else str(exc),
                "warning": str(exc) if weights_done else None,
                "failed_at": datetime.now().isoformat(),
            }
            if run_meta_path.exists():
                try:
                    existing = json.loads(run_meta_path.read_text(encoding="utf-8"))
                    existing.update(failed_meta)
                    failed_meta = existing
                except Exception:
                    pass
            run_meta_path.parent.mkdir(parents=True, exist_ok=True)
            run_meta_path.write_text(json.dumps(json_safe(failed_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    def get_artifacts_dir(self, job_id: str) -> Optional[Path]:
        job = self.get_job(job_id)
        if not job:
            return None
        meta = job.get("meta") or {}
        artifacts = meta.get("artifacts_dir")
        if artifacts:
            p = Path(artifacts)
            if p.exists():
                return p
        project_id = job.get("project_id")
        if project_id:
            candidate = self.output_root / f"project_{project_id}" / job_id / "artifacts"
            if candidate.exists():
                return candidate
        return None

    def get_run_dir(self, job_id: str) -> Optional[Path]:
        job = self.get_job(job_id)
        if not job:
            return None
        meta = job.get("meta") or {}
        run_dir = meta.get("run_dir")
        if run_dir:
            p = Path(run_dir)
            if p.exists():
                return p
        project_id = job.get("project_id")
        if project_id:
            candidate = self.output_root / f"project_{project_id}" / job_id
            if candidate.exists():
                return candidate
        return None

    def resolve_artifact_path(self, job_id: str, file_name: str) -> Optional[Path]:
        """Find artifact under artifacts/ or train/weights/ (YOLO default layout)."""
        safe_name = Path(file_name).name
        run_dir = self.get_run_dir(job_id)
        if not run_dir:
            return None

        candidates = [
            run_dir / "artifacts" / safe_name,
            run_dir / "train" / "weights" / safe_name,
            run_dir / "train" / safe_name,
        ]
        for path in candidates:
            if path.is_file():
                return path

        if safe_name.endswith(".pt"):
            for path in sorted(run_dir.rglob(safe_name)):
                if path.is_file():
                    return path
        return None

    def list_artifact_files(self, job_id: str) -> list[Dict[str, Any]]:
        run_dir = self.get_run_dir(job_id)
        if not run_dir:
            return []

        seen: set[str] = set()
        artifacts: list[Dict[str, Any]] = []

        def _add(path: Path) -> None:
            if not path.is_file() or path.name in seen:
                return
            seen.add(path.name)
            artifacts.append({"name": path.name, "size_bytes": path.stat().st_size})

        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.is_dir():
            for p in sorted(artifacts_dir.glob("*"), key=lambda x: x.name.lower()):
                _add(p)

        weights_dir = run_dir / "train" / "weights"
        if weights_dir.is_dir():
            for name in ("best.pt", "last.pt"):
                _add(weights_dir / name)

        for name in (
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "args.yaml",
        ):
            _add(run_dir / "train" / name)
            _add(artifacts_dir / name)

        return artifacts

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        from .progress import build_progress_snapshot

        job = self.get_job(job_id)
        if not job:
            return None
        run_dir = self.get_run_dir(job_id)
        if not run_dir:
            run_dir = self.output_root / f"project_{job['project_id']}" / job_id
        params = job.get("params") or {}
        total_epochs = params.get("epochs")
        meta = job.get("meta") or {}
        if total_epochs is None:
            total_epochs = (meta.get("params") or {}).get("epochs")
        snapshot = build_progress_snapshot(
            run_dir=run_dir,
            status=str(job.get("status") or "unknown"),
            message=job.get("message"),
            total_epochs=int(total_epochs) if total_epochs else None,
            extra={
                "job_id": job_id,
                "project_id": job.get("project_id"),
                "error": job.get("error"),
            },
        )
        return snapshot
