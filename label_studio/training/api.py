"""REST API for on-server training jobs (YOLO detect)."""

from __future__ import annotations

import os
from pathlib import Path
import json
from typing import Any, Dict, List

import django_rq
from core.permissions import ViewClassPermission, all_permissions
from django.conf import settings
from projects.models import Project
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rq.job import Job

from .jobs import yolo_detect_train_job
from .datasets import prepare_yolo_dataset_from_project_export


def _get_original_models_dir() -> Path:
    """
    Directory that contains locally-available base weights (.pt).

    Change here if you want to move the directory later:
    - Default: <BASE_DIR>/data/training/models/original
    """

    # We derive the repo root from this file location to avoid issues with:
    # - `settings.BASE_DIR` pointing to `label_studio/core/settings`
    # - different CWD when running under gunicorn/uvicorn/management commands
    repo_root = Path(__file__).resolve().parents[2]  # label_studio/training/api.py -> label-studio
    return repo_root / "data" / "training" / "models" / "original"


def _get_training_output_root() -> Path:
    """
    Root directory for training artifacts and metrics.

    Change here if you want to store artifacts elsewhere (e.g. S3):
    - Default: <BASE_DIR>/data/training/models/trained
    """

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "training" / "models" / "trained"


class ProjectTrainingModelsAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        root = _get_original_models_dir()
        root.mkdir(parents=True, exist_ok=True)

        models: List[Dict[str, Any]] = []
        for p in sorted(root.glob("*.pt"), key=lambda x: x.name.lower()):
            st = p.stat()
            models.append(
                {
                    "id": p.name,
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "modified_at": st.st_mtime,
                }
            )

        return Response({"models": models, "root": str(root)})


class ProjectTrainingJobsAPI(APIView):
    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        payload = request.data or {}
        base_weights = payload.get("base_weights")
        data_yaml = payload.get("data_yaml")
        epochs = int(payload.get("epochs", 50))
        imgsz = int(payload.get("imgsz", 640))
        batch = int(payload.get("batch", 16))

        if not base_weights:
            return Response({"detail": "base_weights is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not data_yaml:
            return Response({"detail": "data_yaml is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Fail fast with clear messages (so UI can show actionable errors)
        if os.path.isabs(str(base_weights)) and not Path(str(base_weights)).exists():
            return Response(
                {"detail": f"base_weights not found on server: {base_weights}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not Path(str(data_yaml)).exists():
            return Response(
                {"detail": f"data.yaml not found on server: {data_yaml}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        output_root = _get_training_output_root()
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            queue = django_rq.get_queue("low")
            job = queue.enqueue(
                yolo_detect_train_job,
                kwargs={
                    "project_id": project.id,
                    "base_weights": base_weights,
                    "data_yaml": data_yaml,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "output_root": str(output_root),
                },
                job_timeout=int(payload.get("job_timeout", 60 * 60 * 6)),  # default 6 hours
            )
        except Exception as exc:
            return Response(
                {"detail": f"Failed to enqueue training job. Is Redis/RQ worker running? {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response({"job_id": job.id, "status": job.get_status()})


class ProjectTrainingDatasetPrepareAPI(APIView):
    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        payload = request.data or {}
        train_ratio = float(payload.get("train_ratio", 0.8))
        seed = int(payload.get("seed", 42))

        # We start from LS native JSON internally, but we leverage the built-in export converter
        # to produce YOLO_WITH_IMAGES assets reliably.
        meta = prepare_yolo_dataset_from_project_export(
            project_id=project.id,
            train_ratio=train_ratio,
            seed=seed,
            export_format="YOLO_WITH_IMAGES",
        )

        return Response(meta, status=status.HTTP_200_OK)


class ProjectTrainingJobDetailAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, job_id: str, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        queue = django_rq.get_queue("low")
        try:
            job = Job.fetch(job_id, connection=queue.connection)
        except Exception:
            return Response({"detail": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

        meta = dict(job.meta or {})
        return Response(
            {
                "job_id": job.id,
                "status": job.get_status(),
                "meta": meta,
                "created_at": getattr(job, "created_at", None),
                "enqueued_at": getattr(job, "enqueued_at", None),
                "ended_at": getattr(job, "ended_at", None),
                "exc_info": job.exc_info if job.is_failed else None,
            }
        )


class ProjectTrainingJobArtifactsAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, job_id: str, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        queue = django_rq.get_queue("low")
        job = Job.fetch(job_id, connection=queue.connection)
        artifacts_dir = (job.meta or {}).get("artifacts_dir")
        if not artifacts_dir:
            return Response({"artifacts": []})

        root = Path(artifacts_dir)
        if not root.exists():
            return Response({"artifacts": []})

        artifacts = []
        for p in sorted(root.glob("*"), key=lambda x: x.name.lower()):
            if p.is_dir():
                continue
            artifacts.append(
                {
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "download_url": f"/api/projects/{project.id}/training/jobs/{job.id}/download?file={p.name}",
                }
            )
        return Response({"artifacts": artifacts, "root": str(root)})


class ProjectTrainingJobDownloadAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, job_id: str, *args, **kwargs):
        from django.http import FileResponse, Http404

        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        file_name = request.query_params.get("file")
        if not file_name:
            return Response({"detail": "file query param is required"}, status=status.HTTP_400_BAD_REQUEST)

        queue = django_rq.get_queue("low")
        job = Job.fetch(job_id, connection=queue.connection)
        artifacts_dir = (job.meta or {}).get("artifacts_dir")
        if not artifacts_dir:
            raise Http404

        target = Path(artifacts_dir) / file_name
        if not target.exists():
            raise Http404

        return FileResponse(open(target, "rb"), as_attachment=True, filename=target.name)


class ProjectTrainingHistoryAPI(APIView):
    """
    List previously prepared datasets and trained model runs on disk.

    This does NOT depend on Redis job history (which can expire).
    """

    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, *args, **kwargs):
        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        # Runs are stored by our training job under:
        #   data/training/models/trained/project_<id>/<job_id>/
        runs_root = _get_training_output_root() / f"project_{project.id}"
        runs = []
        if runs_root.exists():
            for d in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not d.is_dir():
                    continue
                run_meta_path = d / "run_meta.json"
                run_meta = {}
                if run_meta_path.exists():
                    try:
                        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        run_meta = {}
                metrics_path = d / "metrics.json"
                metrics = None
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except Exception:
                        metrics = None
                artifacts_dir = d / "artifacts"
                files = []
                if artifacts_dir.exists():
                    for f in sorted(artifacts_dir.glob("*"), key=lambda p: p.name.lower()):
                        if f.is_file():
                            files.append(
                                {
                                    "name": f.name,
                                    "size_bytes": f.stat().st_size,
                                    "download_url": f"/api/projects/{project.id}/training/runs/{d.name}/download?file={f.name}",
                                }
                            )
                runs.append(
                    {
                        "run_id": d.name,
                        "run_dir": str(d),
                        "status": run_meta.get("status", "finished" if metrics else "unknown"),
                        "message": run_meta.get("message"),
                        "params": run_meta.get("params"),
                        "error": run_meta.get("error"),
                        "metrics": metrics,
                        "artifacts": files,
                        "best_download_url": f"/api/projects/{project.id}/training/runs/{d.name}/download?file=best.pt",
                        "last_download_url": f"/api/projects/{project.id}/training/runs/{d.name}/download?file=last.pt",
                        "modified_at": d.stat().st_mtime,
                    }
                )

        # Datasets are stored under:
        #   data/training/datasets/project_<id>/<timestamp>/
        from .datasets import get_datasets_root

        ds_root = get_datasets_root() / f"project_{project.id}"
        datasets = []
        if ds_root.exists():
            for d in sorted(ds_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not d.is_dir():
                    continue
                meta_path = d / "dataset_meta.json"
                meta = None
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = None
                datasets.append(
                    {
                        "dataset_id": d.name,
                        "dataset_root": str(d),
                        "meta": meta,
                        "data_yaml": str(d / "data.yaml") if (d / "data.yaml").exists() else None,
                        "modified_at": d.stat().st_mtime,
                    }
                )

        return Response(
            {
                "runs_root": str(runs_root),
                "datasets_root": str(ds_root),
                "runs": runs,
                "datasets": datasets,
            }
        )


class ProjectTrainingRunDownloadAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, run_id: str, *args, **kwargs):
        from django.http import FileResponse, Http404

        project = Project.objects.get(pk=pk)
        self.check_object_permissions(request, project)

        file_name = request.query_params.get("file")
        if not file_name:
            return Response({"detail": "file query param is required"}, status=status.HTTP_400_BAD_REQUEST)

        base = _get_training_output_root() / f"project_{project.id}" / run_id / "artifacts"
        target = base / file_name
        if not target.exists():
            raise Http404

        return FileResponse(open(target, "rb"), as_attachment=True, filename=target.name)

