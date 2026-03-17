"""REST API for on-server training jobs (YOLO detect)."""

from __future__ import annotations

import os
from pathlib import Path
import json
import re
import time
from typing import Any, Dict, List
import requests

import django_rq
from core.permissions import ViewClassPermission, all_permissions
from django.conf import settings
from django.shortcuts import get_object_or_404
from projects.models import Project
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rq.job import Job

from .jobs import cnn_classification_train_job, yolo_classification_train_job, yolo_detect_train_job
from .datasets import detect_training_interface, prepare_training_dataset_for_project
from .triton_export import (
    export_torchscript_pt_to_triton,
    export_yolo_pt_to_triton,
    get_triton_model_repository_root,
    get_triton_server_url,
    list_triton_model_deployments,
    sanitize_triton_model_name,
)


_TRITON_COUNTER_CACHE: Dict[str, Dict[str, float]] = {}


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


def _read_dataset_config(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"dataset_config not found on server: {path_str}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_project_for_user(request, pk: int) -> Project:
    """
    Resolve a project visible to the current user via enabled membership.
    """

    return get_object_or_404(Project.objects.for_user(request.user), pk=pk)


def _safe_float(value: Any) -> float:
    """
    Convert a metric value to float, returning 0 for invalid values.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_prometheus_labels(label_text: str) -> Dict[str, str]:
    """
    Parse Prometheus label text into a dict.
    """

    labels: Dict[str, str] = {}
    if not label_text:
        return labels

    for key, value in re.findall(r'(\w+)="([^"]*)"', label_text):
        labels[key] = value

    return labels


def _parse_prometheus_metrics(payload: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Prometheus text exposition into grouped metric samples.
    """

    metrics: Dict[str, List[Dict[str, Any]]] = {}

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            metric_part, value_part = line.rsplit(" ", 1)
        except ValueError:
            continue

        label_start = metric_part.find("{")
        if label_start >= 0 and metric_part.endswith("}"):
            metric_name = metric_part[:label_start]
            labels = _parse_prometheus_labels(metric_part[label_start + 1 : -1])
        else:
            metric_name = metric_part
            labels = {}

        metrics.setdefault(metric_name, []).append(
            {
                "labels": labels,
                "value": _safe_float(value_part),
            }
        )

    return metrics


def _sum_metric_samples(
    metrics: Dict[str, List[Dict[str, Any]]],
    metric_name: str,
    *,
    label_key: str | None = None,
    label_value: str | None = None,
) -> float:
    """
    Sum a Prometheus metric, optionally filtered by a label.
    """

    total = 0.0
    for sample in metrics.get(metric_name, []):
        labels = sample.get("labels", {})
        if label_key and labels.get(label_key) != label_value:
            continue
        total += _safe_float(sample.get("value"))
    return total


def _compute_counter_rate(cache_key: str, counter_value: float) -> float:
    """
    Compute a best-effort per-second rate from a cumulative counter.
    """

    now = time.monotonic()
    cached = _TRITON_COUNTER_CACHE.get(cache_key)
    _TRITON_COUNTER_CACHE[cache_key] = {"ts": now, "value": counter_value}

    if not cached:
        return 0.0

    delta_time = now - cached["ts"]
    delta_value = counter_value - cached["value"]

    if delta_time <= 0 or delta_value < 0:
        return 0.0

    return round(delta_value / delta_time, 2)


def _fetch_triton_service_health(timeout: float = 1.5) -> Dict[str, Any]:
    """
    Fetch Triton liveness and readiness endpoints.
    """

    base_url = get_triton_server_url()
    live = False
    ready = False
    detail = None

    try:
        live = requests.get(f"{base_url}/v2/health/live", timeout=timeout).ok
        ready = requests.get(f"{base_url}/v2/health/ready", timeout=timeout).ok
    except requests.RequestException as exc:
        detail = str(exc)

    return {
        "base_url": base_url,
        "live": live,
        "ready": ready,
        "status": "online" if live and ready else "degraded" if live or ready else "offline",
        "detail": detail,
    }


def _fetch_triton_model_ready_state(model_name: str, timeout: float = 1.5) -> bool:
    """
    Fetch Triton readiness for a single deployed model.
    """

    try:
        response = requests.get(f"{get_triton_server_url()}/v2/models/{model_name}/ready", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def _matches_scope(project: Project, user_id: int, deployment: Dict[str, Any], scope: str) -> bool:
    """
    Check whether a deployment belongs to the requested scope.
    """

    if scope != "mine":
        return True

    owner_id = deployment.get("deployed_by_user_id")
    if owner_id == user_id:
        return True

    if owner_id is None and project.created_by_id == user_id:
        return True

    return False


def _build_model_usage_stats(
    project: Project,
    user_id: int,
    scope: str,
    metrics_payload: str | None,
) -> Dict[str, Any]:
    """
    Build per-model usage stats from Triton deployments and Prometheus metrics.
    """

    raw_deployments = list_triton_model_deployments(project_ids=[project.id])
    deployments = [item for item in raw_deployments if _matches_scope(project, user_id, item, scope)]
    parsed_metrics = _parse_prometheus_metrics(metrics_payload or "")

    models = []
    total_success = 0.0
    total_failure = 0.0
    total_latency_us = 0.0
    total_rps = 0.0

    for deployment in deployments:
        model_name = deployment.get("model_name")
        success_count = _sum_metric_samples(
            parsed_metrics,
            "nv_inference_request_success",
            label_key="model",
            label_value=model_name,
        )
        failure_count = _sum_metric_samples(
            parsed_metrics,
            "nv_inference_request_failure",
            label_key="model",
            label_value=model_name,
        )
        request_duration_us = _sum_metric_samples(
            parsed_metrics,
            "nv_inference_request_duration_us",
            label_key="model",
            label_value=model_name,
        )
        request_count = success_count + failure_count
        avg_latency_ms = round((request_duration_us / request_count) / 1000, 2) if request_count > 0 else 0.0
        model_rps = _compute_counter_rate(
            f"project:{project.id}:scope:{scope}:model:{model_name}:success",
            success_count,
        )
        is_ready = _fetch_triton_model_ready_state(model_name) if model_name else False
        is_owned = deployment.get("deployed_by_user_id") == user_id or (
            deployment.get("deployed_by_user_id") is None and project.created_by_id == user_id
        )

        total_success += success_count
        total_failure += failure_count
        total_latency_us += request_duration_us
        total_rps += model_rps

        models.append(
            {
                **deployment,
                "name": model_name,
                "request_count": int(request_count),
                "success_count": int(success_count),
                "error_count": int(failure_count),
                "latency_ms": avg_latency_ms,
                "rps": model_rps,
                "ready": is_ready,
                "owned_by_current_user": is_owned,
                "status": "Online" if deployment.get("exists") and is_ready else "Offline",
            }
        )

    total_requests = total_success + total_failure
    average_latency_ms = round((total_latency_us / total_requests) / 1000, 2) if total_requests > 0 else 0.0
    success_rate = round((total_success / total_requests) * 100, 2) if total_requests > 0 else 100.0

    return {
        "deployments": deployments,
        "models": models,
        "summary": {
            "active_models": len([model for model in models if model.get("ready")]),
            "total_models": len(models),
            "request_count": int(total_requests),
            "rps": round(total_rps, 2),
            "latency": average_latency_ms,
            "success_rate": success_rate,
        },
    }


class ProjectTrainingModelsAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)

        root = _get_original_models_dir()
        root.mkdir(parents=True, exist_ok=True)

        models: List[Dict[str, Any]] = []
        for ext in ["*.pt", "*.pth"]:
            for p in sorted(root.glob(ext), key=lambda x: x.name.lower()):
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

        training_spec = None
        try:
            training_spec = detect_training_interface(project)
        except Exception:
            training_spec = None

        return Response({"models": models, "root": str(root), "training_spec": training_spec})


class ProjectTrainingJobsAPI(APIView):
    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)

        payload = request.data or {}
        base_weights = payload.get("base_weights")
        dataset_config = payload.get("dataset_config") or payload.get("data_yaml")
        epochs = int(payload.get("epochs", 50))
        imgsz = int(payload.get("imgsz", 640))
        batch = int(payload.get("batch", 16))

        if not base_weights:
            return Response({"detail": "base_weights is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not dataset_config:
            return Response({"detail": "dataset_config is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Fail fast with clear messages (so UI can show actionable errors)
        if os.path.isabs(str(base_weights)) and not Path(str(base_weights)).exists():
            return Response(
                {"detail": f"base_weights not found on server: {base_weights}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            dataset_meta = _read_dataset_config(str(dataset_config))
        except FileNotFoundError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        training_model = payload.get("training_model") or dataset_meta.get("training_model")
        if training_model not in {"yolo_detect", "yolo_classify", "cnn_classify"}:
            return Response(
                {"detail": f"Unsupported training_model in dataset_config: {training_model}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        output_root = _get_training_output_root()
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            queue = django_rq.get_queue("low")
            
            if training_model == "yolo_detect":
                job_func = yolo_detect_train_job
            elif training_model == "yolo_classify":
                job_func = yolo_classification_train_job
            elif training_model == "cnn_classify":
                job_func = cnn_classification_train_job
            else:
                 return Response({"detail": f"Unknown model type: {training_model}"}, status=400)

            job = queue.enqueue(
                job_func,
                kwargs={
                    "project_id": project.id,
                    "base_weights": base_weights,
                    "dataset_config": str(dataset_config),
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

        return Response({"job_id": job.id, "status": job.get_status(), "training_model": training_model})


class ProjectTrainingDatasetPrepareAPI(APIView):
    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)

        payload = request.data or {}
        train_ratio = float(payload.get("train_ratio", 0.8))
        seed = int(payload.get("seed", 42))
        export_format = payload.get("export_format")

        meta = prepare_training_dataset_for_project(
            project_id=project.id,
            train_ratio=train_ratio,
            seed=seed,
            export_format=export_format,
        )

        return Response(meta, status=status.HTTP_200_OK)


class ProjectTrainingJobDetailAPI(APIView):
    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, job_id: str, *args, **kwargs):
        project = _get_project_for_user(request, pk)

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
        project = _get_project_for_user(request, pk)

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

        project = _get_project_for_user(request, pk)

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
        project = _get_project_for_user(request, pk)

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
                        "task_type": (run_meta.get("params") or {}).get("task_type"),
                        "training_model": (run_meta.get("params") or {}).get("training_model"),
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
                        "task_type": (meta or {}).get("task_type"),
                        "training_model": (meta or {}).get("training_model"),
                        "dataset_config": (meta or {}).get("dataset_config"),
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

        project = _get_project_for_user(request, pk)

        file_name = request.query_params.get("file")
        if not file_name:
            return Response({"detail": "file query param is required"}, status=status.HTTP_400_BAD_REQUEST)

        base = _get_training_output_root() / f"project_{project.id}" / run_id / "artifacts"
        target = base / file_name
        if not target.exists():
            raise Http404

        return FileResponse(open(target, "rb"), as_attachment=True, filename=target.name)


class ProjectTrainingRunDeployToTritonAPI(APIView):
    """
    Copy a training run's best.pt into Triton model repository and write config files.
    No ML Backend is required for this deployment flow.
    """

    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, run_id: str, *args, **kwargs):
        project = _get_project_for_user(request, pk)

        output_root = _get_training_output_root()
        run_dir = output_root / f"project_{project.id}" / run_id
        artifacts_dir = run_dir / "artifacts"
        best_pt = artifacts_dir / "best.pt"

        if not best_pt.exists():
            return Response(
                {"detail": "No best.pt found for this run. Train the model first."},
                status=status.HTTP_404_NOT_FOUND,
            )

        payload = request.data or {}
        raw_name = (payload.get("model_name") or f"ls_project_{project.id}_{run_id}").strip()
        model_name = sanitize_triton_model_name(raw_name, fallback=f"ls_project_{project.id}_run")

        # Determine model type from run_meta.json
        run_meta = {}
        try:
            run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass

        is_cnn = run_meta.get("kind") == "cnn_classification_train" or \
                 run_meta.get("params", {}).get("training_model") == "cnn_classify"

        if is_cnn:
            # CNN trainer always uses imgsz=224 for ResNet18
            imgsz = int(payload.get("imgsz", 224))
            result = export_torchscript_pt_to_triton(
                best_pt_path=str(best_pt),
                model_name=model_name,
                project_id=project.id,
                run_id=run_id,
                imgsz=imgsz,
                deployed_by_user_id=request.user.id,
                deployed_by_username=request.user.username,
            )
        else:
            imgsz = int(payload.get("imgsz", 640))
            result = export_yolo_pt_to_triton(
                best_pt_path=str(best_pt),
                model_name=model_name,
                project_id=project.id,
                run_id=run_id,
                triton_repo_root=None,
                imgsz=imgsz,
                deployed_by_user_id=request.user.id,
                deployed_by_username=request.user.username,
            )

        if result.get("error"):
            return Response(
                {"detail": result["error"]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "message": "Model copied to Triton repository.",
                "triton_repo_root": str(get_triton_model_repository_root()),
                "triton_server_url": get_triton_server_url(),
                **result,
            },
            status=status.HTTP_200_OK,
        )


class ProjectTrainingTritonModelsAPI(APIView):
    """
    List Triton models that were deployed from training runs in the current project.
    """

    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)
        scope = (request.query_params.get("scope") or "project").strip().lower()
        deployments = list_triton_model_deployments(project_ids=[project.id])
        deployments = [item for item in deployments if _matches_scope(project, request.user.id, item, scope)]

        for item in deployments:
            item["owned_by_current_user"] = item.get("deployed_by_user_id") == request.user.id or (
                item.get("deployed_by_user_id") is None and project.created_by_id == request.user.id
            )

        return Response(
            {
                "project_id": project.id,
                "scope": scope,
                "triton_repo_root": str(get_triton_model_repository_root()),
                "triton_server_url": get_triton_server_url(),
                "models": deployments,
            },
            status=status.HTTP_200_OK,
        )


class ProjectTrainingTritonInferAPI(APIView):
    """
    Proxy a Triton v2 infer request for a project-owned deployed model.
    """

    permission_required = ViewClassPermission(POST=all_permissions.projects_view)

    def post(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)
        payload = dict(request.data or {})
        
        # Verify API Key if project has one configured
        api_key = payload.pop("api_key", None)
        if project.triton_api_key:
            if not api_key or api_key != project.triton_api_key:
                return Response(
                    {"error": "Invalid or missing API key for this project's Triton deployments."},
                    status=status.HTTP_403_FORBIDDEN,
                )

        model_name = sanitize_triton_model_name(payload.pop("model_name", ""))

        if not model_name:
            return Response({"detail": "model_name is required"}, status=status.HTTP_400_BAD_REQUEST)

        deployed_models = {
            item["model_name"]: item for item in list_triton_model_deployments(project_ids=[project.id])
        }
        if model_name not in deployed_models:
            return Response(
                {"detail": f"Model {model_name} is not deployed for this project."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if "inputs" not in payload:
            return Response(
                {"detail": "Triton infer payload must include inputs."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        triton_url = f"{get_triton_server_url()}/v2/models/{model_name}/infer"
        timeout = float(request.query_params.get("timeout", 60))

        try:
            triton_response = requests.post(triton_url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            return Response(
                {
                    "detail": f"Failed to reach Triton server: {exc}",
                    "triton_url": triton_url,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            response_body = triton_response.json()
        except ValueError:
            response_body = triton_response.text

        return Response(
            {
                "model_name": model_name,
                "triton_url": triton_url,
                "ok": triton_response.ok,
                "status_code": triton_response.status_code,
                "body": response_body,
            },
            status=status.HTTP_200_OK if triton_response.ok else triton_response.status_code,
        )

class ProjectTrainingModelUploadAPI(APIView):
    """
    Upload a custom .pt (TorchScript/YOLO) model file and deploy it to Triton.
    """

    permission_required = ViewClassPermission(POST=all_permissions.projects_change)

    def post(self, request, pk: int, *args, **kwargs):
        project = _get_project_for_user(request, pk)
        
        if "file" not in request.FILES:
            return Response({"detail": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES["file"]
        model_name = request.data.get("model_name")
        if not model_name:
            # Fallback to filename without extension
            model_name = Path(uploaded_file.name).stem
            
        imgsz = int(request.data.get("imgsz", 640))
        
        # Save temporary
        temp_dir = _get_training_output_root() / f"project_{project.id}" / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
                
        # Deploy
        # We try to detect if it's YOLO or generic TorchScript
        # For safety/simplicity, if it has 'yolo' in name, we might use yolo exporter, 
        # but generic TorchScript exporter is safer for uploaded .pt
        result = export_torchscript_pt_to_triton(
            best_pt_path=str(temp_path),
            model_name=model_name,
            project_id=project.id,
            run_id="manual_upload",
            imgsz=imgsz,
            deployed_by_user_id=request.user.id,
            deployed_by_username=request.user.username,
        )
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
            
        if result.get("error"):
            return Response({"detail": result["error"]}, status=status.HTTP_400_BAD_REQUEST)
            
        return Response(
            {
                "message": "Model uploaded and deployed to Triton.",
                "triton_server_url": get_triton_server_url(),
                **result,
            },
            status=status.HTTP_201_CREATED,
        )


class ProjectTrainingMetricsAPI(APIView):
    """
    Fetch real-time hardware and Triton inference metrics.
    """

    permission_required = ViewClassPermission(GET=all_permissions.projects_view)

    def get(self, request, pk: int, *args, **kwargs):
        import psutil

        project = _get_project_for_user(request, pk)
        scope = (request.query_params.get("scope") or "project").strip().lower()
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        ram_used_gb = round(ram.used / (1024**3), 2)
        triton_health = _fetch_triton_service_health()
        metrics_payload = None
        metrics_available = False
        parsed_metrics: Dict[str, List[Dict[str, Any]]] = {}
        try:
            metrics_res = requests.get("http://localhost:8002/metrics", timeout=2)
            if metrics_res.ok:
                metrics_payload = metrics_res.text
                parsed_metrics = _parse_prometheus_metrics(metrics_payload)
                metrics_available = True
        except requests.RequestException:
            pass

        gpu_util_samples = [_safe_float(sample.get("value")) for sample in parsed_metrics.get("nv_gpu_utilization", [])]
        gpu_usage = round(sum(gpu_util_samples) / len(gpu_util_samples), 2) if gpu_util_samples else 0.0
        vram_used = _sum_metric_samples(parsed_metrics, "nv_gpu_memory_used_bytes") / (1024**3)
        vram_total = _sum_metric_samples(parsed_metrics, "nv_gpu_memory_total_bytes") / (1024**3)
        usage_stats = _build_model_usage_stats(project, request.user.id, scope, metrics_payload)
        inference_summary = usage_stats["summary"]

        return Response(
            {
                "scope": scope,
                "viewer": {
                    "id": request.user.id,
                    "username": request.user.username,
                },
                "health": {
                    **triton_health,
                    "metrics_available": metrics_available,
                },
                "hardware": {
                    "cpu": round(cpu_usage, 2),
                    "ram": round(ram_usage, 2),
                    "ram_used_gb": ram_used_gb,
                    "gpu": gpu_usage,
                    "vram_used_gb": round(vram_used, 2),
                    "vram_total_gb": round(vram_total, 2),
                },
                "inference": inference_summary,
                "models": usage_stats["models"],
            }
        )
