"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
from django.urls import include, path

from . import api, views
from training import api as training_api

app_name = 'projects'

# reverse for projects:name
_urlpatterns = [
    path('', views.project_list, name='project-index'),
    path('<int:pk>/settings/', views.project_settings, name='project-settings', kwargs={'sub_path': ''}),
    path('<int:pk>/settings/<sub_path>', views.project_settings, name='project-settings-anything'),
]

# reverse for projects:api:name
_api_urlpatterns = [
    # CRUD
    path('', api.ProjectListAPI.as_view(), name='project-list'),
    path('<int:pk>/', api.ProjectAPI.as_view(), name='project-detail'),
    path('counts/', api.ProjectCountsListAPI.as_view(), name='project-counts-list'),
    # Get next task
    path('<int:pk>/next/', api.ProjectNextTaskAPI.as_view(), name='project-next'),
    # Label stream history
    path('<int:pk>/label-stream-history/', api.LabelStreamHistoryAPI.as_view(), name='label-stream-history'),
    # Validate label config in general
    path('validate/', api.LabelConfigValidateAPI.as_view(), name='label-config-validate'),
    # Validate label config for project
    path('<int:pk>/validate/', api.ProjectLabelConfigValidateAPI.as_view(), name='project-label-config-validate'),
    # Project summary
    path('<int:pk>/summary/', api.ProjectSummaryAPI.as_view(), name='project-summary'),
    # Project summary
    path(
        '<int:pk>/summary/reset/',
        api.ProjectSummaryResetAPI.as_view(),
        name='project-summary-reset',
    ),
    # Project import
    path('<int:pk>/imports/<int:import_pk>/', api.ProjectImportAPI.as_view(), name='project-imports'),
    # Project reimport
    path('<int:pk>/reimports/<int:reimport_pk>/', api.ProjectReimportAPI.as_view(), name='project-reimports'),
    # Tasks list for the project: get and destroy
    path('<int:pk>/tasks/', api.ProjectTaskListAPI.as_view(), name='project-tasks-list'),
    # Generate sample task for this project
    path('<int:pk>/sample-task/', api.ProjectSampleTask.as_view(), name='project-sample-task'),
    # List available model versions
    path('<int:pk>/model-versions/', api.ProjectModelVersions.as_view(), name='project-model-versions'),
    # List all annotators for project
    path('<int:pk>/annotators/', api.ProjectAnnotatorsAPI.as_view(), name='project-annotators'),
    # On-server training (YOLO detect)
    path('<int:pk>/training/models/', training_api.ProjectTrainingModelsAPI.as_view(), name='project-training-models'),
    path('<int:pk>/training/jobs/', training_api.ProjectTrainingJobsAPI.as_view(), name='project-training-jobs'),
    path(
        '<int:pk>/training/jobs/<str:job_id>/',
        training_api.ProjectTrainingJobDetailAPI.as_view(),
        name='project-training-job-detail',
    ),
    path(
        '<int:pk>/training/jobs/<str:job_id>/artifacts/',
        training_api.ProjectTrainingJobArtifactsAPI.as_view(),
        name='project-training-job-artifacts',
    ),
    path(
        '<int:pk>/training/jobs/<str:job_id>/download',
        training_api.ProjectTrainingJobDownloadAPI.as_view(),
        name='project-training-job-download',
    ),
    path(
        '<int:pk>/training/datasets/prepare/',
        training_api.ProjectTrainingDatasetPrepareAPI.as_view(),
        name='project-training-dataset-prepare',
    ),
    path(
        '<int:pk>/training/history/',
        training_api.ProjectTrainingHistoryAPI.as_view(),
        name='project-training-history',
    ),
    path(
        '<int:pk>/training/runs/<str:run_id>/download',
        training_api.ProjectTrainingRunDownloadAPI.as_view(),
        name='project-training-run-download',
    ),
    path(
        '<int:pk>/training/runs/<str:run_id>/deploy-to-triton',
        training_api.ProjectTrainingRunDeployToTritonAPI.as_view(),
        name='project-training-run-deploy-to-triton',
    ),
    path(
        '<int:pk>/training/triton/models/',
        training_api.ProjectTrainingTritonModelsAPI.as_view(),
        name='project-training-triton-models',
    ),
    path(
        '<int:pk>/training/triton/health/',
        training_api.ProjectTrainingTritonHealthAPI.as_view(),
        name='project-training-triton-health',
    ),
    path(
        '<int:pk>/training/triton/infer/',
        training_api.ProjectTrainingTritonInferAPI.as_view(),
        name='project-training-triton-infer',
    ),
    path(
        '<int:pk>/training/upload',
        training_api.ProjectTrainingModelUploadAPI.as_view(),
        name='project-training-upload',
    ),
    path(
        '<int:pk>/training/metrics',
        training_api.ProjectTrainingMetricsAPI.as_view(),
        name='project-training-metrics',
    ),
    path(
        '<int:pk>/training/metrics/history',
        training_api.ProjectTrainingMetricsHistoryAPI.as_view(),
        name='project-training-metrics-history',
    ),
]

_api_urlpatterns_templates = [
    path('', api.TemplateListAPI.as_view(), name='template-list'),
]


urlpatterns = [
    path('projects/', include(_urlpatterns)),
    path('api/projects/', include((_api_urlpatterns, app_name), namespace='api')),
    path('api/templates/', include((_api_urlpatterns_templates, app_name), namespace='api-templates')),
]
