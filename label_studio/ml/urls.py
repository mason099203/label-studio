"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
from django.urls import include, path

from . import api

app_name = 'ml'

# ML backend CRUD
_api_urlpatterns = [
    # All ml backends
    path('', api.MLBackendListAPI.as_view(), name='ml-list'),
    path('<int:pk>', api.MLBackendDetailAPI.as_view(), name='ml-detail'),
    path('<int:pk>/train', api.MLBackendTrainAPI.as_view(), name='ml-train'),
    path('<int:pk>/predict/test', api.MLBackendPredictTestAPI.as_view(), name='ml-predict-test'),
    path(
        '<int:pk>/interactive-annotating',
        api.MLBackendInteractiveAnnotating.as_view(),
        name='ml-interactive-annotating',
    ),
    path('<int:pk>/versions', api.MLBackendVersionsAPI.as_view(), name='ml-versions'),
]

_deployment_urlpatterns = [
    path('', api.ModelDeploymentListAPI.as_view(), name='deployment-list'),
    path('predict/', api.ModelDeploymentPredictByKeyAPI.as_view(), name='deployment-predict'),
    path('<int:pk>', api.ModelDeploymentDetailAPI.as_view(), name='deployment-detail'),
]

urlpatterns = [
    path('api/ml/', include((_api_urlpatterns, app_name), namespace='api')),
    path('api/deployments/', include((_deployment_urlpatterns, app_name), namespace='deployments')),
]
