# syntax=docker/dockerfile:1
# Label Studio Train Server — 獨立 YOLO 訓練容器（含 ultralytics / torch）
# 建置：docker compose -f train-server/docker-compose.yml build
# DGX Spark（aarch64 / CUDA 13）：在 Spark 本機 build，預設 cu130 wheel

ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130

FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRAIN_SERVER_HOST=0.0.0.0 \
    TRAIN_SERVER_PORT=8011 \
    TRAIN_SERVER_OUTPUT_ROOT=/data/training/models/trained \
    TRAIN_SERVER_MODELS_DIR=/data/training/models/original \
    TRAIN_SERVER_MAX_WORKERS=1 \
    TRAIN_SERVER_DATALOADER_WORKERS=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY train_server/requirements-train.txt /app/train_server/requirements-train.txt
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --upgrade pip \
    && pip install "numpy>=1.26.4,<2.0" \
        "torch>=2.9.0" "torchvision>=0.24.0" \
        --index-url "${TORCH_CUDA_INDEX_URL}" \
    && pip install -r /app/train_server/requirements-train.txt

COPY train_server /app/train_server

EXPOSE 8011

VOLUME ["/data"]

CMD ["python", "-m", "train_server.main"]
