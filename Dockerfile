# syntax=docker/dockerfile:1
ARG NODE_VERSION=22
# yolo-training（ultralytics/torch）之 manylinux 輪子需 glibc；Alpine/musl 會無法安裝 CUDA/torch 等依賴
ARG PYTHON_VERSION=3.12
ARG POETRY_VERSION=2.3.2
ARG VERSION_OVERRIDE
ARG BRANCH_OVERRIDE

################################ Overview

# This Dockerfile builds a Label Studio environment.
# It consists of five main stages:
# 1. "frontend-builder" - Compiles the frontend assets using Node.
# 2. "frontend-version-generator" - Generates version files for frontend sources.
# 3. "venv-builder" - Prepares the virtualenv environment.
# 4. "py-version-generator" - Generates version files for python sources.
# 5. "prod" - Creates the final production image with the Label Studio, Nginx, and other dependencies.

################################ Stage: frontend-builder (build frontend assets)
FROM --platform=${BUILDPLATFORM} node:${NODE_VERSION}-alpine AS frontend-builder
ENV BUILD_NO_SERVER=true \
    BUILD_NO_HASH=true \
    BUILD_NO_CHUNKS=true \
    BUILD_MODULE=true \
    YARN_CACHE_FOLDER=/root/web/.yarn \
    NX_CACHE_DIRECTORY=/root/web/.nx \
    NODE_ENV=production \
    NODE_OPTIONS="--max-old-space-size=4096"

WORKDIR /label-studio/web

RUN apk add --no-cache \
    build-base \
    pkgconfig \
    cairo-dev \
    giflib-dev \
    libjpeg-turbo-dev \
    libpng-dev \
    pango-dev \
    git \
    python3

COPY web/package.json .
COPY web/yarn.lock .
COPY web/tools tools
# --network-timeout：大專案拉 tarball 較久；迴圈重試：緩解 registry 502（yarnpkg 短暫故障）
RUN --mount=type=cache,target=/root/web/.yarn,id=yarn-cache,sharing=locked \
    --mount=type=cache,target=/root/web/.nx,id=nx-cache,sharing=locked \
    set -e; \
    ok=0; \
    for attempt in 1 2 3 4 5; do \
      if yarn install --prefer-offline --no-progress --pure-lockfile --frozen-lockfile --ignore-engines --non-interactive --production=false --network-timeout 600000; then ok=1; break; fi; \
      echo "yarn install failed (attempt $attempt/5), retrying in 25s..."; \
      sleep 25; \
    done; \
    test "$ok" -eq 1

COPY web/ .
COPY pyproject.toml ../pyproject.toml
RUN --mount=type=cache,target=/root/web/.yarn,id=yarn-cache,sharing=locked \
    --mount=type=cache,target=/root/web/.nx,id=nx-cache,sharing=locked \
    yarn run build

################################ Stage: frontend-version-generator
FROM frontend-builder AS frontend-version-generator
RUN --mount=type=cache,target=/root/web/.yarn,id=yarn-cache,sharing=locked \
    --mount=type=cache,target=/root/web/.nx,id=nx-cache,sharing=locked \
    --mount=type=bind,source=.git,target=../.git \
    yarn version:libs

################################ Stage: venv-builder (prepare the virtualenv)
FROM python:${PYTHON_VERSION}-slim-bookworm AS venv-builder
ARG POETRY_VERSION
ARG PYTHON_VERSION

# PIP_DEFAULT_TIMEOUT / PIP_RETRIES：隔離建置（如 opencv-python-headless）會再拉 PyPI，網路慢時易逾時
# POETRY_INSTALLER_PARALLEL=false：改為序列安裝，減少同時連線、降低逾時機率
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=10 \
    PIP_CACHE_DIR="/.cache" \
    POETRY_CACHE_DIR="/.poetry-cache" \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON=true \
    POETRY_INSTALLER_PARALLEL=false \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpcre3-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ADD https://install.python-poetry.org /tmp/install-poetry.py
RUN python /tmp/install-poetry.py

WORKDIR /label-studio

ENV VENV_PATH="/label-studio/.venv"
ENV PATH="$VENV_PATH/bin:$PATH"

## Starting from this line all packages will be installed in $VENV_PATH

# Copy dependency files
COPY pyproject.toml poetry.lock README.md ./

# Set a default build argument for including dev dependencies
ARG INCLUDE_DEV=false

# Install dependencies
# lock 會解析出含 CUDA 的 torch；slim 映像無系統 libcudnn，import 會報錯。
# 安裝後強制改為 PyTorch 官方 CPU 輪子（較小、無 cuDNN），訓練可用 CPU；若要 GPU 請改用 nvidia/cuda 基底映像並勿覆寫此步驟。
RUN --mount=type=cache,target=/.poetry-cache,id=poetry-cache-bookworm,sharing=locked \
    set -e; \
    poetry check --lock; \
    set +e; \
    if [ "${INCLUDE_DEV:-false}" = "true" ]; then \
        poetry install --no-root --extras uwsgi --extras yolo-training --with test; \
    else \
        poetry install --no-root --without test --extras uwsgi --extras yolo-training; \
    fi; \
    poetry_ec=$?; \
    set -e; \
    if [ "$poetry_ec" -ne 0 ]; then \
        echo "note: poetry install exited $poetry_ec（常見：下載逾時／yanked）；仍嘗試覆寫 CPU torch 並驗證"; \
    fi; \
    /label-studio/.venv/bin/pip install --no-cache-dir --force-reinstall --upgrade \
        "torch" "torchvision" \
        --index-url https://download.pytorch.org/whl/cpu; \
    /label-studio/.venv/bin/python -c "import django, torch, ultralytics; print('deps OK', django.get_version(), torch.__version__, 'cuda=', torch.cuda.is_available())"

# Install LS（再跑一次 --no-root：補齊因上一層快取／曾失敗導致缺 Django 等依賴的 venv）
COPY label_studio label_studio
RUN --mount=type=cache,target=/.poetry-cache,id=poetry-cache-bookworm,sharing=locked \
    set -e; \
    if [ "${INCLUDE_DEV:-false}" = "true" ]; then \
        poetry install --no-root --extras uwsgi --extras yolo-training --with test; \
    else \
        poetry install --no-root --without test --extras uwsgi --extras yolo-training; \
    fi; \
    poetry install --only-root --extras uwsgi --extras yolo-training; \
    /label-studio/.venv/bin/python -c "import django"; \
    /label-studio/.venv/bin/python label_studio/manage.py collectstatic --no-input

################################ Stage: py-version-generator
FROM venv-builder AS py-version-generator
ARG VERSION_OVERRIDE
ARG BRANCH_OVERRIDE

# Create version_.py and ls-version_.py
RUN --mount=type=bind,source=.git,target=./.git \
    VERSION_OVERRIDE=${VERSION_OVERRIDE} BRANCH_OVERRIDE=${BRANCH_OVERRIDE} poetry run python label_studio/core/version.py

################################### Stage: prod
FROM python:${PYTHON_VERSION}-slim-bookworm AS production

ENV LS_DIR=/label-studio \
    HOME=/label-studio \
    LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data \
    OPT_DIR=/opt/heartex/instance-data/etc \
    PATH="/label-studio/.venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=core.settings.label_studio \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $LS_DIR

# 執行期依賴：nginx、redis-server（EMBEDDED_REDIS=1）、opencv/torch 常用系統庫
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    libexpat1 \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nginx \
    procps \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p $LS_DIR $LABEL_STUDIO_BASE_DATA_DIR $OPT_DIR && \
    chown -R 1001:0 $LS_DIR $LABEL_STUDIO_BASE_DATA_DIR $OPT_DIR /var/log/nginx /etc/nginx

COPY --chown=1001:0 deploy/default.conf /etc/nginx/nginx.conf

# Copy essential files for installing Label Studio and its dependencies
COPY --chown=1001:0 pyproject.toml .
COPY --chown=1001:0 poetry.lock .
COPY --chown=1001:0 README.md .
COPY --chown=1001:0 LICENSE LICENSE
COPY --chown=1001:0 licenses licenses
COPY --chown=1001:0 deploy deploy

# Copy files from build stages
COPY --chown=1001:0 --from=venv-builder               $LS_DIR                                           $LS_DIR
COPY --chown=1001:0 --from=py-version-generator       $LS_DIR/label_studio/core/version_.py             $LS_DIR/label_studio/core/version_.py
COPY --chown=1001:0 --from=frontend-builder           $LS_DIR/web/dist                                  $LS_DIR/web/dist
COPY --chown=1001:0 --from=frontend-version-generator $LS_DIR/web/dist/apps/labelstudio/version.json    $LS_DIR/web/dist/apps/labelstudio/version.json
COPY --chown=1001:0 --from=frontend-version-generator $LS_DIR/web/dist/libs/editor/version.json         $LS_DIR/web/dist/libs/editor/version.json
COPY --chown=1001:0 --from=frontend-version-generator $LS_DIR/web/dist/libs/datamanager/version.json    $LS_DIR/web/dist/libs/datamanager/version.json

USER 1001

EXPOSE 8080

ENTRYPOINT ["./deploy/docker-entrypoint.sh"]
CMD ["label-studio"]
