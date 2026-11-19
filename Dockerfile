# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — frontend bundle
#
# Node, npm and every package they drag in (esbuild, vite, postcss, ...) stay in
# this stage; only frontend/dist is copied into the runtime image.
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build

# HuggingFace datasets collection URL (injected at build time for the UI)
ARG VITE_HF_DATASETS_URL="https://huggingface.co/collections/ethicalabs/openaire-ai-hackathon-2026"
ENV VITE_HF_DATASETS_URL=$VITE_HF_DATASETS_URL

COPY frontend/package.json ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python dependencies in a self-contained venv
#
# Compilers, git and the uv binary live here only; the runtime gets /opt/venv.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

WORKDIR /app

RUN uv venv /opt/venv --python 3.12

# Install from the hash-pinned lock so the runtime set is reproducible and
# auditable (`make lock` regenerates it). The +cpu local version of torch only
# exists on PyTorch's CPU index, so both indexes are passed explicitly.
COPY requirements.txt requirements.lock ./
RUN uv pip install --python /opt/venv/bin/python \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --require-hashes -r requirements.lock

# Install the Echo-DSRN library without its training/eval dependency bloat
# (lm-eval, trl, wandb, sentence-transformers are declared by the upstream
# repo but unused at inference time — its __init__ only imports transformers).
RUN uv pip install --python /opt/venv/bin/python --no-deps \
    "git+https://github.com/ethicalabs-ai/Echo-DSRN.git"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Pull the current Debian security updates over whatever the base tag shipped,
# then add only what torch needs at runtime (libgomp for OpenMP, libstdc++).
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY . .
COPY --from=frontend /build/dist ./frontend/dist

# Pre-download the Echo-DSRN intent classifier model (avoids runtime HF download)
# HF_TOKEN is consumed natively by huggingface_hub / transformers
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") \
    python -c "\
from echo_dsrn import EchoForSequenceClassification; \
from transformers import AutoTokenizer; \
model_id = 'ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF'; \
EchoForSequenceClassification.from_pretrained(model_id, trust_remote_code=True); \
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
print('Model cached at', __import__('os').environ['HF_HOME'])"

# Set default model path to the HF cache so the app doesn't re-download
ENV INTENT_CLF_PATH=ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF

# Active collab annotation round (overridable at runtime via env/helm)
ENV MODEL_VERSION=v0.1.4

# Ensure entrypoint is executable
RUN chmod +x docker_entrypoint.sh

# Expose the default Gradio port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]
