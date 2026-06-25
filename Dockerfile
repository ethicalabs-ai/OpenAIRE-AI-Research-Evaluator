# Use a lightweight Python base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies & Node.js for Vite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# HuggingFace cache location (baked into image)
ENV HF_HOME=/app/hf_cache

# HuggingFace token for gated model access (optional build arg)
ARG HF_TOKEN

# Copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Pre-download the Echo-DSRN intent classifier model (avoids runtime HF download)
# HF_TOKEN is consumed natively by huggingface_hub / transformers
RUN HF_TOKEN=${HF_TOKEN} python -c "\
from echo_dsrn import EchoForSequenceClassification; \
from transformers import AutoTokenizer; \
model_id = 'ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF'; \
EchoForSequenceClassification.from_pretrained(model_id, trust_remote_code=True); \
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
print('Model cached at', __import__('os').environ['HF_HOME'])"

# Set default model path to the HF cache so the app doesn't re-download
ENV INTENT_CLF_PATH=ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF

# Copy the entire application code
COPY . .

# Build the frontend locally
RUN cd frontend && npm ci && npm run build

# Ensure entrypoint is executable
RUN chmod +x docker_entrypoint.sh

# Expose the default Gradio port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]
