# Quick Start

## Prerequisites

- **Python** ≥ 3.12 (tested with 3.12.3)
- **Node.js** ≥ 22 (tested with 22.22.3)
- **npm** ≥ 10 (tested with 10.9.8)
- **Docker** (optional, for containerised deployment)

## Docker Compose

```bash
docker compose up -d
```

App serves on `http://localhost:7860`. Redis, the web server, and a Celery worker start automatically.

Set `HF_TOKEN` in `.env` for gated model access during the Docker build (see [`.env.example`](https://github.com/ethicalabs-ai/OpenAIRE-AI-Research-Evaluator/blob/main/.env.example)).

## Python

### uv

```bash
uv run python backend/server.py
```

### pip

```bash
pip install -r requirements.txt
python backend/server.py
```

Build the frontend first: `cd frontend && npm ci && npm run build`. App serves on `http://localhost:7860`.

## Makefile

```bash
make help          # list all targets
make up            # docker compose up -d
make build-docker  # build the Docker image
make test          # run pytest
make lint          # pre-commit hooks
make migrate       # alembic upgrade head
make judge N=100   # run LLM-as-Judge (sync)
make judge-async N=100  # dispatch to Celery worker
```
