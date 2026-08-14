# Quick Start

## Prerequisites

- **Docker** with Compose v2

## Run the stack (Docker)

```bash
make build-docker   # build the image (bakes the Echo-DSRN model)
docker compose up -d
```

This starts:

- **postgres** — the annotation database (`echo_dsrn`, internal to the stack, not exposed on the host)
- **redis** — Celery broker
- **web** — FastAPI + Vue app on `http://localhost:7860`
- **worker-cpu** — Celery worker for async classification/judging

On first start the web container runs `alembic upgrade head` automatically, so the schema is always up to date.

App serves on `http://localhost:7860`.

## Database

The stack uses PostgreSQL. To migrate manually, or run the stack without Docker:

```bash
make migrate       # uv run alembic -c backend/alembic.ini upgrade head
docker compose exec postgres psql -U echo -d echo_dsrn   # inspect the DB
```

## Makefile

```bash
make help          # list all targets
make up            # docker compose up -d
make down          # docker compose down
make build-docker  # build the Docker image
make test          # run pytest
make lint          # pre-commit hooks
make migrate       # alembic upgrade head
make judge N=100   # run LLM-as-Judge (sync)
make judge-async N=100  # dispatch to Celery worker
```
