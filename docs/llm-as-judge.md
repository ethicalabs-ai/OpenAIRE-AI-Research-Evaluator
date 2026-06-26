# LLM-as-Judge

The LLM-as-Judge pipeline evaluates Echo-DSRN predictions against high-capability LLMs and produces a golden consensus dataset in `data/collaborative.db`.

## Search Queries

arXiv and OpenAIRE sources use keyword queries from `assets/topics.txt` — one query per line, blank lines and `#` comments ignored. Edit this file to target specific research domains.

Container path: `/app/assets/topics.txt`

## Setup

Point at a lemonade or llama.cpp server via `.env`:

```env
LLM_BASE_URL=http://192.168.1.40:13305/v1
```

Migrate the database:

```bash
uv run alembic -c backend/alembic.ini upgrade head
```

## LLM Server Setup

Any OpenAI-compatible server works. Configuration via `.env`:

```env
LLM_BASE_URL=<server-url>
LLM_API_KEY=not-needed     # for local servers that skip auth
```

### Ollama

```bash
ollama pull qwen3:4b
ollama serve
```

```env
LLM_BASE_URL=http://localhost:11434/v1
```

### LM Studio

Start the local server in the UI (Developer → Local Server). Default port:

```env
LLM_BASE_URL=http://localhost:1234/v1
```

### Lemonade / llama.cpp

```bash
lemonade --host 0.0.0.0 load --pinned Qwen3.6-35B-A3B-GGUF
```

```env
LLM_BASE_URL=http://192.168.1.40:13305/v1
```

### Docker — LLM on host

When running the app in Docker and the LLM server on the host, use `host.docker.internal`:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
```

The `docker-compose.yaml` already includes `extra_hosts: host.docker.internal:host-gateway`.

## Batch Runner — Sync

Loads models on lemonade, judges papers inline:

```bash
./scripts/run_judges.sh 100
```

Cycles through 11 GGUF models (Qwen3.6, Nemotron, Gemma, GPT-OSS, DeepSeek, GLM) against the catalog source.

## Batch Runner — Async

Dispatches Celery tasks to the worker, no model loading:

```bash
./scripts/run_judges.sh 100 --async
```

## Kubernetes (k3s)

Run the judge CLI directly in the deployed web pod:

```bash
kubectl exec -it deployment/evaluator-openaire-research-evaluator-web -- \
  env LLM_BASE_URL=http://192.168.1.66:13305/v1 \
  python backend/judge_cli.py \
    --source catalog --n 100 \
    --model gpt-oss-20b-mxfp4-GGUF \
    --async
```

Tasks dispatch to the cluster's Celery worker, using the same Redis and PostgreSQL.

## Single Run

```bash
# Local GGUF via llama.cpp / lemonade
uv run python backend/judge_cli.py \
    --source catalog --n 100 \
    --model Qwen3.6-35B-A3B-GGUF

# Google Gemini (OpenAI-compatible endpoint)
uv run python backend/judge_cli.py \
    --source catalog --n 100 \
    --model gemini-2.5-flash \
    --llm-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --llm-key ${GEMINI_API_KEY}
```

## Pre-labeled Datasets

Convert chat-format JSONL to flat records, then judge:

```bash
uv run python scripts/reformat_for_judge.py \
    ~/.ethicalabs/datasets/research-intent/train.jsonl \
    ~/.ethicalabs/datasets/research-intent/train_flat.jsonl

uv run python backend/judge_cli.py \
    --source dataset \
    --dataset-path ~/.ethicalabs/datasets/research-intent/train_flat.jsonl \
    --n 100 --model Qwen3.6-35B-A3B-GGUF
```

## Evaluation

```bash
uv run python backend/judge_eval.py                # all papers
uv run python backend/judge_eval.py --paper-source dataset  # golden set only
```

Computes agreement rates, class-level accuracy, and per-model alignment stats.
