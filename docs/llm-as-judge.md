# LLM-as-Judge

The LLM-as-Judge pipeline evaluates Echo-DSRN predictions against high-capability LLMs and produces a golden consensus dataset in `data/collaborative.db`.

## Setup

Point at a lemonade or llama.cpp server via `.env`:

```env
LLM_BASE_URL=http://192.168.1.40:13305/v1
```

Migrate the database:

```bash
uv run alembic -c backend/alembic.ini upgrade head
```

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
uv run python backend/reformat_for_judge.py \
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
