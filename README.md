# OpenAIRE-AI-Research-Evaluator

A multi-model LLM-as-Judge pipeline that builds annotation datasets to train Echo-DSRN intent classifiers — entry for the [OpenAIRE AI Hackathon 2026](https://innovation.openaire.eu/component/content/article/openaire-ai-hackathon.html), co-organised by OpenAIRE and Alien Intelligence.

---

## Quick Start — Docker Compose

**Prerequisites:** Python ≥ 3.12, Node.js ≥ 22, npm ≥ 10, Docker

The Echo-DSRN classifier is a gated HuggingFace model. Before building:

1. [Request access](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) to the model repo
2. Create an [HF access token](https://huggingface.co/settings/tokens) with read permissions
3. Add it to `.env`: `HF_TOKEN=hf_...`

```bash
make rebuild      # builds Docker image with the gated model
docker compose up -d
```

App serves on `http://localhost:7860`.

---

## Quick Start — Python

### uv

```bash
uv run python backend/server.py
```

### pip

```bash
pip install -r requirements.txt
python backend/server.py
```

Frontend must be pre-built (`cd frontend && npm ci && npm run build`). App serves on `http://localhost:7860`.

---

## LLM-as-Judge

The LLM-as-Judge pipeline evaluates Echo-DSRN predictions against high-capability LLMs and produces a golden consensus dataset in `data/collaborative.db`.

### 1. Migrate the database

```bash
uv run alembic -c backend/alembic.ini upgrade head
```

### 2. Run judges

Point at a lemonade or llama.cpp server via `.env`:

```env
LLM_BASE_URL=http://<llm-server-host>:<llm-server-port>/v1
```

**Batch runner** — cycles through all models and sources (via the Makefile):

```bash
make judge N=100          # sync
make judge-async N=100    # async, via Celery
```

**Single run** — one model, one source:

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

**With pre-labeled datasets** — convert chat-format JSONL to flat records, then judge:

```bash
uv run python scripts/reformat_for_judge.py \
    ~/.ethicalabs/datasets/research-intent/train.jsonl \
    ~/.ethicalabs/datasets/research-intent/train_flat.jsonl

uv run python backend/judge_cli.py \
    --source dataset \
    --dataset-path ~/.ethicalabs/datasets/research-intent/train_flat.jsonl \
    --n 100 --model Qwen3.6-35B-A3B-GGUF
```

### 3. Evaluate

```bash
uv run python backend/judge_eval.py                # all papers
uv run python backend/judge_eval.py --paper-source dataset  # golden set only
```

Computes agreement rates, class-level accuracy, and per-model alignment stats against Echo-DSRN predictions.

---

## Architecture

| Component | Description |
|---|---|
| **Frontend** | Vue 3 / Vite dashboard — tabs: Model Card, Collab Hub, OpenAIRE Stream, Free Text, Saved History |
| **Backend** | FastAPI — inference, arXiv/OpenAIRE proxy, LLM judge orchestration, auth |
| **Classifier** | Echo-DSRN-98M (`EchoForSequenceClassification`) — sub-millisecond CPU inference |
| **Annotation DB** | SQLite (`data/collaborative.db`) — stores user classifications and LLM judge labels |
| **Worker** | Celery + Redis — async classification queue |

**Intent labels:**

| Label | Meaning |
|---|---|
| `Methodology` | Introduces a new method, model, or algorithm |
| `Dataset` | Introduces or documents a dataset or benchmark |
| `Review` | Surveys or synthesises existing work |
| `Applied` | Applies existing methods to a domain problem |
| `Theoretical` | Mathematical or formal analysis without empirical evaluation |

---

## Inference

> **Model:** [`ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF`](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) — gated repo, [request access](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) before use.

```bash
pip install git+https://github.com/ethicalabs-ai/Echo-DSRN.git
```

```python
from echo_dsrn import EchoForSequenceClassification
from transformers import AutoTokenizer

model     = EchoForSequenceClassification.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF", trust_remote_code=True
)

label, probs = model.classify(
    "Title: Attention Is All You Need\n"
    "Abstract: We propose a new simple network architecture, the Transformer...",
    tokenizer=tokenizer,
)
print(label, probs)
# → Methodology  tensor([0.87, 0.03, 0.02, 0.06, 0.02])
```

---

## Legal

© 2026 ethicalabs.ai — incubated by dorvan srl

Operational: Via Privata Farnese 1/3, 20146 Milano (MI)
Registered: Via Enrico Cernuschi 4, 20129 Milano (MI), Italy

This project uses the [OpenAIRE Graph API](https://graph.openaire.eu) — all materials created by OpenAIRE are licensed under CC BY 4.0.
