# Hackathon Story

## The Question

> How can we turn the OpenAIRE Graph — a living index of millions of research outputs — into validated, high-quality training data for AI classifiers, at scale and with minimal human effort?

The OpenAIRE Graph aggregates publication metadata across disciplines, languages, and repositories.

It is an extraordinary resource for building AI tools that understand research, But raw metadata is noisy: titles are truncated, abstracts vary in quality, and classification labels are absent.

To train a classifier that can reliably distinguish a methodology paper from a dataset paper, a review from an applied case study, you need **labeled data** — and labeling thousands of papers by hand is slow, expensive, and fragile.

Our question was: can we automate this with a feedback loop that combines a fast, lightweight classifier with a panel of LLM judges, using the OpenAIRE Graph as both the data source and the proving ground?

---

## The Journey

### Phase 1 — The Classifier

We started with [**Echo-DSRN**](https://www.ethicalabs.ai/research/echo-dsrn/), a recurrent neural network developed built from scratch by us.

Echo-DSRN is small (98M-114M parameters) and runs efficiently on CPU, making it suitable for real-time streaming applications where a GPU is unavailable or impractical.

We fine-tuned it on a 5-class research intent taxonomy:

- **Methodology** — introduces a new method, model, or algorithm
- **Dataset** — introduces or documents a dataset or benchmark
- **Review** — surveys or synthesises existing work
- **Applied** — applies existing methods to a domain problem
- **Theoretical** — mathematical or formal analysis without empirical evaluation

The initial training set was ~6,000 records curated from PubMed, Semantic Scholar, Papers With Code, arXiv, and the OpenAIRE Graph itself — silver-labeled with keyword heuristics and arXiv category priors.

The classifier worked well, but we had no systematic way to measure how well, or to improve it.

### Phase 2 — The App

We built a Vue 3 + FastAPI web application that streams paper metadata from the OpenAIRE Graph API in real time.

Users see papers appear with Echo-DSRN's predicted intent and a probability distribution across all five classes.

They can classify custom text, save results to local history, and browse a Model Card showing the exact architecture, parameter census, and consumption code.

But passive classification is only half the story.

To improve the model, we needed ground truth — and that meant building a pipeline for annotation at scale.

### Phase 3 — LLM-as-Judge

We designed an LLM-as-Judge pipeline where multiple high-capability language models evaluate Echo-DSRN's predictions and produce structured verdicts:

- Is the predicted label correct?
- If not, what should it be?
- Should the paper be flagged (garbled text, non-English, math-only)?
- What is the judge's confidence?

Each judge receives the paper title, abstract, and Echo's prediction, then returns a JSON verdict following a strict Pydantic schema enforced via OpenAI's `response_format`.

We support any OpenAI-compatible inference server: llama.cpp, vLLM, LM Studio, Google Gemini, and lemonade for local GGUF model orchestration.

We assembled a panel of 11 models spanning architectures and parameter counts: `Qwen3.6-35B`, `Qwen3.6-27B`, `Qwen3.5-35B`, `Nemotron-3-Nano-30B`, `Gemma-4-26B`, `GPT-OSS-20B`, `GPT-OSS-120B`, `DeepSeek-Qwen3-8B`, `GLM-4.7-Flash`, `Qwen3.6-35B-A3B`, and `Qwen3.5-4B`.

### Phase 4 — Scale with Celery

Running 11 models across thousands of papers synchronously is slow — each paper needs an LLM API call with its own latency.

We added a Celery + Redis worker queue: the CLI dispatches fine-grained per-paper tasks, and workers pull from the queue and process them asynchronously. The same pipeline can run with a single command:

```bash
./scripts/run_judges.sh 100 --async
```

This dispatches 100 papers per model (1,100 total tasks) to the worker pool. The worker handles classification, LLM judging, and database persistence in one atomic unit, with idempotency guarantees — no paper is judged twice by the same model.

### Phase 5 — The Golden Dataset

All annotations are stored in a SQLite database (`data/collaborative.db`). The `judge_eval.py` script computes:

- Agreement rates between judges
- Per-class accuracy against ground truth (when available)
- Per-model alignment stats against Echo-DSRN

The result is a **golden consensus dataset** where each paper has labels from multiple LLMs alongside Echo-DSRN's original prediction. This dataset becomes the training data for the next iteration of the classifier — closing the feedback loop.

---

## The Insight

**A multi-model LLM-as-Judge pipeline, fed by the OpenAIRE Graph, produces higher-quality training data than any single annotator — human or machine — at a fraction of the cost.**

Three findings stand out:

1. **Diversity matters.** Different models disagree on edge cases (applied vs. methodology, dataset vs. methodology), and consensus across architectures produces more robust labels than the strongest single model alone.

2. **The OpenAIRE Graph is a live data flywheel.** New papers appear daily. A streaming classifier backed by LLM judges can continuously expand the training set without manual curation.

3. **CPU-first is viable.** Echo-DSRN-98M runs fast enough on CPU to be deployable anywhere — from a cloud VM to a Raspberry Pi. The heavy lifting (LLM judging) is offloaded to a remote GPU server, but the classification itself is free.

---

## What Others Can Reuse

This project is designed to be remixed. Here is what you can take and adapt:

| Component | Reuse scenario |
|---|---|
| **LLM-as-Judge pipeline** | Swap the 6-class taxonomy for your own labels. Point at any OpenAI-compatible server. The judge schema, prompt, and evaluation scripts are label-agnostic. |
| **OpenAIRE Graph integration** | The keyword streaming + multilingual query approach works for any research domain. The fetcher handles both arXiv and OpenAIRE APIs. |
| **Async Celery worker pattern** | Fine-grained per-paper tasks with lazy database imports and idempotency — copy the pattern for any batch annotation workflow. |
| **Dataset preparation & export scripts** | `prepare_research_intent.py` / `expand_multilingual_openaire.py` build balanced, multilingual training sets from public APIs; `export_llm_judge_dataset.py` and `export_golden_dataset.py` produce the published LLM-as-Judge and golden CLF datasets (per-model ChatML exports, consensus labelling with 6 classes incl. Unclassifiable, stratified splits). Adapt the keyword banks, label taxonomy, and consensus rules. |
| **Echo-DSRN classifier** | The model is on Hugging Face (gated — request access). `model.classify(title + abstract, tokenizer)` is the entire API. Swap the classification head for your own labels. |

## License

Code: Apache 2.0. OpenAIRE Graph: CC BY 4.0. All hackathon materials: CC-BY.
