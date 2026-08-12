---
title: Echo-DSRN OpenAIRE CLF Docs
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: cc-by-4.0
short_description: Echo-DSRN multi-model LLM-as-Judge docs
---

# Echo-DSRN OpenAIRE CLF Docs

Rendered documentation for the **Echo-DSRN research-intent evaluator** —
a multi-model LLM-as-Judge pipeline built for the OpenAIRE AI Hackathon 2026.

This Space is a static build of the project's MkDocs site. Source lives in the
[OpenAIRE-AI-Research-Evaluator](https://github.com/ethicalabs-ai/OpenAIRE-AI-Research-Evaluator)
repository under `docs/`.

## Contents

- **Quick Start** — run the classifier and the collaborative hub
- **Architecture** — Echo-DSRN intent classifier + multi-model LLM-as-Judge pipeline
- **LLM-as-Judge** — the judge protocol, verdict schema, and dataset export
- **MCP** — agent-facing classification via Server-Sent Events
- **OAuth** — HuggingFace login for the collaborative hub
- **Inference** — local and remote inference
- **Hackathon Story** — question, journey, insight, reusability
- **Deployment** — Docker Compose, Helm/k3s, PostgreSQL

## Datasets & models

- [Research-Intent-Judge](https://huggingface.co/datasets/ethicalabs/Research-Intent-Judge) — per-model LLM-as-Judge exports
- [Research-Intent-Collab](https://huggingface.co/datasets/ethicalabs/Research-Intent-Collab) — consensus golden dataset
- [Echo-DSRN-v0.1.3-Research-Intent-CLF](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) — the classifier

## License

CC-BY 4.0.
