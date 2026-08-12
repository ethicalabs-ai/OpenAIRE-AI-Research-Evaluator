---
library_name: transformers
tags:
- text-classification
- research-intent
- echo-dsrn
- openaire-2026-hackathon
- vllm
base_model:
- ethicalabs/Echo-DSRN-114M-v0.1.2
license: apache-2.0
language:
- en
pipeline_tag: text-classification
---

# Model Card for ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF

[![GitHub](https://img.shields.io/badge/GitHub-ethicalabs.ai-black.svg)](https://github.com/ethicalabs-ai/Echo-DSRN/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-openaire--2026.ethicalabs.ai-brightgreen.svg)](https://openaire-2026.ethicalabs.ai/)
[![OpenAIRE Hackathon](https://img.shields.io/badge/OpenAIRE-AI_Hackathon_2026-orange.svg)](https://innovation.openaire.eu/component/content/article/openaire-ai-hackathon.html)

> [!WARNING]
> This repository contains experimental models designed strictly for academic evaluation and research purposes.
>
> Critical Constraints:
> * **No Production Deployment:** Experimental models must not be deployed in commercial, enterprise, or mission-critical environments under any circumstances.
> * **No Liability:** Experimental models are provided "as-is" without warranties of any kind. The developers assume zero liability for downstream consequences, system integration failures, or regulatory non-compliance resulting from unauthorized deployment.

This is a 98 million parameters sequence classification model based on the **Echo-DSRN** architecture, trained on a single [AMD](https://huggingface.co/amd) GPU using ROCm 7.2.

It is based on [`ethicalabs/Echo-DSRN-114M-v0.1.2`](https://huggingface.co/ethicalabs/Echo-DSRN-114M-v0.1.2).

---

## OpenAIRE AI Hackathon 2026

This model is part of **ethicalabs.ai**'s entry in the [OpenAIRE AI Hackathon 2026](https://innovation.openaire.eu/component/content/article/openaire-ai-hackathon.html), co-organised by OpenAIRE and Alien Intelligence — a 12-week open science build challenge.

**[🔴 Live Demo → openaire-2026.ethicalabs.ai](https://openaire-2026.ethicalabs.ai/)**

The live application classifies research papers in real time using this model, with publication metadata streamed from the [OpenAIRE Graph API](https://graph.openaire.eu) under **CC BY 4.0**.

## What this model does

Echo-DSRN-v0.1.3-Research-Intent-CLF is a **research paper intent classifier** — it reads a paper's title and abstract and predicts one of five categories:

| Label | Meaning |
|---|---|
| **Methodology** | Introduces a new method, model, or algorithm |
| **Dataset** | Introduces or documents a dataset or benchmark |
| **Review** | Surveys or synthesises existing work |
| **Applied** | Applies existing methods to a domain problem |
| **Theoretical** | Mathematical or formal analysis without empirical evaluation |

Inference runs in **under 1 millisecond on CPU**, making it suitable for streaming applications where a GPU is unavailable.

## How to use

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

The model is gated — [request access](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) before downloading.

## Model specs

| Property | Value |
|---|---|
| Architecture | Echo-DSRN (Recurrent Neural Network) |
| Parameters | 98,266,629 (~98M) |
| Layers | 8 DSRN blocks |
| Hidden dim | 512 |
| Attention heads | 4 |
| Vocab size | 32,017 tokens |
| Precision | bfloat16 |
| Base model | Echo-DSRN-114M-v0.1.2 |
| GPU | AMD Radeon AI Pro R9700 (ROCm 7.2) |

## Building a better dataset — we need you

This model is actively powering the live classifier at [openaire-2026.ethicalabs.ai](https://openaire-2026.ethicalabs.ai/).

Every paper classified on the platform becomes part of a growing annotation dataset stored in the system.

**We need human contributors**, not just LLM judges. If you're a researcher, librarian, or domain expert, you can:

- Visit the [Collab Hub](https://openaire-2026.ethicalabs.ai/collab) and vote on papers
- Flag misclassified papers
- Save papers to your private history
- Log in with your HuggingFace account

The multi-model LLM-as-Judge pipeline (20 LLMs across Qwen, Gemma, Nemotron, GPT-OSS, LFM2, Phi, SmolLM, Granite, Bonsai, GLM and Gemini families) validates Echo-DSRN predictions.

But LLM consensus is no substitute for human expertise — your domain knowledge helps us catch edge cases the models miss.

## After the hackathon

The current model will be **fine-tuned on the collected annotation dataset** and re-released as `v0.1.4` after the hackathon concludes (submission deadline: 20 August 2026).

The fine-tuned model will benefit from:

- Thousands of LLM + Human annotations from the OpenAIRE Graph stream.
- Multi-model consensus labels from LLM judges.
- Community feedback from the Collab Hub.

## Data provenance

Training data for the current version includes curated records from PubMed, Semantic Scholar, Papers With Code, and arXiv.

The live application streams additional publication metadata from the [OpenAIRE Graph API](https://graph.openaire.eu), licensed under **CC BY 4.0**.

## License

Apache 2.0 — see [LICENSE](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF/blob/main/LICENSE).

OpenAIRE Graph API data: CC BY 4.0.

## Citation

If you use this model or the annotation dataset, please cite:

> _OpenAIRE-AI-Research-Evaluator — Multi-model LLM-as-Judge pipeline for research intent classification._
> ethicalabs.ai, 2026. Apache-2.0 / CC-BY 4.0.
> [github.com/ethicalabs-ai/OpenAIRE-AI-Research-Evaluator](https://github.com/ethicalabs-ai/OpenAIRE-AI-Research-Evaluator)

