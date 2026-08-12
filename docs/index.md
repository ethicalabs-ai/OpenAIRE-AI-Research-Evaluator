# OpenAIRE-AI-Research-Evaluator

A multi-model LLM-as-Judge pipeline that builds annotation datasets to train [Echo-DSRN](https://github.com/ethicalabs-ai/Echo-DSRN) intent classifiers — entry for the [OpenAIRE AI Hackathon 2026](https://innovation.openaire.eu/component/content/article/openaire-ai-hackathon.html), co-organised by OpenAIRE and Alien Intelligence.

The app streams scientific paper metadata from the [OpenAIRE Graph API](https://graph.openaire.eu), classifies research intent with a 98M-parameter [Echo-DSRN](https://www.ethicalabs.ai/research/echo-dsrn/) model (fast CPU inference), and enlists multiple LLMs as judges to build a golden consensus dataset.

## Features

- **OpenAIRE Stream** — live publication metadata via the OpenAIRE Graph API
- **Collab Hub** — community annotations with LLM judge consensus
- **Free Text** — ad-hoc classification of custom titles and abstracts
- **Saved History** — local browser storage with optional server-side sync
- **Model Card** — architecture specs, parameter census, and consumption code

## Hackathon Deliverable

This project is the artifact submission for the OpenAIRE AI Hackathon 2026.

The accompanying [story](./story.md) explains the question, journey, insight, and what others can reuse.

[Source](https://github.com/ethicalabs-ai/OpenAIRE-AI-Research-Evaluator) · [OpenAIRE Graph](https://graph.openaire.eu) · [Echo-DSRN Model](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF)
