# Architecture

| Component | Description |
|---|---|
| **Frontend** | Vue 3 / Vite dashboard — tabs: Model Card, Collab Hub, OpenAIRE Stream, Free Text, Saved History |
| **Backend** | FastAPI — inference, arXiv/OpenAIRE proxy, LLM judge orchestration, OAuth |
| **Classifier** | Echo-DSRN-98M (`EchoForSequenceClassification`) — fast CPU inference |
| **Annotation DB** | PostgreSQL (`echo_dsrn`) — user classifications and LLM judge labels |
| **Worker** | Celery + Redis — async classification and LLM-as-Judge queue |

## Intent Labels

| Label | Meaning |
|---|---|
| `Methodology` | Introduces a new method, model, or algorithm |
| `Dataset` | Introduces or documents a dataset or benchmark |
| `Review` | Surveys or synthesises existing work |
| `Applied` | Applies existing methods to a domain problem |
| `Theoretical` | Mathematical or formal analysis without empirical evaluation |
| `Unclassifiable` | Garbled / non-English / non-academic content, flagged by multiple LLM judges |

## Versioned Annotation Rounds

The collab hub operates in versioned rounds. Every paper import and annotation
is stamped with `model_version` — the model version being evaluated at the
time (the active round comes from the `MODEL_VERSION` setting). Hub queries
(papers, details, stats) filter to the active round by default.

- **Active round:** annotated against the current model (v0.1.4, 6-class) —
  new votes train the next model (v0.1.5).
- **Archives:** completed rounds stay browsable read-only at
  `/collab/archive/<version>/`, e.g. [`/collab/archive/v0.1.3/`](https://openaire-2026.ethicalabs.ai/collab/archive/v0.1.3/).

## Data Flow

1. **OpenAIRE Graph API** streams paper metadata (title, abstract, DOI, language)
2. **Echo-DSRN-98M** classifies research intent in &lt;1ms on CPU
3. Results displayed in the Vue 3 dashboard with probability bars
4. Optional: **LLM-as-Judge** enlists multiple LLMs to validate Echo's predictions
5. Annotations stored in PostgreSQL, forming a golden consensus dataset
