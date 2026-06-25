# Architecture

| Component | Description |
|---|---|
| **Frontend** | Vue 3 / Vite dashboard — tabs: Model Card, Collab Hub, OpenAIRE Stream, Free Text, Saved History |
| **Backend** | FastAPI — inference, arXiv/OpenAIRE proxy, LLM judge orchestration, OAuth |
| **Classifier** | Echo-DSRN-98M (`EchoForSequenceClassification`) — sub-millisecond CPU inference |
| **Annotation DB** | SQLite (`data/collaborative.db`) — user classifications and LLM judge labels |
| **Worker** | Celery + Redis — async classification and LLM-as-Judge queue |

## Intent Labels

| Label | Meaning |
|---|---|
| `Methodology` | Introduces a new method, model, or algorithm |
| `Dataset` | Introduces or documents a dataset or benchmark |
| `Review` | Surveys or synthesises existing work |
| `Applied` | Applies existing methods to a domain problem |
| `Theoretical` | Mathematical or formal analysis without empirical evaluation |

## Data Flow

1. **OpenAIRE Graph API** streams paper metadata (title, abstract, DOI, language)
2. **Echo-DSRN-98M** classifies research intent in &lt;1ms on CPU
3. Results displayed in the Vue 3 dashboard with probability bars
4. Optional: **LLM-as-Judge** enlists multiple LLMs to validate Echo's predictions
5. Annotations stored in SQLite, forming a golden consensus dataset
