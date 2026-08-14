import os

# =============================================================================
# Collaborative hub — active annotation round
# =============================================================================
# The version tag stamped on every new paper import / annotation during the
# current data-collection round. This is the version of the MODEL being
# evaluated and annotated right now (v0.1.4), NOT the model that will later be
# trained from the collected data (v0.1.5). The hub lists, aggregates, and
# stats are filtered to this version by default. Previous rounds stay
# queryable through the archive (e.g. /collab/archive/v0.1.3/).
# Override via MODEL_VERSION.
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v0.1.4")

# Research-intent classes exposed to the hub UI (human voting + filters).
# Order matters — it drives the label-selector row. The classifier's own
# id2label stays authoritative for model predictions.
INTENT_CLASS_LABELS: list[str] = [
    "Methodology",
    "Dataset",
    "Review",
    "Applied",
    "Theoretical",
    "Unclassifiable",
]

# =============================================================================
# LLM-as-Judge configuration
# =============================================================================

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://host.docker.internal:13305/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")

# Default model used when --model is not specified.
# Any model ID supported by the LLM server is accepted.
LLM_JUDGE_DEFAULT_MODEL: str = os.getenv(
    "LLM_JUDGE_DEFAULT_MODEL", "Gemma-4-E4B-it-GGUF"
)

# =============================================================================
# Export — dataset export filters
# =============================================================================

# LLM models to exclude from dataset exports (underperforming or deprecated).
# Model names must match the `llm_model` column in the annotations table.
EXCLUDED_JUDGE_MODELS: list[str] = [
    "user.Kurtis-E1.1-Qwen3-4B-GGUF-IQ4_XS",
    "Ministral-3-3B-Instruct-2512-GGUF",
    "Jan-v1-4B-GGUF",
    "DeepSeek-Qwen3-8B-GGUF",
    # Test artifacts that leaked into the production DB
    "test-model",
    "model-a",
    "model-b",
]

# =============================================================================
# Export — gold-standard dataset
# =============================================================================
# Public URL for the curated datasets on HuggingFace.
# Override via HF_DATASETS_URL env var.
HF_DATASETS_URL: str = os.getenv(
    "HF_DATASETS_URL",
    "https://huggingface.co/collections/ethicalabs/openaire-ai-hackathon-2026",
)

# =============================================================================
# Collaborative hub — read-only kill switch
# =============================================================================
# Set to "true" / "1" / "yes" to disable all mutation endpoints:
# login, voting, paper import, saved papers, and remote LLM judge submissions.
# The web interface becomes browse-only. Default: writable (empty/unset).
HUB_READ_ONLY: bool = os.getenv("HUB_READ_ONLY", "").lower() in ("1", "true", "yes")

# =============================================================================
# API Key — shared secret for agent-facing endpoints (MCP SSE + judge annotations)
# =============================================================================
# Single API key for machine-to-machine endpoints.
# Set to a strong random value in production.
API_KEY: str = os.getenv("API_KEY", "echo-dsrn-mcp-change-me-in-production")

# =============================================================================
# Remote Judge API — URL of the deployed server for remote annotation persistence
# =============================================================================
# Used by judge_cli.py --remote mode to POST annotations to the deployed server.
JUDGE_API_URL: str = os.getenv("JUDGE_API_URL", "")
