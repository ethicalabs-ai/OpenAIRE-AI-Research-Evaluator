import os

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
    # "DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
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
API_KEY: str = os.getenv(
    "API_KEY", "echo-dsrn-mcp-change-me-in-production"
)

# =============================================================================
# Remote Judge API — URL of the deployed server for remote annotation persistence
# =============================================================================
# Used by judge_cli.py --remote mode to POST annotations to the deployed server.
JUDGE_API_URL: str = os.getenv("JUDGE_API_URL", "")
