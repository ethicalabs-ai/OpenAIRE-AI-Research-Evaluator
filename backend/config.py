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
# Export — gold-standard dataset
# =============================================================================
# Set to "true" / "1" / "yes" to enable the download endpoint and UI button.
# Leave blank (default) to disable until enough human annotations accumulate.
EXPORT_ENABLED: bool = os.getenv("EXPORT_ENABLED", "").lower() in ("1", "true", "yes")

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
