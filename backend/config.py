import os as _os

# =============================================================================
# LLM-as-Judge configuration
# =============================================================================

LLM_BASE_URL: str = _os.getenv("LLM_BASE_URL", "http://host.docker.internal:13305/v1")
LLM_API_KEY: str = _os.getenv("LLM_API_KEY", "not-needed")

# Default model used when --model is not specified.
# Any model ID supported by the LLM server is accepted.
LLM_JUDGE_DEFAULT_MODEL: str = _os.getenv(
    "LLM_JUDGE_DEFAULT_MODEL", "Gemma-4-E4B-it-GGUF"
)

# =============================================================================
# Export — gold-standard dataset
# =============================================================================
# Set to "true" / "1" / "yes" to enable the download endpoint and UI button.
# Leave blank (default) to disable until enough human annotations accumulate.
EXPORT_ENABLED: bool = _os.getenv("EXPORT_ENABLED", "").lower() in ("1", "true", "yes")
