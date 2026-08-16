"""
llm_judge.py
────────────
LLM-as-Judge module for the Echo-DSRN Collaborative Dataset Platform.

The LLM receives the paper title, abstract, and the Echo model's classification
prediction, then returns a structured JSON verdict via OpenAI-compatible API
(JSON schema enforced via response_format + Pydantic).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from config import LLM_API_KEY, LLM_BASE_URL, LLM_JUDGE_DEFAULT_MODEL
from openai import OpenAI
from pydantic import BaseModel, Field


def _extract_json(text: str) -> str:
    """Strip markdown code fences and extract the JSON payload.

    Some models (LFM2, etc.) wrap their response in ```json ... ``` fences.
    """
    text = text.strip()
    # Remove ```json or ``` prefix and trailing ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively make a JSON schema compliant with OpenAI strict mode.

    OpenAI strict mode requires:
      - additionalProperties: false on every object
      - every key in properties must appear in required (optional fields
        use anyOf: [{type: X}, {type: null}] but must still be required)

    Pydantic's model_json_schema() emits neither — this helper adds both.
    """
    schema = dict(schema)
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["properties"] = {
                k: _make_strict_schema(v) for k, v in schema["properties"].items()
            }
            # All properties must be listed in required
            schema["required"] = list(schema["properties"].keys())
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_make_strict_schema(s) for s in schema[key]]
    if "$defs" in schema:
        schema["$defs"] = {
            k: _make_strict_schema(v) for k, v in schema["$defs"].items()
        }
    return schema


def _is_local_server(url: str) -> bool:
    """Return True if the URL points to a local llama.cpp-compatible server.

    Only local servers accept llama.cpp-specific extras (chat_template_kwargs,
    max_tokens, custom temperature).  Any cloud API (OpenAI, Gemini, Anthropic,
    Mistral …) with a real domain name should return False.
    """
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return (
        host in ("localhost", "")
        or host.startswith("127.")
        or host.startswith("192.168.")
        or host.startswith("10.")
        or host.startswith("172.")  # covers 172.16–172.31 private range
    )


# ---------------------------------------------------------------------------
# Pydantic schema — the LLM is forced to output exactly this structure
# ---------------------------------------------------------------------------

VALID_LABELS = Literal["Methodology", "Dataset", "Review", "Applied", "Theoretical"]


class JudgeVerdict(BaseModel):
    """Structured verdict emitted by the LLM judge."""

    label_valid: bool = Field(
        default=False,
        description=(
            "True if the Echo model's predicted label is correct for this paper. "
            "False if the label should be changed."
        ),
    )
    proposed_label: VALID_LABELS = Field(
        default="Methodology",
        description=(
            "The label that best describes the paper's research intent. "
            "Must be one of: Methodology, Dataset, Review, Applied, Theoretical. "
            "If label_valid is True this should match the model prediction."
        ),
    )
    is_flagged: bool = Field(
        default=False,
        description=(
            "True if the paper should be flagged due to quality issues: "
            "garbled text, empty abstract, math-only content, non-English without translation, "
            "or extreme domain mismatch."
        ),
    )
    flag_reason: Optional[str] = Field(
        default=None,
        description="Short explanation of why the paper is flagged. Null if is_flagged is False.",
    )
    rationale: str = Field(
        default="",
        description=(
            "1-3 sentence explanation of why the label is valid/invalid and "
            "any notable observations about the paper content."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Judge's confidence in this verdict.",
    )


# ── field-name aliases for models that use non-standard keys ───────────────
_JUDGE_FIELD_ALIASES: dict[str, str] = {
    "verdict": "proposed_label",
    "label": "proposed_label",
    "intent": "proposed_label",
    "prediction": "proposed_label",
    "valid": "label_valid",
    "correct": "label_valid",
    "flagged": "is_flagged",
    "flag": "is_flagged",
    "issue": "is_flagged",
    "reason": "flag_reason",
    "explanation": "rationale",
    "summary": "rationale",
    "certainty": "confidence",
}


def _normalize_judge_json(
    data: dict[str, Any], model_prediction: str = ""
) -> dict[str, Any]:
    """Normalize field names from loose model outputs to the Pydantic schema.

    Lower-priority alias mappings never overwrite an already-set field.
    """
    normalized: dict[str, Any] = {}
    for key, val in data.items():
        mapped = _JUDGE_FIELD_ALIASES.get(key.lower().strip(), key.lower().strip())
        # Never overwrite a value already set by a higher-priority key
        if mapped not in normalized:
            normalized[mapped] = val

    # Derive label_valid from proposed_label if not explicitly set
    if "proposed_label" in normalized and "label_valid" not in normalized:
        normalized["label_valid"] = (
            normalized["proposed_label"] == model_prediction
        )

    # Derive is_flagged from flag_reason
    if (
        "flag_reason" in normalized
        and normalized.get("flag_reason")
        and "is_flagged" not in normalized
    ):
        normalized["is_flagged"] = True

    return normalized


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert research classification judge for the Echo-DSRN Collaborative Dataset.

Your task is to evaluate whether an automated model's classification of a research paper
is correct, and provide a structured verdict.

## Classification labels
- **Methodology**: Introduces a new algorithm, model architecture, training technique, or mathematical method.
- **Dataset**: Primarily contributes a new dataset, benchmark, annotation scheme, or data collection process.
- **Review**: Survey, systematic review, meta-analysis, or position/opinion paper with no new empirical contribution.
- **Applied**: Applies existing methods to a specific domain or real-world problem (medicine, climate, NLP tasks, etc.).
- **Theoretical**: Mathematical or formal analysis, proofs, convergence guarantees, complexity bounds, no empirical evaluation.

## Instructions
1. Read the title and abstract carefully.
2. Assess the model's predicted label.
3. Return a JSON verdict strictly following the provided schema.
4. Be concise in the rationale (max 3 sentences).
5. Flag a paper if the abstract is empty, garbled, non-scientific, or math-symbol-only.
6. Do NOT make up information not present in the abstract.
"""


# ---------------------------------------------------------------------------
# Judge function
# ---------------------------------------------------------------------------


def judge_paper(
    title: str,
    abstract: str,
    model_prediction: str,
    *,
    model: str = LLM_JUDGE_DEFAULT_MODEL,
    timeout: float = 300.0,
) -> JudgeVerdict:
    """
    Call the LLM judge and return a structured JudgeVerdict.

    Uses standard chat.completions.create with response_format json_schema —
    supported by vLLM, llama.cpp, LM Studio, and any OpenAI-compatible server.
    The response is parsed with Pydantic's model_validate_json().

    Raises:
        openai.APIError: on network / server errors
        pydantic.ValidationError: if the response cannot be parsed
    """
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=timeout,
        max_retries=1,
    )

    user_message = (
        f"**Paper title:** {title}\n\n"
        f"**Abstract:**\n{abstract or '(no abstract provided)'}\n\n"
        f"**Echo model prediction:** {model_prediction}\n\n"
        "Evaluate whether the prediction is correct. Set `proposed_label` to YOUR "
        "own classification (what YOU think the label should be), not to the Echo prediction. "
        "Return your verdict as JSON."
    )

    # chat_template_kwargs / repeat_penalty are llama.cpp server extensions —
    # the OpenAI client only forwards them via extra_body. Cloud APIs
    # (OpenAI, Gemini, Anthropic, …) reject them with 400 errors.
    is_local = _is_local_server(LLM_BASE_URL or "")
    extra: dict = (
        {
            "chat_template_kwargs": {"enable_thinking": False},
            # Some judges (e.g. Nemotron-3.5-Lightning) degenerate into
            # repetitive rationales that blow the token budget mid-JSON.
            "repeat_penalty": 1.15,
        }
        if is_local
        else {}
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "JudgeVerdict",
                "strict": True,
                "schema": _make_strict_schema(JudgeVerdict.model_json_schema()),
            },
        },
        # Reasoning models (o1, o3, gpt-5.5+) only accept temperature=1 (default) — omit it.
        # Local llama.cpp servers accept any value.
        **{"temperature": 0.1} if is_local else {},
        # max_tokens vs max_completion_tokens: OpenAI reasoning models use the latter.
        # Some newer judges emit long rationales and truncate before closing the
        # JSON — 2048 tokens caused "Unterminated string" parse failures.
        **{"max_tokens": 4096} if is_local else {"max_completion_tokens": 4096},
        extra_body=extra or None,
    )

    raw_json = response.choices[0].message.content or ""
    data = json.loads(_extract_json(raw_json))
    data = _normalize_judge_json(data, model_prediction)
    verdict = JudgeVerdict.model_validate(data)

    # Correct LLM internal inconsistency: if the judge proposes the same label
    # as Echo but still says valid=False, override to True.
    if (
        not verdict.label_valid
        and verdict.proposed_label == model_prediction
    ):
        verdict.label_valid = True

    return verdict
