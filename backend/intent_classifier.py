"""
backend/intent_classifier.py
────────────────────────────
Lazy-loaded singleton wrapper around EchoForSequenceClassification for the
research-intent classifier.

The model is loaded once on first call to ``get_classifier()`` and reused for
every subsequent request.  Loading is thread-safe via a module-level lock so
concurrent startup requests do not race.

Environment variables
─────────────────────
    INTENT_CLF_PATH   Path to the EchoForSequenceClassification checkpoint
                      directory.  Defaults to
                      ``outputs/echo-research-intent-clf`` relative to the
                      repository root (two directories above this file).

    INTENT_CLF_DEVICE ``cpu`` | ``cuda`` | ``auto``  (default: ``auto``)
                      ``auto`` selects CUDA if available, otherwise CPU.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

# ── Default model path: Hugging Face repo ID or local checkpoint ──────────────
_DEFAULT_CLF_PATH = "ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF"


@dataclass
class ClassifyResult:
    """Return type of :func:`classify_paper`."""

    label: str
    probabilities: dict[str, float]  # label → probability (all 5 classes)
    model_path: str


# ── Singleton state ───────────────────────────────────────────────────────────
_lock = threading.Lock()
_model = None
_tokenizer = None
_model_path: Optional[str] = None


def get_classifier():
    """Return (model, tokenizer), loading from disk on first call."""
    global _model, _tokenizer, _model_path

    if _model is not None:
        return _model, _tokenizer

    with _lock:
        # Double-checked locking
        if _model is not None:
            return _model, _tokenizer

        path = os.environ.get("INTENT_CLF_PATH", _DEFAULT_CLF_PATH)
        device_env = os.environ.get("INTENT_CLF_DEVICE", "auto")

        is_local = Path(path).exists()
        if not is_local and "/" not in path:
            raise FileNotFoundError(
                f"Intent classifier not found at {path!r}. "
                "Set INTENT_CLF_PATH or train the model first:\n"
                "  uv run python training/train_clf.py ..."
            )

        log.info("Loading intent classifier from %s …", path)

        from transformers import AutoTokenizer

        from echo_dsrn import EchoForSequenceClassification

        if device_env == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_env

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = EchoForSequenceClassification.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.eval()

        _model = model
        _tokenizer = tokenizer
        _model_path = path

        labels = list(model.config.id2label.values())
        log.info("Intent classifier ready — labels: %s  device: %s", labels, device)

    return _model, _tokenizer


def classify_paper(title: str, abstract: str) -> ClassifyResult:
    """Classify a research paper by title + abstract.

    Parameters
    ----------
    title:
        Paper title (plain text, no markup).
    abstract:
        Paper abstract (plain text, no markup).  Should be ≥ 120 chars for
        reliable predictions.

    Returns
    -------
    ClassifyResult
        Predicted label and per-class softmax probabilities.
    """
    model, tokenizer = get_classifier()
    text = f"Title: {title}\nAbstract: {abstract}"

    with torch.no_grad():
        pred_label, probs = model.classify(text, tokenizer=tokenizer)

    id2label: dict[int, str] = model.config.id2label
    prob_dict = {id2label[i]: round(float(p), 4) for i, p in enumerate(probs.tolist())}

    return ClassifyResult(
        label=pred_label,
        probabilities=prob_dict,
        model_path=_model_path or "",
    )


def is_loaded() -> bool:
    """Return True if the model singleton has been initialised."""
    return _model is not None
