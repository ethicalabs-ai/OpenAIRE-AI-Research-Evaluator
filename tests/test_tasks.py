import os
import sys

import pytest

# Ensure backend directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from tasks import ping_task


def test_ping_task():
    """Verify that ping_task returns 'pong'."""
    assert ping_task() == "pong"


# ── classify_and_judge tests ──────────────────────────────────────────────────


def test_classify_and_judge_happy_path(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """A new paper should be classified, judged, and annotated successfully."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.1",
        title="Test Paper",
        abstract="We propose a novel method.",
        model="test-model",
    )

    assert result["status"] == "ok"
    assert result["proposed_label"] == "Methodology"
    assert result["label_valid"] is True
    assert result["is_flagged"] is False


def test_classify_and_judge_already_judged(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """Calling the task twice with the same DOI + model should skip the second."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result1 = classify_and_judge(
        doi="10.1234/test.2",
        title="Test Paper 2",
        abstract="We propose another method.",
        model="test-model",
    )
    assert result1["status"] == "ok"

    result2 = classify_and_judge(
        doi="10.1234/test.2",
        title="Test Paper 2",
        abstract="We propose another method.",
        model="test-model",
    )
    assert result2["status"] == "skipped"
    assert result2["reason"] == "already-judged"


def test_classify_and_judge_llm_error(
    temp_db, mock_classifier, monkeypatch
):
    """When the LLM judge raises, the task should return an error status."""
    def _raise_llm_error(*a, **kw):
        raise ValueError("LLM timeout")

    monkeypatch.setattr(
        "llm_judge.judge_paper",
        _raise_llm_error,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.3",
        title="Test Paper 3",
        abstract="An abstract that will cause an error.",
        model="test-model",
    )

    assert result["status"] == "error"
    assert "llm-error" in result["reason"]


def test_classify_and_judge_different_models_same_paper(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """Same paper with different models should create separate annotations."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result1 = classify_and_judge(
        doi="10.1234/test.4",
        title="Test Paper 4",
        abstract="Multi-model test.",
        model="model-a",
    )
    assert result1["status"] == "ok"

    result2 = classify_and_judge(
        doi="10.1234/test.4",
        title="Test Paper 4",
        abstract="Multi-model test.",
        model="model-b",
    )
    assert result2["status"] == "ok"


def test_classify_and_judge_with_initial_intent(
    temp_db, mock_judge_verdict, monkeypatch
):
    """When initial_intent is provided, the classifier should not be called."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.5",
        title="Test Paper 5",
        abstract="Pre-labeled paper.",
        model="test-model",
        initial_intent="Review",
        source="dataset",
    )

    assert result["status"] == "ok"
    # Verify the paper record was created with the given initial_intent
    from database import SessionLocal
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = db.query(PaperRecord).filter(PaperRecord.doi == "10.1234/test.5").first()
        assert p is not None
        assert p.initial_intent == "Review"
        assert p.source == "dataset"
    finally:
        db.close()


# ── classify_mcp tests ────────────────────────────────────────────────────────


class FakeMCPResult:
    label = "Methodology"
    probabilities = {"Methodology": 0.87, "Dataset": 0.03, "Review": 0.02, "Applied": 0.06, "Theoretical": 0.02}


def test_classify_mcp_returns_label_and_probabilities(monkeypatch):
    """MCP task should return label + all 5 probabilities without DB writes."""
    monkeypatch.setattr(
        "intent_classifier.classify_paper",
        lambda title, abstract: FakeMCPResult(),
    )

    from tasks import classify_mcp

    result = classify_mcp("Test Title", "Test Abstract")

    assert result["label"] == "Methodology"
    assert "probabilities" in result
    assert len(result["probabilities"]) == 5
    assert result["probabilities"]["Methodology"] == 0.87
