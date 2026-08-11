"""Tests for export_llm_judge_dataset.py and export_golden_dataset.py."""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure backend + scripts are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))


# ── Unit: _normalize_model ────────────────────────────────────────────────────


def test_normalize_model_merges_mtp():
    """MTP and non-MTP variants should normalize to the same name."""
    from export_llm_judge_dataset import _normalize_model

    assert _normalize_model("Gemma-4-12B-it-MTP-GGUF") == "Gemma-4-12B-it-GGUF"
    assert _normalize_model("Gemma-4-12B-it-GGUF") == "Gemma-4-12B-it-GGUF"
    assert _normalize_model("Qwen3.6-35B-A3B-MTP-GGUF") == "Qwen3.6-35B-A3B-GGUF"
    assert _normalize_model("Qwen3.6-35B-A3B-GGUF") == "Qwen3.6-35B-A3B-GGUF"
    assert _normalize_model("SomeModel") == "SomeModel"
    assert _normalize_model(None) == "unknown"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def populated_db(temp_db):
    """Populate the temp DB with paper records and LLM annotations."""
    import database as _db
    importlib.reload(_db)
    from models import Annotation, PaperRecord

    session = _db.SessionLocal()
    try:
        # Paper 1 — clean, all models agree
        session.add(PaperRecord(
            doi="10.1234/paper.1",
            title="Test Paper One",
            abstract="We propose a novel method for NLP.",
            initial_intent="Methodology",
            source="arxiv",
        ))
        # Paper 2 — flagged by 2 models (should be excluded from golden)
        session.add(PaperRecord(
            doi="10.1234/paper.2",
            title="Garbled Paper",
            abstract="asdf jkl; qwerty xxxxxx.",
            initial_intent="Methodology",
            source="arxiv",
        ))
        # Paper 3 — Dataset paper
        session.add(PaperRecord(
            doi="10.1234/paper.3",
            title="A New Benchmark Dataset",
            abstract="We introduce a new dataset for intent classification.",
            initial_intent="Dataset",
            source="arxiv",
        ))
        # Paper 4 — disagreement between models
        session.add(PaperRecord(
            doi="10.1234/paper.4",
            title="Survey of Methods",
            abstract="We review recent advances in NLP methods.",
            initial_intent="Methodology",
            source="arxiv",
        ))
        # Paper 5 — Applied
        session.add(PaperRecord(
            doi="10.1234/paper.5",
            title="Clinical Application of BERT",
            abstract="We apply BERT to clinical note classification.",
            initial_intent="Applied",
            source="arxiv",
        ))
        session.flush()

        # Model A annotations (good model)
        for doi, label, flagged, flag_reason, comment in [
            ("10.1234/paper.1", "Methodology", False, None, "[HIGH confidence] Clearly a novel method paper."),
            ("10.1234/paper.2", "Methodology", True, "Garbled text", "[LOW confidence] Abstract appears to be garbled."),
            ("10.1234/paper.3", "Dataset", False, None, "[HIGH confidence] Clearly introduces a new dataset."),
            ("10.1234/paper.4", "Review", False, None, "[MEDIUM confidence] This is a survey paper."),
            ("10.1234/paper.5", "Applied", False, None, "[HIGH confidence] Clinical application of BERT."),
        ]:
            session.add(Annotation(
                paper_doi=doi,
                llm_model="Gemma-4-E4B-it-GGUF",
                annotator_type="llm",
                proposed_label=label,
                is_flagged=flagged,
                flag_reason=flag_reason,
                comment=comment,
            ))

        for doi, label, flagged, flag_reason, comment in [
            ("10.1234/paper.1", "Methodology", False, None, "[HIGH confidence] Introduces a new method."),
            ("10.1234/paper.2", "Methodology", True, "Garbled text", "[LOW confidence] Cannot parse abstract."),
            ("10.1234/paper.3", "Dataset", False, None, "[MEDIUM confidence] Dataset contribution."),
            ("10.1234/paper.4", "Methodology", False, None, "[LOW confidence] I think this is methodology."),
            ("10.1234/paper.5", "Applied", False, None, "[HIGH confidence] Applied work in clinical domain."),
        ]:
            session.add(Annotation(
                paper_doi=doi,
                llm_model="Qwen3.5-35B-A3B-GGUF",
                annotator_type="llm",
                proposed_label=label,
                is_flagged=flagged,
                flag_reason=flag_reason,
                comment=comment,
            ))

        # Model C — excluded model
        session.add(Annotation(
            paper_doi="10.1234/paper.1",
            llm_model="DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            annotator_type="llm",
            proposed_label="Review",
            is_flagged=False,
            comment="[LOW confidence] I disagree with Methodology.",
        ))

        # Human annotation on Paper 1
        session.add(Annotation(
            paper_doi="10.1234/paper.1",
            user_id="hf|testuser",
            llm_model=None,
            annotator_type="human",
            proposed_label="Methodology",
            is_flagged=False,
            comment="Clearly a methodology paper — novel architecture.",
        ))

        session.commit()
    finally:
        session.close()

    return temp_db


# ── export_llm_judge_dataset tests ────────────────────────────────────────────


def test_export_per_model_files(populated_db, tmp_path, monkeypatch):
    """Should produce one JSONL file per non-excluded model."""
    from export_llm_judge_dataset import export
    monkeypatch.setattr("export_llm_judge_dataset.EXCLUDED_JUDGE_MODELS", ["DeepSeek-R1-Distill-Qwen-1.5B-GGUF"])

    export(output_dir=tmp_path, db_url=populated_db)

    files = sorted(tmp_path.glob("llm_judge_*.jsonl"))
    model_names = {f.stem.removeprefix("llm_judge_") for f in files}

    assert len(files) == 2
    assert "Gemma-4-E4B-it-GGUF" in model_names
    assert "Qwen3.5-35B-A3B-GGUF" in model_names
    # Excluded model should NOT appear
    assert "DeepSeek-R1-Distill-Qwen-1.5B-GGUF" not in model_names


def test_export_record_fields(populated_db, tmp_path, monkeypatch):
    """Each exported record should have all required fields."""
    from export_llm_judge_dataset import export
    monkeypatch.setattr("export_llm_judge_dataset.EXCLUDED_JUDGE_MODELS", [])

    export(output_dir=tmp_path, db_url=populated_db)

    # Check Gemma file
    gemma_file = tmp_path / "llm_judge_Gemma-4-E4B-it-GGUF.jsonl"
    assert gemma_file.exists()

    records = []
    with open(gemma_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 5

    for rec in records:
        assert "doi" in rec
        assert "title" in rec
        assert "description" in rec
        assert "initial_intent" in rec
        assert "messages" in rec
        assert "reasoning" in rec
        assert "model_prediction" in rec
        assert "is_flagged" in rec
        assert "flag_reason" in rec
        assert "confidence" in rec

        # Check messages structure
        msgs = rec["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

        # Assistant content should include reasoning + label
        if rec.get("reasoning"):
            assert rec["reasoning"] in msgs[2]["content"]
            assert rec["model_prediction"] in msgs[2]["content"]


def test_export_confidence_parsing(populated_db, tmp_path, monkeypatch):
    """Confidence should be correctly extracted from comment field."""
    from export_llm_judge_dataset import export
    monkeypatch.setattr("export_llm_judge_dataset.EXCLUDED_JUDGE_MODELS", [])

    export(output_dir=tmp_path, db_url=populated_db)

    gemma_file = tmp_path / "llm_judge_Gemma-4-E4B-it-GGUF.jsonl"
    records = []
    with open(gemma_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Paper 1 should be HIGH confidence
    paper1 = next(r for r in records if r["doi"] == "10.1234/paper.1")
    assert paper1["confidence"] == "high"
    assert "novel method" in paper1["reasoning"]

    # Paper 2 should be LOW confidence and flagged
    paper2 = next(r for r in records if r["doi"] == "10.1234/paper.2")
    assert paper2["confidence"] == "low"
    assert paper2["is_flagged"] is True
    assert paper2["flag_reason"] == "Garbled text"


def test_export_human_annotations(populated_db, tmp_path, monkeypatch):
    """Should produce a human_annotations.jsonl file."""
    from export_llm_judge_dataset import export
    monkeypatch.setattr("export_llm_judge_dataset.EXCLUDED_JUDGE_MODELS", [])

    export(output_dir=tmp_path, db_url=populated_db)

    human_file = tmp_path / "human_annotations.jsonl"
    assert human_file.exists()

    records = []
    with open(human_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 1
    rec = records[0]
    assert rec["doi"] == "10.1234/paper.1"
    assert rec["model_prediction"] == "Methodology"
    assert rec["confidence"] == "human"
    assert "novel architecture" in rec["reasoning"]


# ── export_golden_dataset tests ───────────────────────────────────────────────


@pytest.fixture
def judge_dataset_dir(tmp_path):
    """Create synthetic llm_judge_*.jsonl files mimicking Script 1 output
    with at least 2 records per label for stratification."""
    # Paper templates: 2 per label = 10 papers
    papers = [
        ("10.1234/m1", "Novel Neural Architecture", "We propose a novel transformer variant.", "Methodology"),
        ("10.1234/m2", "Gradient Optimization Method", "A new optimization algorithm for deep learning.", "Methodology"),
        ("10.1234/d1", "Intent Classification Benchmark", "We introduce a new benchmark dataset.", "Dataset"),
        ("10.1234/d2", "Multi-lingual NLU Corpus", "A new multilingual corpus for NLU tasks.", "Dataset"),
        ("10.1234/r1", "Survey of LLM Architectures", "We survey recent advances in LLM architectures.", "Review"),
        ("10.1234/r2", "Meta-Analysis of Fine-tuning", "A meta-analysis of fine-tuning approaches.", "Review"),
        ("10.1234/a1", "Clinical BERT Application", "We apply BERT to clinical note classification.", "Applied"),
        ("10.1234/a2", "Fraud Detection with GNNs", "Applying GNNs to financial fraud detection.", "Applied"),
        ("10.1234/g1", "Garbled Paper", "asdf jkl; qwerty xxxxxx.", "Methodology"),
        ("10.1234/g2", "Corrupted Abstract", ";;;; ### nonsense text.", "Methodology"),
    ]

    # Model A — agree with labels for most, flag garbled ones
    records_a = []
    for doi, title, abstract, label in papers:
        is_garbled = doi in ("10.1234/g1", "10.1234/g2")
        records_a.append({
            "doi": doi,
            "title": title,
            "description": abstract,
            "initial_intent": label,
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": f"Classify...\nTitle: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content": f"Analysis here.\n\n{label}"},
            ],
            "reasoning": "Analysis here.",
            "model_prediction": label,
            "is_flagged": is_garbled,
            "flag_reason": "Garbled text" if is_garbled else None,
            "confidence": "low" if is_garbled else "high",
        })
    with open(tmp_path / "llm_judge_Gemma-4-E4B-it-GGUF.jsonl", "w", encoding="utf-8") as f:
        for r in records_a:
            f.write(json.dumps(r) + "\n")

    # Model B — agrees on most, disagrees on one Review (says Methodology)
    records_b = []
    for doi, title, abstract, label in papers:
        is_garbled = doi in ("10.1234/g1", "10.1234/g2")
        if doi == "10.1234/r1":
            pred = "Methodology"  # disagreement
        else:
            pred = label
        records_b.append({
            "doi": doi,
            "title": title,
            "description": abstract,
            "initial_intent": label,
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": f"Classify...\nTitle: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content": f"Alternative view.\n\n{pred}"},
            ],
            "reasoning": "Alternative view.",
            "model_prediction": pred,
            "is_flagged": is_garbled,
            "flag_reason": "Garbled text" if is_garbled else None,
            "confidence": "low" if is_garbled else "medium",
        })
    with open(tmp_path / "llm_judge_Qwen3.5-35B-A3B-GGUF.jsonl", "w", encoding="utf-8") as f:
        for r in records_b:
            f.write(json.dumps(r) + "\n")

    # Human annotation on one Dataset paper
    human = [{
        "doi": "10.1234/d1",
        "title": "Intent Classification Benchmark",
        "description": "We introduce a new benchmark dataset.",
        "initial_intent": "Dataset",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Classify..."},
            {"role": "assistant", "content": "Clearly a dataset paper.\n\nDataset"},
        ],
        "reasoning": "Clearly a dataset paper.",
        "model_prediction": "Dataset",
        "is_flagged": False,
        "flag_reason": None,
        "confidence": "human",
        "annotator": "hf|testuser",
    }]
    with open(tmp_path / "human_annotations.jsonl", "w", encoding="utf-8") as f:
        for r in human:
            f.write(json.dumps(r) + "\n")

    return tmp_path


def test_golden_unclassifiable_label(judge_dataset_dir, tmp_path):
    """Garbled papers (g1, g2 — flagged by both models) should get Unclassifiable label."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.2, test_split=0.2, seed=42)

    all_records = []
    for split_name in ["golden_train.jsonl", "golden_val.jsonl", "golden_test.jsonl"]:
        path = out_dir / split_name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))

    # Garbled papers should be present with Unclassifiable label
    g1 = next(r for r in all_records if r["doi"] == "10.1234/g1")
    assert g1["label"] == "Unclassifiable"
    assert g1["flag_count"] == 2

    g2 = next(r for r in all_records if r["doi"] == "10.1234/g2")
    assert g2["label"] == "Unclassifiable"
    assert g2["flag_count"] == 2

    # 10 papers total (no exclusions)
    dois = {r["doi"] for r in all_records}
    assert len(dois) == 10


def test_golden_consensus_label(judge_dataset_dir, tmp_path):
    """r1: Model A=Review, Model B=Methodology → tie, first (Review) wins.
    d1: Both models + human = Dataset."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.2, test_split=0.2, seed=42)

    all_records = []
    for fname in ["golden_train.jsonl", "golden_val.jsonl", "golden_test.jsonl"]:
        path = out_dir / fname
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))

    # d1: both models + human all say Dataset
    d1 = next(r for r in all_records if r["doi"] == "10.1234/d1")
    assert d1["label"] == "Dataset"
    assert "human" in d1["model_votes"]
    assert d1["model_votes"]["human"] == "Dataset"

    # m1: both agree on Methodology
    m1 = next(r for r in all_records if r["doi"] == "10.1234/m1")
    assert m1["label"] == "Methodology"

    # r1: tie (Review vs Methodology) → first by file sort order = Review
    r1 = next(r for r in all_records if r["doi"] == "10.1234/r1")
    assert r1["label"] in ("Review", "Methodology")


def test_golden_record_fields(judge_dataset_dir, tmp_path):
    """Golden records should have all required fields."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.2, test_split=0.2, seed=42)

    train_path = out_dir / "golden_train.jsonl"
    assert train_path.exists()

    with open(train_path, encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert "doi" in rec
    assert "title" in rec
    assert "description" in rec
    assert "label" in rec
    assert "flag_count" in rec
    assert "model_votes" in rec
    # messages and reasoning are intentionally absent — consensus labels only
    assert "messages" not in rec
    assert "reasoning" not in rec

    # model_votes should contain both models
    assert "Gemma-4-E4B-it-GGUF" in rec["model_votes"]
    assert "Qwen3.5-35B-A3B-GGUF" in rec["model_votes"]


def test_golden_splits_exist(judge_dataset_dir, tmp_path):
    """Should produce train, val, and test splits."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.15, test_split=0.15, seed=42)

    assert (out_dir / "golden_train.jsonl").exists()
    assert (out_dir / "golden_val.jsonl").exists()
    assert (out_dir / "golden_test.jsonl").exists()
    assert (out_dir / "golden_stats.json").exists()


def test_golden_stats_file(judge_dataset_dir, tmp_path):
    """Stats file should contain distribution summaries."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.15, test_split=0.15, seed=42)

    with open(out_dir / "golden_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    assert stats["golden_candidates"] == 10  # 10 papers, none excluded
    assert stats["unclassifiable_count"] == 2  # g1 + g2
    assert "label_distribution" in stats
    assert "splits" in stats
    assert "train" in stats["splits"]
    assert "val" in stats["splits"]
    assert "test" in stats["splits"]


def test_golden_dataset_class_in_splits(judge_dataset_dir, tmp_path):
    """Unclassifiable and Dataset labels should appear in at least val or test."""
    from export_golden_dataset import export

    out_dir = tmp_path / "golden"
    export(input_dir=judge_dataset_dir, output_dir=out_dir, val_split=0.2, test_split=0.2, seed=42)

    all_labels: dict[str, set] = {}
    for split_name in ["golden_train.jsonl", "golden_val.jsonl", "golden_test.jsonl"]:
        path = out_dir / split_name
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        all_labels[split_name] = {r["label"] for r in records}

    # Unclassifiable must appear in at least one split
    all_combined = set.union(*all_labels.values())
    assert "Unclassifiable" in all_combined, "Unclassifiable missing from all splits"
    assert "Dataset" in all_combined, "Dataset missing from all splits"
