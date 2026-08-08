#!/usr/bin/env python3
"""
export_llm_judge_dataset.py
────────────────────────────
Export the LLM-as-Judge dataset from the collaborative database.

Produces one JSONL file per LLM model, with ChatML messages (inline
reasoning) plus a separate ``reasoning`` field for downstream flexibility.

Output record format
────────────────────
    {
      "doi": "10.48550/arXiv.2106.01345",
      "title": "A Corpus-free State2Seq User Simulator ...",
      "description": "Recent reinforcement learning algorithms ...",
      "initial_intent": "Methodology",
      "messages": [
        {"role": "system", "content": "You are a multilingual ..."},
        {"role": "user", "content": "Classify the research intent ..."},
        {"role": "assistant", "content": "The paper introduces a new dataset ...\\n\\nDataset"}
      ],
      "reasoning": "The paper introduces a new dataset ...",
      "model_prediction": "Dataset",
      "is_flagged": false,
      "flag_reason": null,
      "confidence": "high"
    }

Stdout summary
──────────────
    Per-model: total annotations, Echo agreement rate, flag count,
    confidence distribution, label distribution.

Excluded models
───────────────
    Controlled by ``EXCLUDED_JUDGE_MODELS`` in ``backend/config.py``.

Usage
─────
    uv run python scripts/export_llm_judge_dataset.py
    uv run python scripts/export_llm_judge_dataset.py --output-dir ./datasets/llm-judge
    uv run python scripts/export_llm_judge_dataset.py --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
_BACKEND_DIR = _SCRIPTS_DIR.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND_DIR.parent / ".env", override=False)
except ImportError:
    pass

from config import EXCLUDED_JUDGE_MODELS  # noqa: E402
from database import SessionLocal  # noqa: E402
from models import Annotation, PaperRecord  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_llm_judge")

# ── ChatML templates (matching prepare_research_intent.py) ────────────────────
SYSTEM_PROMPT = (
    "You are a multilingual research paper intent classifier. "
    "Given a paper title and abstract, classify its primary research intent "
    "as exactly one of: Methodology, Dataset, Review, Applied, Theoretical, "
    "Unclassifiable."
)

USER_TEMPLATE = (
    "Classify the research intent of this paper:\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_comment(comment: str | None) -> tuple[str, str]:
    """Extract (confidence, rationale) from an Annotation.comment field.

    Expected format: ``[HIGH confidence] rationale text...``
    """
    if not comment:
        return "unknown", ""
    if comment.startswith("["):
        end = comment.find(" confidence]")
        if end != -1:
            conf = comment[1:end].lower()
            rationale = comment[end + len(" confidence]"):].strip()
            return conf, rationale
    return "unknown", comment


def _build_messages(title: str, abstract: str, label: str, rationale: str) -> list[dict]:
    """Build ChatML messages with inline reasoning in the assistant content."""
    assistant_content = f"{rationale}\n\n{label}" if rationale else label
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": assistant_content},
    ]


def _query_human_annotations(Session, excluded: set) -> list:
    """Query human annotations from the database, excluding papers judged
    only by excluded models."""
    db = Session()
    try:
        return (
            db.query(Annotation, PaperRecord)
            .join(PaperRecord, Annotation.paper_doi == PaperRecord.doi)
            .filter(Annotation.annotator_type == "human")
            .order_by(PaperRecord.doi)
            .all()
        )
    finally:
        db.close()


def _normalize_model(name: str | None) -> str:
    """Merge MTP and non-MTP variants of the same model.

    ``Gemma-4-12B-it-MTP-GGUF`` and ``Gemma-4-12B-it-GGUF`` produce
    identical predictions — strip ``-MTP`` and treat as one model.
    """
    if not name:
        return "unknown"
    return name.replace("-MTP", "")


# ── Export ────────────────────────────────────────────────────────────────────


def export(output_dir: Path, db_url: str | None = None) -> None:
    """Query annotations, write per-model JSONL files, print summary."""
    if db_url:
        import os as _os
        _os.environ["DATABASE_URL"] = db_url
        import importlib
        import database as _db
        importlib.reload(_db)
        Session = _db.SessionLocal
    else:
        Session = SessionLocal

    excluded = set(EXCLUDED_JUDGE_MODELS)
    if excluded:
        log.info(f"Excluded models: {', '.join(sorted(excluded))}")

    db = Session()
    try:
        rows = (
            db.query(Annotation, PaperRecord)
            .join(PaperRecord, Annotation.paper_doi == PaperRecord.doi)
            .filter(Annotation.annotator_type == "llm")
            .filter(~Annotation.llm_model.in_(excluded))
            .order_by(Annotation.llm_model, PaperRecord.doi)
            .all()
        )
    finally:
        db.close()

    if not rows:
        log.warning("No LLM annotations found in the database.")
        return

    # Group by model
    by_model: dict[str, list[tuple[Annotation, PaperRecord]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()  # (normalized_model, doi) for dedup
    for ann, paper in rows:
        model = _normalize_model(ann.llm_model)
        key = (model, paper.doi)
        if key in seen:
            continue  # MTP/non-MTP duplicate — same predictions
        seen.add(key)
        by_model[model].append((ann, paper))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-model summary header ──────────────────────────────────────────
    all_models = sorted(by_model.keys())
    total_all = len(rows)
    log.info(f"Models found: {len(all_models)}")
    log.info(f"Total annotations: {total_all}")
    log.info(f"Output directory: {output_dir.resolve()}\n")

    print(f"\n{'='*70}")
    print(f"  LLM-as-Judge Dataset Export")
    print(f"  Models: {len(all_models)}  |  Annotations: {total_all:,}")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*70}")

    for model in all_models:
        model_rows = by_model[model]
        n = len(model_rows)
        safe_name = model.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"llm_judge_{safe_name}.jsonl"

        # Stats accumulators
        agreed = 0
        flagged = 0
        conf_counts: Counter = Counter()
        label_counts: Counter = Counter()

        with open(out_path, "w", encoding="utf-8") as f:
            for ann, paper in model_rows:
                conf, rationale = _parse_comment(ann.comment)

                record = {
                    "doi": paper.doi,
                    "title": paper.title,
                    "description": paper.abstract,
                    "initial_intent": paper.initial_intent,
                    "messages": _build_messages(
                        paper.title,
                        paper.abstract,
                        ann.proposed_label or "",
                        rationale,
                    ),
                    "reasoning": rationale or None,
                    "model_prediction": ann.proposed_label,
                    "is_flagged": ann.is_flagged,
                    "flag_reason": ann.flag_reason or None,
                    "confidence": conf,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Accumulate stats
                if ann.proposed_label == paper.initial_intent:
                    agreed += 1
                if ann.is_flagged:
                    flagged += 1
                conf_counts[conf] += 1
                label_counts[ann.proposed_label or "None"] += 1

        # ── Per-model summary ──────────────────────────────────────────
        agree_pct = 100 * agreed / n if n else 0
        print(f"\n  ── {model}  ({n:,} records) ──")
        print(f"     File:        {out_path.name}")
        print(f"     Agreement:   {agreed}/{n} ({agree_pct:.1f}%)")
        print(f"     Flagged:     {flagged} ({100*flagged/n:.1f}%)")
        print(f"     Confidence:  high={conf_counts.get('high',0)}  "
              f"medium={conf_counts.get('medium',0)}  "
              f"low={conf_counts.get('low',0)}  "
              f"unknown={conf_counts.get('unknown',0)}")
        print(f"     Labels:      ", end="")
        label_parts = []
        for lbl in ["Methodology", "Dataset", "Review", "Applied", "Theoretical", "Unclassifiable"]:
            cnt = label_counts.get(lbl, 0)
            if cnt:
                label_parts.append(f"{lbl}={cnt}")
        print(", ".join(label_parts))

    # ── Human annotations export ───────────────────────────────────────
    human_rows = _query_human_annotations(Session, excluded)
    n_human = len(human_rows) if human_rows else 0

    if human_rows:
        human_out = output_dir / "human_annotations.jsonl"
        human_agreed = 0
        human_label_counts: Counter = Counter()

        with open(human_out, "w", encoding="utf-8") as f:
            for ann, paper in human_rows:
                record = {
                    "doi": paper.doi,
                    "title": paper.title,
                    "description": paper.abstract,
                    "initial_intent": paper.initial_intent,
                    "messages": _build_messages(
                        paper.title,
                        paper.abstract,
                        ann.proposed_label or "",
                        ann.comment or "",
                    ),
                    "reasoning": ann.comment or None,
                    "model_prediction": ann.proposed_label,
                    "is_flagged": ann.is_flagged,
                    "flag_reason": ann.flag_reason or None,
                    "confidence": "human",
                    "annotator": ann.user_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if ann.proposed_label == paper.initial_intent:
                    human_agreed += 1
                human_label_counts[ann.proposed_label or "None"] += 1

        n_h = len(human_rows)
        agree_pct = 100 * human_agreed / n_h if n_h else 0
        print(f"\n  ── Human Annotations  ({n_h:,} records) ──")
        print(f"     File:        {human_out.name}")
        print(f"     Agreement:   {human_agreed}/{n_h} ({agree_pct:.1f}%)")
        print(f"     Labels:      ", end="")
        label_parts = []
        for lbl in ["Methodology", "Dataset", "Review", "Applied", "Theoretical", "Unclassifiable"]:
            cnt = human_label_counts.get(lbl, 0)
            if cnt:
                label_parts.append(f"{lbl}={cnt}")
        print(", ".join(label_parts) if label_parts else "none")
    else:
        log.info("No human annotations found — skipping human export.")

    print(f"\n{'='*70}")
    print(f"  Done — {total_all:,} LLM + {n_human:,} human records")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LLM-as-Judge dataset — one JSONL file per model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/llm-judge"),
        help="Directory for per-model JSONL files (default: ./datasets/llm-judge)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override DATABASE_URL (default: from environment)",
    )
    args = parser.parse_args()
    export(args.output_dir, args.db_url)


if __name__ == "__main__":
    main()
