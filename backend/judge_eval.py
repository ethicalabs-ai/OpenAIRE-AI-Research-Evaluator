#!/usr/bin/env python3
"""
judge_eval.py
─────────────
Evaluate LLM-as-Judge annotations stored in the collaborative DB.
All output is Markdown — pipe to a file or render in any viewer.

Usage
─────
    uv run python backend/judge_eval.py
    uv run python backend/judge_eval.py --model Gemma-4-26B-A4B-it-GGUF
    uv run python backend/judge_eval.py --verbose
    uv run python backend/judge_eval.py > report.md

    # Only analyse dataset-source evaluation records (ground-truth mode):
    uv run python backend/judge_eval.py --paper-source dataset
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, __file__.replace("/judge_eval.py", ""))

from database import SessionLocal

from models import Annotation, PaperRecord

LABELS = ["Methodology", "Dataset", "Review", "Applied", "Theoretical"]


# ── helpers ──────────────────────────────────────────────────────────────────


def _conf(comment: str | None) -> str:
    if comment and comment.startswith("["):
        end = comment.find(" confidence]")
        if end != -1:
            return comment[1:end].lower()
    return "unknown"


def _rationale(comment: str | None) -> str:
    if not comment:
        return ""
    parts = comment.split("] ", 1)
    return parts[-1] if len(parts) > 1 else comment


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| "
        + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers)))
        + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}"


# ── main evaluation ───────────────────────────────────────────────────────────


def evaluate(
    model_filter: str | None = None,
    paper_source: str | None = None,
    verbose: bool = False,
) -> None:
    db = SessionLocal()
    try:
        q = (
            db.query(Annotation, PaperRecord)
            .join(PaperRecord, Annotation.paper_doi == PaperRecord.doi)
            .filter(Annotation.annotator_type == "llm")
        )
        if model_filter:
            q = q.filter(Annotation.llm_model == model_filter)
        if paper_source:
            # 'dataset' → evaluation records only; any other value → live sources
            q = q.filter(PaperRecord.source == paper_source)
        else:
            # Default: exclude evaluation-only records from mixed analysis
            q = q.filter(PaperRecord.source != "dataset")
        rows = q.all()
    finally:
        db.close()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("# LLM-as-Judge Evaluation Report")
    print(f"\n_Generated: {ts}_")
    if model_filter:
        print(f"\n_Model filter: `{model_filter}`_")
    if paper_source:
        print(f"\n_Paper source filter: `{paper_source}`_")

    if not rows:
        print("\n> ⚠️ No LLM annotations found in the database.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    by_model: dict[str, list[tuple]] = defaultdict(list)
    for ann, paper in rows:
        by_model[ann.llm_model or "unknown"].append((ann, paper))

    total = len(rows)
    agreed = sum(1 for ann, p in rows if ann.proposed_label == p.initial_intent)
    flagged = sum(1 for ann, _ in rows if ann.is_flagged)
    conf_counts: Counter = Counter(_conf(ann.comment) for ann, _ in rows)
    disagreements = [
        (ann, p) for ann, p in rows if ann.proposed_label != p.initial_intent
    ]

    confusion: dict[str, Counter] = defaultdict(Counter)
    for ann, paper in rows:
        confusion[paper.initial_intent or "Unknown"][
            ann.proposed_label or "Unknown"
        ] += 1

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n## Summary\n")
    print(
        _md_table(
            ["Metric", "Value"],
            [
                ["Total annotations", str(total)],
                ["Models evaluated", str(len(by_model))],
                ["Echo agreement", f"{agreed}/{total} ({100*agreed/total:.1f}%)"],
                ["Disagreements (Echo corrections)", str(len(disagreements))],
                ["Flagged papers", f"{flagged} ({100*flagged/total:.1f}%)"],
                ["Confidence — high", str(conf_counts.get("high", 0))],
                ["Confidence — medium", str(conf_counts.get("medium", 0))],
                ["Confidence — low", str(conf_counts.get("low", 0))],
            ],
        )
    )

    # ── per-model ─────────────────────────────────────────────────────────────
    print("\n## Per-Model Breakdown\n")
    model_rows = []
    for mdl, mdl_rows in sorted(by_model.items()):
        n = len(mdl_rows)
        a = sum(1 for ann, p in mdl_rows if ann.proposed_label == p.initial_intent)
        f = sum(1 for ann, _ in mdl_rows if ann.is_flagged)
        model_rows.append([mdl, str(n), f"{a}/{n} ({100*a/n:.1f}%)", str(f)])
    print(_md_table(["Model", "Annotations", "Agreement", "Flagged"], model_rows))

    # ── confusion matrix ──────────────────────────────────────────────────────
    print("\n## Confusion Matrix\n")
    print("_Rows = Echo `initial_intent` · Columns = LLM `proposed_label`_\n")
    conf_headers = ["Echo \\ LLM"] + LABELS
    conf_rows = []
    for echo_lbl in LABELS:
        row = [echo_lbl]
        for llm_lbl in LABELS:
            n = confusion[echo_lbl][llm_lbl]
            row.append(str(n) if n else "·")
        conf_rows.append(row)
    print(_md_table(conf_headers, conf_rows))

    # ── disagreements ─────────────────────────────────────────────────────────
    print(f"\n## Disagreements ({len(disagreements)})\n")
    print("_Cases where the LLM judge proposes a different label than Echo._\n")
    if disagreements:
        disag_rows = []
        for ann, paper in disagreements:
            conf = _conf(ann.comment)
            rationale = _rationale(ann.comment)[:120]
            disag_rows.append(
                [
                    (paper.title or "")[:55],
                    paper.initial_intent or "?",
                    ann.proposed_label or "?",
                    conf,
                    ann.llm_model or "?",
                    rationale,
                ]
            )
        print(
            _md_table(
                ["Title", "Echo", "LLM", "Conf", "Model", "Rationale"], disag_rows
            )
        )
    else:
        print("_No disagreements — perfect agreement._")

    # ── verbose full listing ──────────────────────────────────────────────────
    if verbose:
        print(f"\n## Full Annotation Listing ({total})\n")
        full_rows = []
        for ann, paper in rows:
            match = "✅" if ann.proposed_label == paper.initial_intent else "❌"
            full_rows.append(
                [
                    match,
                    (paper.title or "")[:55],
                    paper.initial_intent or "?",
                    ann.proposed_label or "?",
                    "🚩" if ann.is_flagged else "",
                    _conf(ann.comment),
                    ann.llm_model or "?",
                ]
            )
        print(
            _md_table(["", "Title", "Echo", "LLM", "Flag", "Conf", "Model"], full_rows)
        )

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-as-Judge annotations (Markdown output)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a specific LLM model name (default: all)",
    )
    parser.add_argument(
        "--paper-source",
        default=None,
        metavar="SOURCE",
        help=(
            "Restrict analysis to papers from a specific source "
            "(e.g. 'dataset' for ground-truth evaluation records, "
            "'arxiv', 'openaire'). Default: all except 'dataset'."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Append full annotation listing"
    )
    args = parser.parse_args()
    evaluate(
        model_filter=args.model,
        paper_source=args.paper_source,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
