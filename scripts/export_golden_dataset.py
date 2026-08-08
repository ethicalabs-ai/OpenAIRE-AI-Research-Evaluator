#!/usr/bin/env python3
"""
export_golden_dataset.py
─────────────────────────
Produce the golden (curated) dataset from the LLM-as-Judge exports.

Requires ``export_llm_judge_dataset.py`` to have been run first.

Pipeline
────────
1. Load all ``llm_judge_*.jsonl`` files from the input directory.
2. Group records by DOI — each paper has N model-judgments.
3. Exclude papers flagged by **more than 1** model (quality filter).
4. Assign a golden label via majority-vote consensus across models.
5. Stratified train / validation / test split, with explicit handling
   for the underrepresented "Dataset" class.

Output
──────
    <output_dir>/golden_train.jsonl
    <output_dir>/golden_val.jsonl
    <output_dir>/golden_test.jsonl
    <output_dir>/golden_stats.json

Golden record format
────────────────────
    {
      "doi": "10.48550/arXiv.2106.01345",
      "title": "A Corpus-free State2Seq User Simulator ...",
      "description": "Recent reinforcement learning algorithms ...",
      "messages": [
        {"role": "system", "content": "You are a multilingual ..."},
        {"role": "user", "content": "Classify the research intent ..."},
        {"role": "assistant", "content": "The paper introduces a new dataset ...\\n\\nDataset"}
      ],
      "reasoning": "The paper introduces a new dataset ...",
      "label": "Dataset",
      "flag_count": 0,
      "model_votes": {
        "Gemma-4-E4B-it-GGUF": "Dataset",
        "Qwen3.5-35B-A3B-GGUF": "Methodology"
      }
    }

Usage
─────
    uv run python scripts/export_golden_dataset.py
    uv run python scripts/export_golden_dataset.py --input-dir ./datasets/llm-judge --output-dir ./datasets/golden
    uv run python scripts/export_golden_dataset.py --val-split 0.15 --test-split 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_golden")

LABELS = ["Methodology", "Dataset", "Review", "Applied", "Theoretical", "Unclassifiable"]
MIN_VAL_PER_CLASS = 10  # floor for validation records per label


# ── Helpers ───────────────────────────────────────────────────────────────────


def _majority_vote(labels: list[str]) -> str | None:
    """Return the most common label, breaking ties by first occurrence."""
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _load_judge_dataset(input_dir: Path) -> dict[str, dict]:
    """Load all llm_judge_*.jsonl and human_annotations.jsonl, grouped by DOI.

    Human annotations are included as additional judgments (model="human")
    so they participate in consensus voting alongside LLM judges.

    Returns
    -------
    dict[doi, dict]
        {
          "doi": str,
          "title": str,
          "description": str,
          "messages": list[dict],   # from first record (template)
          "judgments": list[dict],  # one per model / human
        }
    """
    jsonl_files = sorted(input_dir.glob("llm_judge_*.jsonl"))

    # Include human annotations if available
    human_file = input_dir / "human_annotations.jsonl"
    if human_file.exists():
        jsonl_files.append(human_file)
        log.info(f"Including human annotations: {human_file.name}")

    if not jsonl_files:
        log.error(f"No llm_judge_*.jsonl files found in {input_dir}")
        sys.exit(1)

    log.info(f"Loading {len(jsonl_files)} source files from {input_dir}")

    papers: dict[str, dict] = {}

    for fpath in jsonl_files:
        is_human = fpath.name == "human_annotations.jsonl"
        model_name = "human" if is_human else fpath.stem.removeprefix("llm_judge_")
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doi = rec["doi"]
                count += 1

                if doi not in papers:
                    papers[doi] = {
                        "doi": doi,
                        "title": rec["title"],
                        "description": rec["description"],
                        "messages": rec.get("messages", []),
                        "judgments": [],
                    }

                papers[doi]["judgments"].append({
                    "model": model_name,
                    "model_prediction": rec.get("model_prediction"),
                    "is_flagged": rec.get("is_flagged", False),
                    "flag_reason": rec.get("flag_reason"),
                    "confidence": rec.get("confidence", "unknown"),
                    "reasoning": rec.get("reasoning"),
                })
        log.info(f"  {fpath.name}: {count:,} records")

    log.info(f"Total unique papers: {len(papers):,}")
    return papers


def _stratified_split(
    records: list[dict],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/val/test split with Dataset-class safeguarding.

    Uses two-pass ``train_test_split`` with stratification on the label.
    After splitting, verifies that every label appears in every split and
    the "Dataset" class has at least ``MIN_VAL_PER_CLASS`` validation records.
    If not, moves records from train → val to satisfy the floor.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.error("scikit-learn is required. Install: uv add scikit-learn")
        sys.exit(1)

    labels = [r["label"] for r in records]
    rng = random.Random(seed)
    label_counts = Counter(labels)

    # If any class has < 2 records or the test split would be smaller than
    # the number of classes, stratification is impossible → fall back to
    # simple random split (real datasets will be much larger).
    n_test = max(1, int(len(records) * test_split))
    n_classes = len(label_counts)
    can_stratify = (
        all(cnt >= 2 for cnt in label_counts.values())
        and len(records) >= 4
        and n_test >= n_classes
    )

    if can_stratify:
        train_val, test = train_test_split(
            records,
            test_size=test_split,
            stratify=labels,
            random_state=seed,
        )
        tv_labels = [r["label"] for r in train_val]
        val_ratio = val_split / (1.0 - test_split)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=tv_labels,
            random_state=seed,
        )
    else:
        # Simple random split for small datasets
        log.info("  ⚠ Dataset too small for stratification — using random split")
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_split))
        n_val = max(1, int(len(shuffled) * val_split))
        test = shuffled[:n_test]
        val = shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]

    # ── Dataset-class safeguarding ────────────────────────────────────────
    for split_name, split_data in [("val", val), ("test", test)]:
        split_labels = Counter(r["label"] for r in split_data)
        for lbl in LABELS:
            if split_labels.get(lbl, 0) < MIN_VAL_PER_CLASS:
                needed = MIN_VAL_PER_CLASS - split_labels.get(lbl, 0)
                candidates = [r for r in train if r["label"] == lbl]
                # Never move more than half of train to avoid emptying it
                max_move = max(0, len(train) // 2 - 1)
                capped = min(needed, len(candidates), max_move)
                if capped > 0:
                    moved = rng.sample(candidates, capped)
                    for r in moved:
                        train.remove(r)
                        split_data.append(r)
                    log.info(
                        f"  ⚠ Moved {len(moved)} '{lbl}' records from train → {split_name} "
                        f"(floor: {MIN_VAL_PER_CLASS})"
                    )

    return train, val, test


def _pick_reasoning(judgments: list[dict], consensus_label: str) -> str | None:
    """Return the reasoning from a model whose prediction matches the consensus."""
    for j in judgments:
        if j["model_prediction"] == consensus_label and j.get("reasoning"):
            return j["reasoning"]
    # Fallback: any reasoning
    for j in judgments:
        if j.get("reasoning"):
            return j["reasoning"]
    return None


# ── Messages (matching prepare_research_intent.py) ────────────────────────────

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


def _build_messages(title: str, abstract: str, label: str, reasoning: str | None) -> list[dict]:
    """Build ChatML messages with inline reasoning in the assistant content."""
    assistant_content = f"{reasoning}\n\n{label}" if reasoning else label
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": assistant_content},
    ]


# ── Export ────────────────────────────────────────────────────────────────────


def export(input_dir: Path, output_dir: Path, val_split: float, test_split: float, seed: int) -> None:
    papers = _load_judge_dataset(input_dir)

    # ── Filter & compute golden labels ────────────────────────────────────
    golden: list[dict] = []
    excluded_no_consensus = 0
    label_dist: Counter = Counter()
    flag_dist: Counter = Counter()

    for doi, paper in papers.items():
        judgments = paper["judgments"]
        flag_count = sum(1 for j in judgments if j["is_flagged"])

        # Papers flagged by > 1 model become "Unclassifiable"
        if flag_count > 1:
            flag_dist[flag_count] += 1
            flag_reasons = [j.get("flag_reason") for j in judgments if j.get("flag_reason")]
            reasoning = "; ".join(flag_reasons) if flag_reasons else None

            model_votes = {
                j["model"]: j["model_prediction"]
                for j in judgments
                if j["model_prediction"]
            }

            golden.append({
                "doi": doi,
                "title": paper["title"],
                "description": paper["description"],
                "messages": _build_messages(
                    paper["title"], paper["description"], "Unclassifiable", reasoning,
                ),
                "reasoning": reasoning,
                "label": "Unclassifiable",
                "flag_count": flag_count,
                "model_votes": model_votes,
            })
            label_dist["Unclassifiable"] += 1
            continue

        # Majority-vote consensus
        model_labels = [j["model_prediction"] for j in judgments if j["model_prediction"]]
        consensus = _majority_vote(model_labels)
        if not consensus:
            excluded_no_consensus += 1
            continue

        reasoning = _pick_reasoning(judgments, consensus)

        model_votes = {
            j["model"]: j["model_prediction"]
            for j in judgments
            if j["model_prediction"]
        }

        golden.append({
            "doi": doi,
            "title": paper["title"],
            "description": paper["description"],
            "messages": _build_messages(
                paper["title"], paper["description"], consensus, reasoning,
            ),
            "reasoning": reasoning,
            "label": consensus,
            "flag_count": flag_count,
            "model_votes": model_votes,
        })
        label_dist[consensus] += 1

    log.info(f"Golden candidates: {len(golden):,}")
    log.info(f"Unclassifiable (flags > 1): {label_dist.get('Unclassifiable', 0)}")
    log.info(f"Excluded (no consensus): {excluded_no_consensus}")

    # ── Stratified split ──────────────────────────────────────────────────
    train, val, test = _stratified_split(golden, val_split, test_split, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}

    for name, data in splits.items():
        path = output_dir / f"golden_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(f"  {path.name}: {len(data):,} records")

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = {
        "source": str(input_dir.resolve()),
        "total_papers": len(papers),
        "golden_candidates": len(golden),
        "unclassifiable_count": label_dist.get("Unclassifiable", 0),
        "excluded_no_consensus": excluded_no_consensus,
        "flag_count_distribution": dict(flag_dist),
        "label_distribution": dict(label_dist),
        "splits": {
            name: {
                "count": len(data),
                "labels": dict(Counter(r["label"] for r in data)),
            }
            for name, data in splits.items()
        },
    }
    stats_path = output_dir / "golden_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"  {stats_path.name}: written")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Golden Dataset Export")
    print(f"  Source:  {len(papers):,} papers  →  {len(golden):,} golden")
    print(f"  Output:  {output_dir.resolve()}")
    print(f"{'='*70}")
    print(f"\n  Label distribution (golden):")
    for lbl in LABELS:
        cnt = label_dist.get(lbl, 0)
        pct = 100 * cnt / max(len(golden), 1)
        print(f"    {lbl:15s}  {cnt:6,}  ({pct:5.1f}%)")
    print(f"\n  Split sizes:")
    for name, data in splits.items():
        split_labels = Counter(r["label"] for r in data)
        parts = ", ".join(f"{lbl}={split_labels.get(lbl, 0)}" for lbl in LABELS)
        print(f"    {name:5s}  {len(data):6,}  [{parts}]")
    print(f"\n{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce golden dataset from LLM-as-Judge exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./datasets/llm-judge"),
        help="Directory containing llm_judge_*.jsonl files (default: ./datasets/llm-judge)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/golden"),
        help="Directory for golden_*.jsonl output (default: ./datasets/golden)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    export(args.input_dir, args.output_dir, args.val_split, args.test_split, args.seed)


if __name__ == "__main__":
    main()
