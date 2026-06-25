"""
datasets_scripts/research_intent/analyze_dataset.py
─────────────────────────────────────────────────────────────────────────────
Quality analysis tool for the research-intent classification dataset
produced by prepare_research_intent.py.

Run this script after generating the dataset to verify quality before
starting a training run.  All sections are computed in a single pass
over train.jsonl and val.jsonl — no model is loaded.

Sections reported
─────────────────
  1. Label distribution   — absolute count + % for train and val splits.
                            Bars turn red if a class is severely under-
                            represented (< 5% of the split).
  2. Per-source breakdown  — how many records each data source contributed
                            per label (train only).
  3. Text length stats     — median / min char-counts for title and abstract
                            per label.  Flags abstracts shorter than 100
                            chars in red.
  4. Language breakdown    — reads the pre-computed ``lang`` ISO-639-1 field
                            stored in the JSONL by prepare_research_intent.
                            Shows per-label en / non-en counts and the top
                            non-English languages overall.
                            Requires prepare_research_intent to have been run
                            with langdetect or langid installed; otherwise
                            lang fields default to 'en' and this section
                            will say so.  Pass --no_lang to skip entirely.
  5. Suspicious entries    — flags records with very short titles (< 10
                            chars), very short abstracts (< 80 chars), or
                            titles that look like section headings
                            (e.g. "Introduction", "3.2 Methods").
  6. Random samples        — prints N random train examples per class for
                            a quick human sanity-check (disabled by default;
                            enable with --samples N).

Dependencies
────────────
  Standard library only.  Language detection at analysis time is no longer
  performed here — the ``lang`` field is read directly from the JSONL.
  To populate lang fields, install langdetect before running the prepare
  script:  ``uv add langdetect``

Usage
─────
    # Default (reads ~/.ethicalabs/datasets/research-intent):
    uv run python datasets_scripts/research_intent/analyze_dataset.py

    # Custom directory, 2 sample examples per class:
    uv run python datasets_scripts/research_intent/analyze_dataset.py \\
        --data_dir ~/.ethicalabs/datasets/research-intent \\
        --samples 2

    # Skip language section (instant):
    uv run python datasets_scripts/research_intent/analyze_dataset.py \\
        --no_lang
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

# ── Colour helpers ────────────────────────────────────────────────────────────
_CLR = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def _c(text: str, *codes: str) -> str:
    """Wrap *text* with ANSI colour/style codes and append a reset sequence."""
    return "".join(_CLR[c] for c in codes) + str(text) + _CLR["reset"]


def _bar(count: int, total: int, width: int = 36) -> str:
    """Return a Unicode block-character progress bar of the given *width*."""
    filled = int(width * count / max(total, 1))
    return "█" * filled + "░" * (width - filled)


# ── Language detection ────────────────────────────────────────────────────────


def _detect_lang(text: str) -> str:
    """Return ISO 639-1 language code, or '??' on failure."""
    try:
        from langdetect import LangDetectException, detect

        try:
            return detect(text[:1000])
        except LangDetectException:
            return "??"
    except ImportError:
        pass
    try:
        import langid

        lang, _ = langid.classify(text[:1000])
        return lang
    except ImportError:
        return "??"


# ── Record parsing ────────────────────────────────────────────────────────────


def _parse_record(line: str) -> Optional[dict]:
    """Extract title, abstract, label, source from a JSONL line."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None

    msgs = rec.get("messages", [])
    label = ""
    user_text = ""
    for m in msgs:
        if m.get("role") == "assistant":
            label = m.get("content", "").strip()
        elif m.get("role") == "user":
            user_text = m.get("content", "")

    # Format: "Classify the research intent of this paper:\n\nTitle: ...\nAbstract: ..."
    raw = user_text.replace(
        "Classify the research intent of this paper:\n\n", ""
    ).strip()

    title = ""
    abstract = ""
    title_m = re.search(r"Title:\s*(.+?)(?:\nAbstract:|\Z)", raw, re.DOTALL)
    abs_m = re.search(r"Abstract:\s*(.+)", raw, re.DOTALL)
    if title_m:
        title = title_m.group(1).strip().replace("\n", " ")
    if abs_m:
        abstract = abs_m.group(1).strip().replace("\n", " ")

    # Fallback: first line = title, rest = abstract
    if not title:
        lines = raw.split("\n", 1)
        title = lines[0].strip()
        abstract = lines[1].strip() if len(lines) > 1 else ""

    return {
        "label": label,
        "source": rec.get("source", "unknown"),
        "lang": rec.get("lang", "??"),
        "title": title,
        "abstract": abstract,
        "text": f"{title} {abstract}",
    }


# ── Stats helpers ─────────────────────────────────────────────────────────────


def _length_stats(values: list[int]) -> dict:
    """Return descriptive statistics (min/max/mean/median/stdev) for *values*."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values)),
        "median": round(median(values)),
        "stdev": round(stdev(values)) if len(values) > 1 else 0,
    }


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyse(data_dir: Path, samples_per_class: int, detect_lang: bool) -> None:
    """
    Run all analysis sections over the dataset at *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing ``train.jsonl`` and ``val.jsonl``.
    samples_per_class:
        Number of random training examples to print per label (0 = skip).
    detect_lang:
        If True, print the language-breakdown section using the pre-computed
        ``lang`` field stored in each JSONL record.
    """
    splits = {}
    for fname in ("train.jsonl", "val.jsonl"):
        p = data_dir / fname
        if not p.exists():
            print(_c(f"  ⚠  {fname} not found in {data_dir}", "yellow"))
            continue
        records = []
        for line in p.read_text(encoding="utf-8").splitlines():
            rec = _parse_record(line)
            if rec:
                records.append(rec)
        splits[fname] = records
        print(_c(f"  Loaded {len(records):,} records from {fname}", "grey"))

    if not splits:
        print(_c("No data found.", "red"))
        sys.exit(1)

    all_labels = sorted({r["label"] for rs in splits.values() for r in rs})

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Label distribution
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{_c('━'*70, 'cyan')}")
    print(_c("  LABEL DISTRIBUTION", "bold", "cyan"))
    print(_c("━" * 70, "cyan"))

    for split_name, records in splits.items():
        total = len(records)
        counts = Counter(r["label"] for r in records)
        print(f"\n  {_c(split_name, 'bold')}  ({total:,} records)")
        print(f"  {'Label':15s}  {'Count':>6}  {'%':>5}  Distribution")
        print("  " + "─" * 60)
        for lbl in all_labels:
            n = counts.get(lbl, 0)
            pct = 100 * n / max(total, 1)
            colour = "green" if pct > 15 else ("yellow" if pct > 5 else "red")
            print(f"  {lbl:15s}  {n:6,}  {pct:4.1f}%  {_c(_bar(n, total), colour)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Per-source breakdown
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{_c('━'*70, 'cyan')}")
    print(_c("  PER-SOURCE BREAKDOWN  (train)", "bold", "cyan"))
    print(_c("━" * 70, "cyan"))

    if "train.jsonl" in splits:
        src_label: dict[str, Counter] = defaultdict(Counter)
        for r in splits["train.jsonl"]:
            src_label[r["source"]][r["label"]] += 1
        for src, cnts in sorted(src_label.items()):
            total_src = sum(cnts.values())
            parts = ", ".join(f"{lbl}: {n}" for lbl, n in sorted(cnts.items()))
            print(f"  {_c(src, 'bold'):30s} {total_src:5,}  [{parts}]")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Text length stats
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{_c('━'*70, 'cyan')}")
    print(_c("  TEXT LENGTH STATS  (train, chars)", "bold", "cyan"))
    print(_c("━" * 70, "cyan"))

    if "train.jsonl" in splits:
        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in splits["train.jsonl"]:
            by_label[r["label"]].append(r)

        print(
            f"\n  {'Label':15s}  {'Title med':>9}  {'Abstr med':>9}  {'Abstr min':>9}  {'Short(<100)':>11}"
        )
        print("  " + "─" * 60)
        for lbl in all_labels:
            recs = by_label[lbl]
            t_lens = [len(r["title"]) for r in recs]
            a_lens = [len(r["abstract"]) for r in recs]
            short = sum(1 for length in a_lens if length < 100)
            ts = _length_stats(t_lens)
            as_ = _length_stats(a_lens)
            short_col = _c(f"{short:>5}", "red") if short > 5 else f"{short:>5}"
            print(
                f"  {lbl:15s}  {ts['median']:9,}  {as_['median']:9,}  {as_['min']:9,}  {short_col:>11}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Language detection
    # ─────────────────────────────────────────────────────────────────────────
    if detect_lang:
        print(f"\n{_c('━'*70, 'cyan')}")
        print(
            _c("  LANGUAGE BREAKDOWN  (train, from stored lang field)", "bold", "cyan")
        )
        print(_c("━" * 70, "cyan"))

        if "train.jsonl" in splits:
            by_label: dict[str, list[dict]] = defaultdict(list)
            for r in splits["train.jsonl"]:
                by_label[r["label"]].append(r)

            has_lang_field = any(r["lang"] != "??" for r in splits["train.jsonl"])
            if not has_lang_field:
                print(
                    _c(
                        "  ⚠  No lang field in JSONL — re-run prepare_research_intent.py "
                        "to add language tags at source.",
                        "yellow",
                    )
                )
            else:
                print(
                    f"\n  {'Label':15s}  {'en':>5}  {'non-en':>6}  {'??':>4}  Top non-en langs"
                )
                print("  " + "─" * 60)
                for lbl in all_labels:
                    recs = by_label[lbl]
                    lc = Counter(r["lang"] for r in recs)
                    en = lc.pop("en", 0)
                    unk = lc.pop("??", 0)
                    non_en = sum(lc.values())
                    top = ", ".join(f"{lang}:{n}" for lang, n in lc.most_common(4))
                    ne_col = _c(f"{non_en:6}", "red") if non_en > 10 else f"{non_en:6}"
                    print(f"  {lbl:15s}  {en:5}  {ne_col}  {unk:4}  {top or '-'}")

            # Overall lang distribution
            all_langs = Counter(r["lang"] for r in splits["train.jsonl"])
            total = len(splits["train.jsonl"])
            print("\n  Overall top languages (train):")
            for lang, cnt in all_langs.most_common(10):
                pct = 100 * cnt / max(total, 1)
                print(
                    f"    {lang:6s}  {cnt:5,}  {pct:5.1f}%  {_c(_bar(cnt, total, 20), 'cyan')}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Suspicious entries
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{_c('━'*70, 'cyan')}")
    print(
        _c(
            "  SUSPICIOUS ENTRIES  (abstract < 80 chars or title < 10 chars)",
            "bold",
            "cyan",
        )
    )
    print(_c("━" * 70, "cyan"))

    _section_re = re.compile(
        r"^\s*(\d+[\.\d]*\s+)?(introduction|methods?|results?|discussion|"
        r"conclusion|background|related\s+work|abstract)\s*$",
        re.IGNORECASE,
    )

    flagged = 0
    for split_name, records in splits.items():
        for r in records:
            issues = []
            if len(r["title"]) < 10:
                issues.append("short-title")
            if len(r["abstract"]) < 80:
                issues.append("short-abstract")
            if _section_re.match(r["title"]):
                issues.append("section-heading-title")
            if issues:
                flagged += 1
                if flagged <= 20:  # show first 20
                    print(
                        f"\n  [{split_name}] {_c(r['label'], 'yellow')} | "
                        f"{_c(', '.join(issues), 'red')}"
                    )
                    print(f"    title:    {r['title'][:80]!r}")
                    print(f"    abstract: {r['abstract'][:120]!r}")

    if flagged == 0:
        print(_c("\n  ✅  No suspicious entries found.", "green"))
    else:
        print(
            f"\n  {_c(f'⚠  {flagged} suspicious entries total', 'yellow')} "
            f"(showing first 20)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Random samples per class
    # ─────────────────────────────────────────────────────────────────────────
    if samples_per_class > 0 and "train.jsonl" in splits:
        import random

        rng = random.Random(99)
        by_label = defaultdict(list)
        for r in splits["train.jsonl"]:
            by_label[r["label"]].append(r)

        print(f"\n{_c('━'*70, 'cyan')}")
        print(
            _c(
                f"  RANDOM SAMPLES  ({samples_per_class} per class, train)",
                "bold",
                "cyan",
            )
        )
        print(_c("━" * 70, "cyan"))

        for lbl in all_labels:
            print(f"\n  {_c(lbl, 'bold', 'green')}")
            sample = rng.sample(
                by_label[lbl], min(samples_per_class, len(by_label[lbl]))
            )
            for i, r in enumerate(sample, 1):
                print(f"  {_c(str(i), 'grey')}. [{r['source']}] {r['title'][:80]}")
                print(
                    f"     {r['abstract'][:160]}{'…' if len(r['abstract']) > 160 else ''}"
                )

    print(f"\n{_c('━'*70, 'cyan')}\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """
    CLI entry point.  Parses arguments and delegates to :func:`analyse`.

    Arguments
    ---------
    --data_dir  PATH
        Directory containing train.jsonl and val.jsonl.
        Default: ``~/.ethicalabs/datasets/research-intent``.
    --samples N
        Print N random training examples per class for a quick human
        sanity-check.  0 (default) disables this section.
    --no_lang
        Skip the language-breakdown section.  Useful when the JSONL files
        have not been rebuilt with the lang field yet, or when speed matters.
    """
    parser = argparse.ArgumentParser(
        description="Analyse research-intent dataset (train.jsonl + val.jsonl)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path.home() / ".ethicalabs/datasets/research-intent",
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        metavar="N",
        help="Print N random samples per class (0 = skip)",
    )
    parser.add_argument(
        "--no_lang",
        action="store_true",
        help="Skip language detection (faster)",
    )
    args = parser.parse_args()

    print(f"\n{_c('━'*70, 'bold', 'cyan')}")
    print(_c(f"  Dataset Analysis  →  {args.data_dir}", "bold", "cyan"))
    print(_c("━" * 70, "bold", "cyan"))

    analyse(args.data_dir, args.samples, detect_lang=not args.no_lang)


if __name__ == "__main__":
    main()
