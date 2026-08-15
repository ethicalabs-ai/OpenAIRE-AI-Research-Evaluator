#!/usr/bin/env python
"""Backfill a round's stored predictions with the current Echo classifier.

Host-side judge runs fell back to the old model default (v0.1.3, 5-class)
until the code default was fixed, so papers of the new round stored
old-model predictions (no Unclassifiable possible). This script re-classifies
a round's papers with the configured model (INTENT_CLF_PATH) and refreshes
``paper_records.initial_intent`` — classification only, NO LLM judge.

Usage (inside the web container, which has the model baked + DB access)::

    python /app/scripts/backfill_initial_intent.py --version v0.1.4 --dry-run
    python /app/scripts/backfill_initial_intent.py --version v0.1.4

Run it when no judge CLI is active. ``--dry-run`` only reports.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import MODEL_VERSION  # noqa: E402
from database import SessionLocal  # noqa: E402
from intent_classifier import classify_paper  # noqa: E402
from models import PaperRecord  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--version",
        default=MODEL_VERSION,
        help=f"round to re-classify (default: active MODEL_VERSION={MODEL_VERSION})",
    )
    ap.add_argument("--limit", type=int, default=0, help="max papers (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        q = db.query(PaperRecord).filter(
            PaperRecord.source != "dataset",
            PaperRecord.model_version == args.version,
        ).order_by(PaperRecord.created_at)
        if args.limit:
            q = q.limit(args.limit)
        papers = q.all()

        if not papers:
            print(f"No papers in round {args.version}. Nothing to do.", flush=True)
            return

        print("Loading intent classifier (first call)…", flush=True)
        before = Counter(p.initial_intent or "NULL" for p in papers)
        changed = failed = 0

        from tqdm import tqdm

        progress = tqdm(
            papers,
            desc=f"Re-classifying {len(papers)} papers ({args.version})",
            unit="paper",
        )
        for p in progress:
            try:
                label = classify_paper(p.title, p.abstract).label
            except Exception as e:  # per-paper: keep going
                failed += 1
                tqdm.write(f"  ! classify failed {p.doi}: {str(e)[:120]}")
                continue
            if label != p.initial_intent:
                changed += 1
                if not args.dry_run:
                    p.initial_intent = label
            progress.set_postfix(changed=changed, failed=failed)
        progress.close()

        if not args.dry_run:
            db.commit()
        after = Counter(p.initial_intent or "NULL" for p in papers)

        print(f"\nBefore: {dict(before)}", flush=True)
        print(f"After : {dict(after)}", flush=True)
        print(f"Changed: {changed}  |  classify failures: {failed}", flush=True)
        if args.dry_run:
            print("DRY RUN — nothing was written.", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
