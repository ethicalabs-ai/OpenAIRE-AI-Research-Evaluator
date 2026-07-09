#!/usr/bin/env python3
"""
judge_cli.py  —  LLM-as-Judge batch runner
═══════════════════════════════════════════
Iterates over papers from arXiv or OpenAIRE, classifies each with the
Echo-DSRN intent model, then passes the result to the LLM judge.
Verdicts are saved as LLM annotations in the collaborative database.

Usage examples
──────────────
# Judge 50 papers from arXiv using the default model
uv run python backend/judge_cli.py --source arxiv --n 50

# Judge OpenAIRE papers with a specific model and custom rate limit
uv run python backend/judge_cli.py --source openaire --n 100 \\
    --model Qwen3.5-35B-A3B-GGUF --delay 2.5

# Dry-run: classify + judge but skip DB write
uv run python backend/judge_cli.py --source arxiv --n 10 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path

import requests  # noqa: E402
from config import API_KEY, JUDGE_API_URL, LLM_JUDGE_DEFAULT_MODEL  # noqa: E402
from database import SessionLocal  # noqa: E402
from intent_classifier import classify_paper  # noqa: E402
from llm_judge import JudgeVerdict, judge_paper  # noqa: E402

from models import Annotation as DBAnnotation  # noqa: E402
from models import PaperRecord

# ── Path setup ────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(_BACKEND_DIR))

# Load .env before anything else
try:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND_DIR.parent / ".env", override=False)
except ImportError:
    pass


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("judge_cli")

# ── Constants ─────────────────────────────────────────────────────────────────
ARXIV_NS = "http://www.w3.org/2005/Atom"
ARXIV_API = "https://export.arxiv.org/api/query"
OPENAIRE_API = "https://api.openaire.eu/search/publications"

# Research keywords used to pull diverse papers from external sources.
# Loaded from assets/topics.txt at runtime (container path: /app/assets/topics.txt).

def _load_search_queries() -> list[str]:
    """Load search queries from the topics file, falling back to an empty list."""
    import os
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "assets", "topics.txt"),
        "/app/assets/topics.txt",
    ]
    for path in candidates:
        try:
            with open(path) as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                if queries:
                    return queries
        except FileNotFoundError:
            pass
    return []

SEARCH_QUERIES = _load_search_queries()


# ── Paper fetchers ─────────────────────────────────────────────────────────────


def fetch_arxiv(query: str, max_results: int = 25) -> list[dict]:
    """Fetch papers from arXiv API."""
    params: dict = {
        "search_query": f"ti:{query} OR abs:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
    }

    try:
        r = requests.get(ARXIV_API, params=params, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"arXiv fetch failed for '{query}': {e}")
        return []

    papers = []
    try:
        root = ET.fromstring(r.content)
        for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
            title = (
                (entry.findtext(f"{{{ARXIV_NS}}}title") or "")
                .strip()
                .replace("\n", " ")
            )
            abstract = (
                (entry.findtext(f"{{{ARXIV_NS}}}summary") or "")
                .strip()
                .replace("\n", " ")
            )
            link = next(
                (
                    lnk.get("href", "")
                    for lnk in entry.findall(f"{{{ARXIV_NS}}}link")
                    if lnk.get("rel") == "alternate"
                ),
                "",
            )
            arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else ""
            doi = (
                f"10.48550/arXiv.{arxiv_id}"
                if arxiv_id
                else f"arxiv-{uuid.uuid4().hex[:8]}"
            )
            if title and abstract:
                papers.append(
                    {
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "source": "arxiv",
                    }
                )
    except Exception as e:
        log.warning(f"arXiv parse error: {e}")
    return papers


def fetch_openaire(query: str, max_results: int = 25, *, sort_fresh: bool = False) -> list[dict]:
    """Fetch papers from OpenAIRE API."""
    import re

    params: dict = {"title": query, "format": "json", "page": 1, "size": max_results}
    if sort_fresh:
        params["sortBy"] = "dateofcollection,descending"

    try:
        r = requests.get(OPENAIRE_API, params=params, timeout=20)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"OpenAIRE fetch failed for '{query}': {e}")
        return []

    papers = []
    try:
        results_raw = (
            r.json().get("response", {}).get("results", {}).get("result", []) or []
        )
        if isinstance(results_raw, dict):
            results_raw = [results_raw]
        for res in results_raw:
            meta = res.get("metadata", {}).get("oaf:entity", {}).get("oaf:result", {})
            title_obj = meta.get("title", {})
            title = (
                title_obj[0].get("$", "")
                if isinstance(title_obj, list)
                else title_obj.get("$", "")
            ).strip()
            if not title:
                continue
            desc_obj = meta.get("description", "")
            if isinstance(desc_obj, list):
                abstract = desc_obj[0].get("$", "") if desc_obj else ""
            elif isinstance(desc_obj, dict):
                abstract = desc_obj.get("$", "")
            else:
                abstract = str(desc_obj)
            abstract = re.sub(r"<[^>]+>", "", abstract).strip().replace("\n", " ")
            if not abstract:
                continue
            # Extract DOI
            doi = None
            pid = meta.get("pid", [])
            if not isinstance(pid, list):
                pid = [pid]
            for p in pid:
                if isinstance(p, dict) and p.get("@classid") == "doi":
                    doi = p.get("$", "").strip().removeprefix("https://doi.org/")
            if not doi:
                oa_id = res.get("header", {}).get("dri:objIdentifier", {}).get("$", "")
                doi = (
                    f"openaire-{oa_id}" if oa_id else f"openaire-{uuid.uuid4().hex[:8]}"
                )
            papers.append(
                {"doi": doi, "title": title, "abstract": abstract, "source": "openaire"}
            )
    except Exception as e:
        log.warning(f"OpenAIRE parse error: {e}")
    return papers


# ── Core logic ─────────────────────────────────────────────────────────────────


def post_annotation_remote(
    paper: dict,
    verdict: JudgeVerdict,
    model: str,
    api_url: str,
    api_key: str,
) -> str:
    """POST an annotation to a remote server. Returns 'saved', 'skipped', or 'error'."""
    url = api_url.rstrip("/") + "/api/annotations/judge"
    payload = {
        "doi": paper["doi"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "initial_intent": paper.get("initial_intent", "Methodology"),
        "source": paper.get("source", "arxiv"),
        "proposed_label": verdict.proposed_label,
        "is_flagged": verdict.is_flagged,
        "flag_reason": verdict.flag_reason or "",
        "comment": f"[{verdict.confidence.upper()} confidence] {verdict.rationale}",
        "llm_model": model,
    }
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if r.status_code == 401:
            log.error("  ❌ Remote rejected: invalid API key")
            return "error"
        if not r.ok:
            log.error(f"  ❌ Remote returned {r.status_code}: {r.text[:200]}")
            return "error"
        result = r.json()
        return result.get("status", "error")
    except requests.RequestException as e:
        log.error(f"  ❌ Remote request failed: {e}")
        return "error"


def process_paper(
    paper: dict,
    model: str,
    dry_run: bool,
    db,
    *,
    remote_url: str = "",
    remote_api_key: str = "",
) -> bool:
    """Classify + judge a single paper. Returns True if an annotation was saved."""
    doi = paper["doi"].strip().lower().removeprefix("https://doi.org/")
    title = paper["title"]
    abstract = paper["abstract"]
    is_remote = bool(remote_url and remote_api_key)

    # ── 1. Skip if already judged (local-DB only; remote endpoint is idempotent) ─
    if not is_remote:
        existing = (
            db.query(DBAnnotation)
            .filter(DBAnnotation.paper_doi == doi, DBAnnotation.llm_model == model)
            .first()
        )
        if existing:
            return False

    # ── 2. Determine initial_intent ──────────────────────────────────────────
    if not is_remote:
        # Local mode: ensure paper record exists in the local DB
        p = db.query(PaperRecord).filter(PaperRecord.doi == doi).first()
        if not p:
            initial_intent = (
                paper.get("initial_intent") or classify_paper(title, abstract).label
            )
            p = PaperRecord(
                doi=doi,
                title=title,
                abstract=abstract,
                initial_intent=initial_intent,
                source=paper.get("source", "arxiv"),
            )
            if not dry_run:
                db.add(p)
                try:
                    db.flush()
                except Exception:
                    db.rollback()
                    p = db.query(PaperRecord).filter(PaperRecord.doi == doi).first()
                    if p is None:
                        log.warning(
                            f"  ⚠ Could not resolve DOI conflict for {doi} — skipping"
                        )
                        return False
                    initial_intent = p.initial_intent or "Methodology"
            log.info(f"  📄 New paper: {doi[:60]}  →  {initial_intent}")
        else:
            initial_intent = p.initial_intent or "Methodology"
            log.info(f"  📄 Existing: {doi[:60]}  →  {initial_intent}")
    else:
        # Remote mode: run Echo classification locally; server creates PaperRecord
        initial_intent = paper.get("initial_intent") or classify_paper(title, abstract).label
        log.info(f"  📄 {doi[:60]}  →  {initial_intent}")

    # ── 3. Call LLM judge ────────────────────────────────────────────────────
    try:
        verdict: JudgeVerdict = judge_paper(
            title, abstract, initial_intent, model=model
        )
    except Exception as e:
        log.error(f"  ❌ LLM judge error for {doi}: {e}")
        return False

    log.info(
        f"  🤖 Verdict: valid={verdict.label_valid}  "
        f"label={verdict.proposed_label}  "
        f"flagged={verdict.is_flagged}  "
        f"conf={verdict.confidence}"
    )

    if dry_run:
        return True

    # ── 4. Persist annotation ─────────────────────────────────────────────────
    if is_remote:
        status = post_annotation_remote(paper, verdict, model, remote_url, remote_api_key)
        if status == "saved":
            return True
        if status == "skipped":
            log.info(f"  ⏭️ Already judged (remote): {doi[:60]}")
        return False

    annotation = DBAnnotation(
        paper_doi=doi,
        user_id=None,
        llm_model=model,
        annotator_type="llm",
        proposed_label=verdict.proposed_label,
        is_flagged=verdict.is_flagged,
        flag_reason=verdict.flag_reason,
        comment=f"[{verdict.confidence.upper()} confidence] {verdict.rationale}",
    )
    db.add(annotation)
    db.commit()
    return True


def load_dataset_file(path: str) -> list[dict]:
    """Load a JSONL dataset file for evaluation mode.

    Each line must have: title, abstract, label (ground-truth), doi (optional).
    The ground-truth label is stored as initial_intent — Echo classification is
    skipped entirely.  Source is fixed to 'dataset' so the frontend hides it.
    """
    import json as _json
    import uuid as _uuid

    records: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception as e:
                log.warning(f"  dataset line {i}: parse error — {e}")
                continue
            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            label = (obj.get("label") or "").strip()
            if not title or not abstract or not label:
                log.warning(
                    f"  dataset line {i}: missing title/abstract/label — skipping"
                )
                continue
            doi = (obj.get("doi") or "").strip() or f"dataset-{_uuid.uuid4().hex[:12]}"
            records.append(
                {
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "initial_intent": label,  # ground-truth — skip Echo inference
                    "source": "dataset",  # hidden from frontend
                }
            )
    log.info(f"Dataset file: {len(records)} records loaded from {path}")
    return records


def fetch_remote_catalog(
    api_url: str,
    api_key: str,
    model: str,
    limit: int,
) -> list[dict]:
    """Fetch papers from the remote server, excluding those already judged by model."""
    papers: list[dict] = []
    offset = 0
    page_size = min(limit, 100)

    while len(papers) < limit:
        r = requests.get(
            f"{api_url.rstrip('/')}/api/annotations/papers",
            params={
                "limit": page_size,
                "offset": offset,
                "exclude_model": model,
                "sort_by": "recent",
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if r.status_code == 401:
            log.error("Remote catalog: invalid API key")
            break
        if not r.ok:
            log.error(f"Remote catalog: HTTP {r.status_code}")
            break

        data = r.json()
        page = data.get("papers", [])
        total = data.get("total", 0)

        for p in page:
            papers.append(
                {
                    "doi": p["doi"],
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "initial_intent": p.get("initial_intent", "Methodology"),
                    "source": p.get("source", "catalog"),
                }
            )

        offset += len(page)
        if len(page) < page_size or offset >= total:
            break

    log.info(f"Remote catalog: {len(papers)} papers fetched (excluding already judged by {model})")
    return papers


def run(args: argparse.Namespace) -> None:
    import random

    import llm_judge as _lj
    from database import engine as _engine
    from sqlalchemy import event

    # Override LLM endpoint if provided via CLI
    if args.llm_url:
        _lj.LLM_BASE_URL = args.llm_url  # type: ignore[attr-defined]
        log.info(f"LLM URL   : {args.llm_url}  (CLI override)")
    if args.llm_key:
        _lj.LLM_API_KEY = args.llm_key  # type: ignore[attr-defined]
        log.info("LLM key   : *** (CLI override)")

    # Enable WAL journal mode so multiple CLI sessions can run concurrently
    # without hitting "database is locked" errors.
    @event.listens_for(_engine, "connect")
    def _set_wal(dbapi_conn, _rec):
        try:
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass  # non-SQLite backends ignore this

    model = args.model
    mode = "ASYNC (Celery)" if getattr(args, "async_mode", False) else "SYNC"
    remote_url = getattr(args, "api_url", "") or JUDGE_API_URL
    remote_key = getattr(args, "api_key", "") or API_KEY
    is_remote = bool(getattr(args, "remote", False))
    sort_fresh = getattr(args, "sort", "default") == "fresh"

    if is_remote and not remote_url:
        log.error("--api-url (or JUDGE_API_URL env var) is required with --remote")
        return
    if is_remote and not remote_key:
        log.error("--api-key (or API_KEY env var) is required with --remote")
        return

    log.info("═══ LLM-as-Judge CLI ═══")
    log.info(f"Mode   : {mode}")
    log.info(f"Source : {args.source}")
    log.info(f"Target : {args.n} annotations")
    log.info(f"Model  : {model}")
    log.info(f"Delay  : {args.delay}s between dispatches")
    if args.source == "openaire":
        log.info(f"Sort   : {args.sort}")
    if is_remote:
        log.info(f"Remote : {remote_url}")
    if not getattr(args, "async_mode", False):
        log.info(f"Dry-run: {args.dry_run}")

    if getattr(args, "async_mode", False):
        # ── Async mode: dispatch Celery tasks to the worker ──────────────────
        from tasks import classify_and_judge as _async_task

        dispatched = 0

        if args.source == "dataset":
            if not args.dataset_path:
                log.error("--dataset-path is required when --source=dataset")
                return
            papers = load_dataset_file(args.dataset_path)
            random.shuffle(papers)
            for paper in papers:
                if dispatched >= args.n:
                    break
                _async_task.delay(
                    doi=paper["doi"],
                    title=paper["title"],
                    abstract=paper["abstract"],
                    model=model,
                    initial_intent=paper.get("initial_intent"),
                    source=paper.get("source", "dataset"),
                    remote_url=remote_url if is_remote else "",
                    remote_api_key=remote_key if is_remote else "",
                )
                dispatched += 1
                log.info(f"  📤 Enqueued {dispatched}/{min(args.n, len(papers))}: {paper['doi'][:60]}")
                time.sleep(args.delay)
        elif args.source == "catalog":
            if is_remote:
                # Remote catalog: fetch papers from the remote API
                papers = fetch_remote_catalog(remote_url, remote_key, model, args.n)
                target = args.n if args.n > 0 else len(papers)
                for paper in papers:
                    if dispatched >= target:
                        break
                    _async_task.delay(
                        doi=paper["doi"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        model=model,
                        initial_intent=paper.get("initial_intent", "Methodology"),
                        source=paper.get("source", "catalog"),
                        remote_url=remote_url,
                        remote_api_key=remote_key,
                    )
                    dispatched += 1
                    log.info(f"  📤 Enqueued {dispatched}/{target}: {paper['doi'][:60]}")
                    time.sleep(args.delay)
            else:
                db = SessionLocal()
                try:
                    from models import PaperRecord

                    total = db.query(PaperRecord).count()
                    judged_dois = (
                        db.query(DBAnnotation.paper_doi)
                        .filter(DBAnnotation.llm_model == model)
                        .scalar_subquery()
                    )
                    all_papers = (
                        db.query(PaperRecord)
                        .filter(~PaperRecord.doi.in_(judged_dois))
                        .order_by(PaperRecord.doi)
                        .all()
                    )
                    skipped = total - len(all_papers)
                    if skipped:
                        log.info(
                            f"Catalog filter: {skipped}/{total} already judged by {model} — excluded upfront"
                        )
                    target = args.n if args.n > 0 else len(all_papers)
                    for p in all_papers:
                        if dispatched >= target:
                            break
                        _async_task.delay(
                            doi=p.doi,
                            title=p.title,
                            abstract=p.abstract,
                            model=model,
                            initial_intent=p.initial_intent or "Methodology",
                            source=p.source or "catalog",
                        )
                        dispatched += 1
                        log.info(f"  📤 Enqueued {dispatched}/{target}: {p.doi[:60]}")
                        time.sleep(args.delay)
                finally:
                    db.close()
        else:
            fetcher = fetch_arxiv if args.source == "arxiv" else fetch_openaire
            queries = list(SEARCH_QUERIES)
            random.shuffle(queries)
            for query in queries:
                if dispatched >= args.n:
                    break
                log.info(f"\n🔍 Query: '{query}'")
                if args.source == "openaire":
                    papers = fetcher(query, max_results=min(25, args.n - dispatched + 5), sort_fresh=sort_fresh)
                else:
                    papers = fetcher(query, max_results=min(25, args.n - dispatched + 5))
                random.shuffle(papers)
                for paper in papers:
                    if dispatched >= args.n:
                        break
                    _async_task.delay(
                        doi=paper["doi"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        model=model,
                        source=paper.get("source", "arxiv"),
                        remote_url=remote_url if is_remote else "",
                        remote_api_key=remote_key if is_remote else "",
                    )
                    dispatched += 1
                    log.info(f"  📤 Enqueued {dispatched}/{args.n}: {paper['doi'][:60]}")
                    time.sleep(args.delay)
        log.info(f"\n🏁 Done — {dispatched} tasks dispatched to Celery worker")
        return

    # ── Sync mode ─────────────────────────────────────────────────────────
    db = SessionLocal() if not is_remote else None
    judged = 0

    try:
        if args.source == "dataset":
            # ── Dataset evaluation mode ─────────────────────────────────────
            # Echo classification is skipped; ground-truth label from file is
            # used as initial_intent directly.
            if not args.dataset_path:
                log.error("--dataset-path is required when --source=dataset")
                return
            papers = load_dataset_file(args.dataset_path)
            random.shuffle(papers)
            for paper in papers:
                if judged >= args.n:
                    break
                ok = process_paper(
                    paper, model, args.dry_run, db,
                    remote_url=remote_url, remote_api_key=remote_key,
                )
                if ok:
                    judged += 1
                    log.info(f"  ✅ {judged}/{min(args.n, len(papers))} evaluated")
                time.sleep(args.delay)
        elif args.source == "catalog":
            # ── Catalog mode ────────────────────────────────────────────────
            if is_remote:
                papers = fetch_remote_catalog(remote_url, remote_key, model, args.n)
                target = args.n if args.n > 0 else len(papers)
                for paper in papers:
                    if judged >= target:
                        break
                    ok = process_paper(
                        paper, model, args.dry_run, None,
                        remote_url=remote_url, remote_api_key=remote_key,
                    )
                    if ok:
                        judged += 1
                        log.info(f"  ✅ {judged}/{target} annotated")
                    time.sleep(args.delay)
            else:
                from models import PaperRecord

                total = db.query(PaperRecord).count()
                judged_dois = (
                    db.query(DBAnnotation.paper_doi)
                    .filter(DBAnnotation.llm_model == model)
                    .scalar_subquery()
                )
                all_papers = (
                    db.query(PaperRecord)
                    .filter(~PaperRecord.doi.in_(judged_dois))
                    .order_by(PaperRecord.doi)
                    .all()
                )
                skipped = total - len(all_papers)
                if skipped:
                    log.info(
                        f"Catalog filter: {skipped}/{total} already judged by {model} — excluded upfront"
                    )
                target = args.n if args.n > 0 else len(all_papers)
                for p in all_papers:
                    if judged >= target:
                        break
                    paper = {
                        "doi": p.doi,
                        "title": p.title,
                        "abstract": p.abstract,
                        "initial_intent": p.initial_intent or "Methodology",
                        "source": p.source or "catalog",
                    }
                    ok = process_paper(
                        paper, model, args.dry_run, db,
                        remote_url=remote_url, remote_api_key=remote_key,
                    )
                    if ok:
                        judged += 1
                        log.info(f"  ✅ {judged}/{target} annotated")
                    time.sleep(args.delay)
        else:
            # ── Live source mode (arxiv / openaire) ─────────────────────────
            fetcher = fetch_arxiv if args.source == "arxiv" else fetch_openaire
            queries = list(SEARCH_QUERIES)
            random.shuffle(queries)
            for query in queries:
                if judged >= args.n:
                    break
                log.info(f"\n🔍 Query: '{query}'")
                if args.source == "openaire":
                    papers = fetcher(query, max_results=min(25, args.n - judged + 5), sort_fresh=sort_fresh)
                else:
                    papers = fetcher(query, max_results=min(25, args.n - judged + 5))
                random.shuffle(papers)
                for paper in papers:
                    if judged >= args.n:
                        break
                    ok = process_paper(
                        paper, model, args.dry_run, db,
                        remote_url=remote_url, remote_api_key=remote_key,
                    )
                    if ok:
                        judged += 1
                        log.info(f"  ✅ {judged}/{args.n} annotated")
                    time.sleep(args.delay)
    finally:
        if db is not None:
            db.close()

    log.info(f"\n🏁 Done — {judged} annotations saved (dry_run={args.dry_run})")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: batch-classify papers and persist LLM verdicts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["arxiv", "openaire", "dataset", "catalog"],
        default="openaire",
        help="Paper source: arxiv/openaire fetch live papers; dataset reads a local JSONL file (skips Echo inference); catalog judges all papers registered in the local database.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        metavar="PATH",
        help="Path to JSONL file when --source=dataset. Each line: {title, abstract, label, doi}.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of papers to annotate.",
    )
    parser.add_argument(
        "--model",
        default=LLM_JUDGE_DEFAULT_MODEL,
        help="LLM model ID (any model supported by the server).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between successive LLM API calls (rate limiting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Classify and judge but do NOT write to the database.",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        default=False,
        help="Dispatch Celery tasks instead of running synchronously. Requires Redis + worker.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device for Echo intent classifier (default: auto — cuda if available, else cpu).",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        metavar="URL",
        help="Override LLM base URL (e.g. http://other-host:13305/v1).",
    )
    parser.add_argument(
        "--llm-key",
        default=None,
        metavar="KEY",
        help="Override LLM API key (default: value from .env / config.py).",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help="POST annotations to a remote server instead of writing the local database.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help="Remote judge API base URL (env: JUDGE_API_URL). Required with --remote.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Bearer token for the remote judge API (env: API_KEY). Required with --remote.",
    )
    parser.add_argument(
        "--sort",
        choices=["default", "fresh"],
        default="fresh",
        help="Sort order for OpenAIRE paper discovery: relevance or dateofcollection desc. Only applies to --source openaire.",
    )
    args = parser.parse_args()
    import os as _os

    _os.environ["INTENT_CLF_DEVICE"] = args.device
    run(args)


if __name__ == "__main__":
    main()
