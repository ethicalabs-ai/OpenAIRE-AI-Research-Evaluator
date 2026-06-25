import logging
import os
import sys

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="tasks.ping_task")
def ping_task() -> str:
    """A dummy task to keep Celery workers functional and verify health."""
    return "pong"


@celery_app.task(name="tasks.classify_and_judge")
def classify_and_judge(
    doi: str,
    title: str,
    abstract: str,
    model: str,
    *,
    initial_intent: str | None = None,
    source: str = "catalog",
) -> dict:
    """
    Classify a paper with Echo-DSRN (if needed) and judge with an LLM.

    Idempotent — skips if already judged by *model* for this *doi*.

    Returns a status dict: {"status": "ok"|"skipped"|"error", ...}
    """
    from database import SessionLocal
    from llm_judge import judge_paper
    from models import Annotation, PaperRecord

    db = SessionLocal()
    try:
        # ── 1. Skip if already judged by this model ────────────────────────
        existing = (
            db.query(Annotation)
            .filter(Annotation.paper_doi == doi, Annotation.llm_model == model)
            .first()
        )
        if existing:
            return {"status": "skipped", "reason": "already-judged", "doi": doi}

        # ── 2. Ensure paper record exists ──────────────────────────────────
        p = db.query(PaperRecord).filter(PaperRecord.doi == doi).first()
        if not p:
            if not initial_intent:
                from intent_classifier import classify_paper

                initial_intent = classify_paper(title, abstract).label
            p = PaperRecord(
                doi=doi,
                title=title,
                abstract=abstract,
                initial_intent=initial_intent,
                source=source,
            )
            db.add(p)
            try:
                db.flush()
            except Exception:
                db.rollback()
                p = db.query(PaperRecord).filter(PaperRecord.doi == doi).first()
                if p is None:
                    return {"status": "error", "reason": "doi-conflict", "doi": doi}
        else:
            initial_intent = p.initial_intent or "Methodology"

        # ── 3. Call LLM judge ──────────────────────────────────────────────
        try:
            verdict = judge_paper(title, abstract, initial_intent, model=model)
        except Exception as e:
            log.error("LLM judge error for %s: %s", doi, e)
            return {"status": "error", "reason": "llm-error", "doi": doi, "detail": str(e)[:200]}

        # ── 4. Persist annotation ──────────────────────────────────────────
        annotation = Annotation(
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

        return {
            "status": "ok",
            "doi": doi,
            "proposed_label": verdict.proposed_label,
            "label_valid": verdict.label_valid,
            "is_flagged": verdict.is_flagged,
            "confidence": verdict.confidence,
        }
    except Exception as e:
        db.rollback()
        log.exception("Unhandled error in classify_and_judge for %s", doi)
        return {"status": "error", "reason": "unhandled", "doi": doi, "detail": str(e)[:200]}
    finally:
        db.close()


@celery_app.task(name="tasks.classify_mcp")
def classify_mcp(title: str, abstract: str) -> dict:
    """
    Classify a paper via MCP — returns label + all probabilities.

    No database writes. Stateless, fast, agent-facing.
    """
    from intent_classifier import classify_paper

    result = classify_paper(title, abstract)
    return {
        "label": result.label,
        "probabilities": result.probabilities,
    }
