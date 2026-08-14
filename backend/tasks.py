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
    remote_url: str = "",
    remote_api_key: str = "",
) -> dict:
    """
    Classify a paper with Echo-DSRN (if needed) and judge with an LLM.

    When remote_url + remote_api_key are provided, POSTs the verdict to the
    remote server instead of writing the local database.

    Idempotent — skips if already judged by *model* for this *doi*.

    Returns a status dict: {"status": "ok"|"skipped"|"error", ...}
    """
    is_remote = bool(remote_url and remote_api_key)

    from config import MODEL_VERSION
    from database import SessionLocal
    from llm_judge import judge_paper
    from models import Annotation, PaperRecord

    db = SessionLocal() if not is_remote else None
    try:
        # ── 1. Skip if already judged by this model ────────────────────────
        if not is_remote:
            existing = (
                db.query(Annotation)
                .filter(Annotation.paper_doi == doi, Annotation.llm_model == model)
                .first()
            )
            if existing:
                return {"status": "skipped", "reason": "already-judged", "doi": doi}

        # ── 2. Ensure paper record exists ──────────────────────────────────
        if not is_remote:
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
                    model_version=MODEL_VERSION,
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
        else:
            if not initial_intent:
                from intent_classifier import classify_paper

                initial_intent = classify_paper(title, abstract).label

        # ── 3. Call LLM judge ──────────────────────────────────────────────
        try:
            verdict = judge_paper(title, abstract, initial_intent, model=model)
        except Exception as e:
            log.error("LLM judge error for %s: %s", doi, e)
            return {
                "status": "error",
                "reason": "llm-error",
                "doi": doi,
                "detail": str(e)[:200],
            }

        # ── 4. Persist annotation ──────────────────────────────────────────
        if is_remote:
            import requests as _requests

            url = remote_url.rstrip("/") + "/api/annotations/judge"
            payload = {
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "initial_intent": initial_intent,
                "source": source,
                "proposed_label": verdict.proposed_label,
                "is_flagged": verdict.is_flagged,
                "flag_reason": verdict.flag_reason or "",
                "comment": f"[{verdict.confidence.upper()} confidence] {verdict.rationale}",
                "llm_model": model,
            }
            try:
                r = _requests.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {remote_api_key}"},
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()
                return {
                    "status": result.get("status", "error"),
                    "doi": doi,
                    "remote_response": result,
                }
            except Exception as e:
                log.error("Remote POST failed for %s: %s", doi, e)
                return {
                    "status": "error",
                    "reason": "remote-post-failed",
                    "doi": doi,
                    "detail": str(e)[:200],
                }

        annotation = Annotation(
            paper_doi=doi,
            user_id=None,
            llm_model=model,
            annotator_type="llm",
            proposed_label=verdict.proposed_label,
            is_flagged=verdict.is_flagged,
            flag_reason=verdict.flag_reason,
            comment=f"[{verdict.confidence.upper()} confidence] {verdict.rationale}",
            model_version=MODEL_VERSION,
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
        if db is not None:
            db.rollback()
        log.exception("Unhandled error in classify_and_judge for %s", doi)
        return {
            "status": "error",
            "reason": "unhandled",
            "doi": doi,
            "detail": str(e)[:200],
        }
    finally:
        if db is not None:
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
