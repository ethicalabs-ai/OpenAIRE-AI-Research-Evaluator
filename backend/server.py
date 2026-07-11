import os
import sys
import time as _time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure backend is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Database & Auth additions for Collaborative Hub
from collections import Counter

import httpx
from auth_utils import (
    COOKIE_NAME,
    create_session_token,
    get_current_user,
    require_current_user,
)
from celery_app import celery_app
from config import API_KEY, EXPORT_ENABLED
from database import get_db
from fastapi import Depends, Request, Response, Header
from intent_classifier import classify_paper, get_classifier, is_loaded
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from models import Annotation as DBAnnotation
from models import PaperRecord
from models import SavedPaper as DBSavedPaper
from models import User as DBUser

app = FastAPI(title="OpenAIRE 2026 — Research Paper Classifier API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secure Uploads and Temp Workspace Directory
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ClassifyRequest(BaseModel):
    title: str
    abstract: str


@app.get("/api/health")
def health_check():
    """Return health status checking connection to Redis broker."""
    try:
        # Ping Redis connection
        celery_app.connection().connect()
        return {
            "status": "healthy",
            "message": "Server active. Redis broker connection online.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to connect to Redis broker.",
                "error": str(e),
            },
        )


# ── arXiv random paper proxy ─────────────────────────────────────────────────


_ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.NE",
    "cs.IR",
    "stat.ML",
    "math.ST",
    "q-bio.QM",
    "econ.EM",
]
_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = "http://www.w3.org/2005/Atom"


@app.get("/api/arxiv/random")
def arxiv_random():
    """Return metadata for a pseudo-random recent arXiv paper.

    Picks a random category and a random offset (0–500) then fetches one
    result from the arXiv Atom API, parses the XML server-side, and returns
    clean JSON.  The frontend never touches arXiv directly (avoids CORS).

    Response::

        {
            "arxiv_id": "2406.12345",
            "title": "...",
            "abstract": "...",
            "authors": ["Alice", "Bob"],
            "link": "https://arxiv.org/abs/2406.12345",
            "category": "cs.LG"
        }
    """
    import random
    import xml.etree.ElementTree as ET

    try:
        import requests as _req
    except ImportError:
        raise HTTPException(status_code=503, detail="requests not installed")

    category = random.choice(_ARXIV_CATEGORIES)
    offset = random.randint(0, 500)

    try:
        r = _req.get(
            _ARXIV_API,
            params={
                "search_query": f"cat:{category}",
                "start": offset,
                "max_results": 1,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv unreachable: {e}")

    try:
        root = ET.fromstring(r.content)
        entry = root.find(f"{{{_ARXIV_NS}}}entry")
        if entry is None:
            raise HTTPException(
                status_code=404, detail="No paper found for this offset"
            )

        title = (
            (entry.findtext(f"{{{_ARXIV_NS}}}title") or "").strip().replace("\n", " ")
        )
        abstract = (
            (entry.findtext(f"{{{_ARXIV_NS}}}summary") or "").strip().replace("\n", " ")
        )
        authors = [
            a.findtext(f"{{{_ARXIV_NS}}}name") or ""
            for a in entry.findall(f"{{{_ARXIV_NS}}}author")
        ]
        # Prefer the abs link
        link = ""
        for lnk in entry.findall(f"{{{_ARXIV_NS}}}link"):
            if lnk.get("rel") == "alternate":
                link = lnk.get("href", "")
                break
        arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else ""

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XML parse error: {e}")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "link": link,
        "category": category,
    }


# ── OpenAIRE random paper proxy ──────────────────────────────────────────────


_OPENAIRE_QUERIES = [
    "deep learning",
    "neural network",
    "transformer model",
    "supervised learning",
    "reinforcement learning",
    "unsupervised learning",
    "natural language processing",
    "computer vision",
    "artificial intelligence",
    "machine learning",
    "large language model",
    "generative adversarial network",
]


@app.get("/api/openaire/random")
def openaire_random():
    """Return metadata for a pseudo-random OpenAIRE paper.

    Picks a random query keyword and random page, then fetches results from the OpenAIRE Graph API,
    parses the JSON response, and returns clean JSON.
    """
    import random
    import re

    try:
        import requests as _req
    except ImportError:
        raise HTTPException(status_code=503, detail="requests not installed")

    query = random.choice(_OPENAIRE_QUERIES)
    page = random.randint(1, 20)

    try:
        r = _req.get(
            "https://api.openaire.eu/search/publications",
            params={
                "keywords": query,
                "format": "json",
                "page": page,
                "size": 10,
            },
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAIRE unreachable: {e}")

    try:
        data = r.json()
        results = data.get("response", {}).get("results", {}).get("result", []) or []
        if not results:
            if page > 1:
                r = _req.get(
                    "https://api.openaire.eu/search/publications",
                    params={
                        "keywords": query,
                        "format": "json",
                        "page": 1,
                        "size": 10,
                    },
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
                results = (
                    data.get("response", {}).get("results", {}).get("result", []) or []
                )

        if not results:
            raise HTTPException(status_code=404, detail="No paper found for this query")

        random.shuffle(results)

        selected = None
        title = ""
        abstract = ""
        authors = []
        link = ""
        lang = "en"

        for res in results:
            try:
                meta = res["metadata"]["oaf:entity"]["oaf:result"]
                title_raw = meta.get("title", [{}])
                t = (
                    (
                        title_raw[0].get("$", "")
                        if isinstance(title_raw, list)
                        else title_raw.get("$", "")
                    )
                    .strip()
                    .replace("\n", " ")
                )

                desc_raw = meta.get("description", [])
                if isinstance(desc_raw, list):
                    ab = (desc_raw[0].get("$", "") if desc_raw else "").strip()
                elif isinstance(desc_raw, dict):
                    ab = desc_raw.get("$", "").strip()
                else:
                    ab = str(desc_raw).strip()

                ab = re.sub(r"<[^>]+>", "", ab).strip().replace("\n", " ")

                if t and ab and len(ab) >= 50:
                    selected = res
                    title = t
                    abstract = ab
                    break
            except Exception:
                continue

        if not selected:
            raise HTTPException(
                status_code=404, detail="No paper with valid title/abstract found"
            )

        meta = selected["metadata"]["oaf:entity"]["oaf:result"]

        # Parse creators (authors)
        creators = meta.get("creator", [])
        if isinstance(creators, list):
            for c in creators:
                if isinstance(c, dict):
                    authors.append(c.get("$", ""))
                else:
                    authors.append(str(c))
        elif isinstance(creators, dict):
            authors.append(creators.get("$", ""))

        authors = [a for a in authors if a]

        # Parse links
        pid = meta.get("pid", {})
        if isinstance(pid, list) and pid:
            for p in pid:
                if isinstance(p, dict) and p.get("@classid") == "doi":
                    link = f"https://doi.org/{p.get('$', '')}"
                    break
        elif isinstance(pid, dict):
            if pid.get("@classid") == "doi":
                link = f"https://doi.org/{pid.get('$', '')}"

        if not link:
            instance = meta.get("instance", {})
            if isinstance(instance, list) and instance:
                link = instance[0].get("webresource", {}).get("url", {}).get("$", "")
            elif isinstance(instance, dict):
                link = instance.get("webresource", {}).get("url", {}).get("$", "")

        # Parse language
        lang_obj = meta.get("language", {})
        if isinstance(lang_obj, dict):
            classid = lang_obj.get("@classid", "eng").lower()
            _iso3_to_2 = {
                "eng": "en",
                "fra": "fr",
                "deu": "de",
                "ita": "it",
                "spa": "es",
                "por": "pt",
                "nld": "nl",
                "pol": "pl",
                "rus": "ru",
                "zho": "zh",
                "ara": "ar",
                "jpn": "ja",
                "kor": "ko",
                "tur": "tr",
                "swe": "sv",
                "nor": "no",
                "dan": "da",
                "fin": "fi",
                "ces": "cs",
                "hun": "hu",
                "ron": "ro",
                "hrv": "hr",
                "slk": "sk",
                "ell": "el",
                "cat": "ca",
                "ukr": "uk",
                "und": "en",
            }
            lang = _iso3_to_2.get(classid, classid[:2] if len(classid) >= 2 else "en")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAIRE parse error: {e}")

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "link": link,
        "category": query,
        "lang": lang,
    }


# ── Streams Search endpoints ──────────────────────────────────────────────────


@app.get("/api/arxiv/search")
def arxiv_search(q: str):
    """Search arXiv by title or DOI and return list of publications."""
    import re as _re
    import uuid
    import xml.etree.ElementTree as ET

    import requests as _req

    if not q or not q.strip():
        return []

    q = q.strip()

    # Build the arXiv search query: detect DOI or arxiv-id patterns
    _doi_arxiv = _re.match(r"(?:10\.48550/arXiv\.)(.+)", q, _re.IGNORECASE)
    _bare_id = _re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", q)
    if _doi_arxiv:
        search_query = f"id:{_doi_arxiv.group(1)}"
    elif _bare_id:
        search_query = f"id:{q}"
    else:
        search_query = f"ti:{q} OR abs:{q}"

    try:
        r = _req.get(
            _ARXIV_API,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": 20,
                "sortBy": "relevance",
            },
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv unreachable: {e}")

    try:
        root = ET.fromstring(r.content)
        entries = root.findall(f"{{{_ARXIV_NS}}}entry")
        results = []
        for entry in entries:
            title = (
                (entry.findtext(f"{{{_ARXIV_NS}}}title") or "")
                .strip()
                .replace("\n", " ")
            )
            abstract = (
                (entry.findtext(f"{{{_ARXIV_NS}}}summary") or "")
                .strip()
                .replace("\n", " ")
            )
            authors = [
                a.findtext(f"{{{_ARXIV_NS}}}name") or ""
                for a in entry.findall(f"{{{_ARXIV_NS}}}author")
            ]
            link = ""
            for lnk in entry.findall(f"{{{_ARXIV_NS}}}link"):
                if lnk.get("rel") == "alternate":
                    link = lnk.get("href", "")
                    break
            arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else ""
            doi = (
                f"10.48550/arXiv.{arxiv_id}"
                if arxiv_id
                else f"arxiv-{uuid.uuid4().hex[:8]}"
            )
            results.append(
                {
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "link": link,
                    "source": "arxiv",
                }
            )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XML parse error: {e}")


@app.get("/api/openaire/search")
def openaire_search(q: str):
    """Search OpenAIRE by title or DOI and return list of publications."""
    import re
    import uuid

    import requests as _req

    if not q or not q.strip():
        return []

    q = q.strip()

    # Detect DOI pattern (starts with 10. followed by registrant/suffix)
    _is_doi = re.match(r"^10\.[0-9]{4,}[/.].+", q)
    params = {
        "format": "json",
        "page": 1,
        "size": 20,
    }
    if _is_doi:
        params["doi"] = q
    else:
        params["title"] = q

    try:
        r = _req.get(
            "https://api.openaire.eu/search/publications",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAIRE unreachable: {e}")

    try:
        data = r.json()
        results_raw = (
            data.get("response", {}).get("results", {}).get("result", []) or []
        )
        if isinstance(results_raw, dict):
            results_raw = [results_raw]

        results = []
        for res in results_raw:
            try:
                metadata = (
                    res.get("metadata", {}).get("oaf:entity", {}).get("oaf:result", {})
                )
                title_obj = metadata.get("title", {})
                if isinstance(title_obj, list):
                    title = title_obj[0].get("$", "") if title_obj else ""
                else:
                    title = title_obj.get("$", "") if title_obj else ""

                if not title:
                    continue

                desc_obj = metadata.get("description", "")
                if isinstance(desc_obj, list):
                    abstract = desc_obj[0].get("$", "") if desc_obj else ""
                elif isinstance(desc_obj, dict):
                    abstract = desc_obj.get("$", "")
                else:
                    abstract = str(desc_obj)

                abstract = re.sub(r"<[^>]+>", "", abstract).strip().replace("\n", " ")

                creator_obj = metadata.get("creator", [])
                if not isinstance(creator_obj, list):
                    creator_obj = [creator_obj]
                authors = []
                for creator in creator_obj:
                    if isinstance(creator, dict) and "$" in creator:
                        authors.append(creator["$"])
                    elif isinstance(creator, str):
                        authors.append(creator)
                authors = [a for a in authors if a]

                pid = metadata.get("pid", [])
                if not isinstance(pid, list):
                    pid = [pid]
                doi = ""
                for p in pid:
                    if isinstance(p, dict) and p.get("@classid") == "doi":
                        doi = p.get("$", "")
                        break

                if not doi:
                    openaire_id = (
                        res.get("header", {}).get("dri:objIdentifier", {}).get("$", "")
                    )
                    doi = (
                        f"openaire-{openaire_id}"
                        if openaire_id
                        else f"openaire-{uuid.uuid4().hex[:8]}"
                    )

                link = (
                    f"https://doi.org/{doi}"
                    if "openaire-" not in doi
                    else "https://search.openaire.eu"
                )

                results.append(
                    {
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "link": link,
                        "source": "openaire",
                    }
                )
            except Exception:
                continue
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAIRE parse error: {e}")


# ── Research-intent classifier endpoints ─────────────────────────────────────


@app.post("/api/classify/intent")
def classify_intent(req: ClassifyRequest):
    """Classify the research intent of a paper from its title and abstract.

    Returns the predicted label and per-class softmax probabilities.
    The model is loaded lazily on first call (~2-3 s) and cached in memory
    for subsequent requests.

    Request body::

        {"title": "...", "abstract": "..."}

    Response::

        {
            "label": "Methodology",
            "probabilities": {
                "Methodology": 0.945,
                "Dataset":     0.004,
                "Review":      0.005,
                "Applied":     0.007,
                "Theoretical": 0.039
            }
        }
    """
    title = req.title.strip()
    abstract = req.abstract.strip()

    if not title:
        raise HTTPException(status_code=400, detail="'title' must not be empty.")
    if not abstract:
        raise HTTPException(status_code=400, detail="'abstract' must not be empty.")
    if len(abstract) < 50:
        raise HTTPException(
            status_code=400,
            detail="'abstract' is too short (<50 chars). Provide the full abstract.",
        )

    try:
        result = classify_paper(title, abstract)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)[:300]}",
        )

    return {
        "label": result.label,
        "probabilities": result.probabilities,
    }


@app.get("/api/model/card")
def get_model_card():
    """Return architecture specifications and parameter counts of the loaded Echo-DSRN model.

    If the model is not loaded, calling this endpoint will trigger lazy loading.
    """
    try:
        model, tokenizer = get_classifier()
        from intent_classifier import _model_path
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded/available: {e}",
        )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())

    embedding_params = 0
    if hasattr(model.model, "embedding"):
        embedding_params = sum(p.numel() for p in model.model.embedding.parameters())
    elif hasattr(model.model, "embed_tokens"):
        embedding_params = sum(p.numel() for p in model.model.embed_tokens.parameters())

    blocks_params = sum(p.numel() for p in model.model.blocks.parameters())

    classifier_params = 0
    if hasattr(model, "classifier"):
        classifier_params = sum(p.numel() for p in model.classifier.parameters())

    config = model.config

    # Precision and device details
    dtype_str = str(next(model.parameters()).dtype).split(".")[-1]
    device_str = str(next(model.parameters()).device)

    # Trainable vs non-trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "model_path": (
            os.path.basename(os.path.normpath(_model_path or ""))
            if _model_path
            else "Unknown"
        ),
        "model_type": getattr(config, "model_type", "echo"),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "total_params": total_params,
        "embedding_params": embedding_params,
        "blocks_params": blocks_params,
        "classifier_params": classifier_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "dtype": dtype_str,
        "device": device_str,
    }


@app.get("/api/classify/intent/health")
def classify_intent_health():
    """Return the load status of the intent classifier.

    Response::

        {"status": "ready",   "model_path": "outputs/echo-research-intent-clf"}
        {"status": "unloaded", "model_path": null}
    """
    from intent_classifier import _model_path

    return {
        "status": "ready" if is_loaded() else "unloaded",
        "model_path": _model_path,
    }


# --- Collaborative Annotation & Auth Endpoints ---


# Pydantic Models for requests
class PaperImportRequest(BaseModel):
    doi: str
    title: str
    abstract: str
    initial_intent: str = None
    source: str = "arxiv"


class VoteRequest(BaseModel):
    doi: str
    proposed_label: str = None
    is_flagged: bool = False
    flag_reason: str = None
    comment: str = None


class ProfileUpdateRequest(BaseModel):
    first_name: str = None
    last_name: str = None
    institution: str = None


import json as _json
from typing import Any, Optional


def _serialize_probabilities(probs: Any) -> Optional[str]:
    """Serialize probabilities to JSON string for DB storage."""
    if probs is None:
        return None
    if isinstance(probs, str):
        return probs
    return _json.dumps(probs)


class SavePaperRequest(BaseModel):
    title: str
    abstract: Optional[str] = None
    label: Optional[str] = None
    probabilities: Optional[Any] = None
    doi: Optional[str] = None
    link: Optional[str] = None
    lang: Optional[str] = None
    openaire: bool = False
    source: str = "unknown"
    timestamp: Optional[int] = None


# Helper to normalize DOIs
def sanitize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def get_consensus_label(annotations) -> str:
    votes = [a.proposed_label for a in annotations if a.proposed_label]
    if not votes:
        return None
    counter = Counter(votes)
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    return None


# Auth endpoints — Hugging Face OAuth
HF_CLIENT_ID = os.getenv("HF_CLIENT_ID")
HF_CLIENT_SECRET = os.getenv("HF_CLIENT_SECRET")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:7860")

if not HF_CLIENT_ID or not HF_CLIENT_SECRET:
    import warnings

    warnings.warn(
        "HF_CLIENT_ID and/or HF_CLIENT_SECRET are not set. "
        "OAuth login will fail. Create an OAuth app at "
        "https://huggingface.co/settings/applications/new and set the "
        "environment variables.",
        stacklevel=1,
    )

# In-memory OAuth state store: state → expiry timestamp

_oauth_states: dict[str, float] = {}


def _cleanup_expired_states():
    now = _time.time()
    expired = [s for s, exp in _oauth_states.items() if exp < now]
    for s in expired:
        del _oauth_states[s]


@app.get("/api/auth/login/hf")
def oauth_login():
    _cleanup_expired_states()
    state = uuid.uuid4().hex
    _oauth_states[state] = _time.time() + 600  # 10-minute expiry

    redirect_uri = f"{APP_BASE_URL}/api/auth/callback/hf"
    auth_url = (
        "https://huggingface.co/oauth/authorize"
        f"?response_type=code"
        f"&client_id={HF_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&scope=openid%20profile%20email"
        f"&state={state}"
    )
    return {"auth_url": auth_url, "state": state}


@app.get("/api/auth/callback/hf")
async def oauth_callback(
    code: str,
    state: str = None,
    db: Session = Depends(get_db),
):
    if not state or state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
    del _oauth_states[state]

    redirect_uri = f"{APP_BASE_URL}/api/auth/callback/hf"

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://huggingface.co/oauth/token",
            data={
                "code": code,
                "client_id": HF_CLIENT_ID,
                "client_secret": HF_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_res.is_error:
            raise HTTPException(
                status_code=400,
                detail=f"OAuth token exchange failed: {token_res.text}",
            )
        tokens = token_res.json()

        profile_res = await client.get(
            "https://huggingface.co/oauth/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if profile_res.is_error:
            raise HTTPException(
                status_code=400,
                detail=f"OAuth userinfo request failed: {profile_res.text}",
            )
        profile = profile_res.json()

    user_id = f"hf|{profile['preferred_username']}"
    user_name = profile.get("name") or profile.get("preferred_username", "HF User")
    user_email = profile.get("email")
    avatar_url = profile.get("picture") or profile.get("avatar_url")

    # Get or create user
    db_user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not db_user:
        name_parts = user_name.split(" ")
        db_user = DBUser(
            id=user_id,
            name=user_name,
            email=user_email,
            avatar_url=avatar_url,
            first_name=name_parts[0],
            last_name=name_parts[1] if len(name_parts) > 1 else "",
            institution="Academic/Independent",
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    else:
        # Update existing user's profile fields on each login
        db_user.name = user_name
        db_user.email = user_email
        db_user.avatar_url = avatar_url
        db.commit()
        db.refresh(db_user)

    token = create_session_token(db_user.id, db_user.name)

    response = RedirectResponse(url="/collab")
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=7 * 24 * 60 * 60,
        samesite="lax",
    )
    return response


@app.get("/api/auth/me")
def get_me(request: Request, db: Session = Depends(get_db)):
    user_info = get_current_user(request)
    if not user_info:
        return {"authenticated": False}

    db_user = db.query(DBUser).filter(DBUser.id == user_info["id"]).first()
    if not db_user:
        return {"authenticated": False}

    return {
        "authenticated": True,
        "id": db_user.id,
        "name": db_user.name,
        "email": db_user.email,
        "first_name": db_user.first_name,
        "last_name": db_user.last_name,
        "institution": db_user.institution,
        "avatar_url": db_user.avatar_url,
    }


@app.get("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"message": "Logged out successfully"}


@app.put("/api/auth/profile")
def update_profile(
    req: ProfileUpdateRequest, request: Request, db: Session = Depends(get_db)
):
    user_info = require_current_user(request)
    db_user = db.query(DBUser).filter(DBUser.id == user_info["id"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if req.first_name is not None:
        db_user.first_name = req.first_name
    if req.last_name is not None:
        db_user.last_name = req.last_name
    if req.institution is not None:
        db_user.institution = req.institution
    db.commit()
    return {"message": "Profile updated"}


# ---------------------------------------------------------------------------
# Saved papers endpoints — synced history for authenticated users
# ---------------------------------------------------------------------------


def _saved_to_dict(sp: DBSavedPaper) -> dict:
    return {
        "id": sp.id,
        "title": sp.title,
        "abstract": sp.abstract,
        "label": sp.label,
        "probabilities": sp.probabilities,
        "doi": sp.doi,
        "link": sp.link,
        "lang": sp.lang,
        "openaire": sp.openaire,
        "source": sp.source,
        "timestamp": sp.timestamp,
    }


@app.get("/api/saved")
def list_saved(request: Request, db: Session = Depends(get_db)):
    user_info = require_current_user(request)
    papers = (
        db.query(DBSavedPaper)
        .filter(DBSavedPaper.user_id == user_info["id"])
        .order_by(DBSavedPaper.id.desc())
        .all()
    )
    return [_saved_to_dict(p) for p in papers]


@app.post("/api/saved")
def save_paper(req: SavePaperRequest, request: Request, db: Session = Depends(get_db)):
    user_info = require_current_user(request)
    existing = (
        db.query(DBSavedPaper)
        .filter(
            DBSavedPaper.user_id == user_info["id"],
            DBSavedPaper.title == req.title,
        )
        .first()
    )
    if existing:
        return _saved_to_dict(existing)

    sp = DBSavedPaper(
        user_id=user_info["id"],
        title=req.title,
        abstract=req.abstract,
        label=req.label,
        probabilities=_serialize_probabilities(req.probabilities),
        doi=req.doi,
        link=req.link,
        lang=req.lang,
        openaire=req.openaire,
        source=req.source,
        timestamp=req.timestamp,
    )
    db.add(sp)
    db.commit()
    db.refresh(sp)
    return _saved_to_dict(sp)


@app.delete("/api/saved/{paper_id}")
def delete_saved(paper_id: int, request: Request, db: Session = Depends(get_db)):
    user_info = require_current_user(request)
    sp = (
        db.query(DBSavedPaper)
        .filter(
            DBSavedPaper.id == paper_id,
            DBSavedPaper.user_id == user_info["id"],
        )
        .first()
    )
    if not sp:
        raise HTTPException(status_code=404, detail="Saved paper not found")
    db.delete(sp)
    db.commit()
    return {"message": "Deleted"}


@app.post("/api/saved/sync")
def sync_saved(
    items: list[SavePaperRequest], request: Request, db: Session = Depends(get_db)
):
    """Upsert items from localStorage into the DB and return the merged list."""
    user_info = require_current_user(request)
    user_id = user_info["id"]

    for item in items:
        existing = (
            db.query(DBSavedPaper)
            .filter(
                DBSavedPaper.user_id == user_id,
                DBSavedPaper.title == item.title,
            )
            .first()
        )
        if existing:
            continue
        sp = DBSavedPaper(
            user_id=user_id,
            title=item.title,
            abstract=item.abstract,
            label=item.label,
            probabilities=_serialize_probabilities(item.probabilities),
            doi=item.doi,
            link=item.link,
            lang=item.lang,
            openaire=item.openaire,
            source=item.source,
            timestamp=item.timestamp,
        )
        db.add(sp)

    db.commit()

    all_papers = (
        db.query(DBSavedPaper)
        .filter(DBSavedPaper.user_id == user_id)
        .order_by(DBSavedPaper.id.desc())
        .all()
    )
    return [_saved_to_dict(p) for p in all_papers]


# Annotation endpoints
@app.get("/api/annotations/papers")
def list_papers(
    limit: int = 10,
    offset: int = 0,
    sort_by: str = "recent",
    label: str = "",
    source: str = "",
    exclude_model: str = "",
    db: Session = Depends(get_db),
):
    # Exclude internal evaluation records (source='dataset') from the graph UI
    base = db.query(PaperRecord).filter(PaperRecord.source != "dataset")

    # Server-side filters
    if label:
        base = base.filter(PaperRecord.initial_intent == label)
    if source:
        base = base.filter(PaperRecord.source == source)
    if exclude_model:
        judged = (
            db.query(DBAnnotation.paper_doi)
            .filter(DBAnnotation.llm_model == exclude_model)
            .subquery()
        )
        base = base.filter(~PaperRecord.doi.in_(judged))

    total = base.count()

    sort_map = {
        "recent": PaperRecord.created_at.desc(),
        "votes": func.count(DBAnnotation.id).desc(),
        "flagged": func.sum(case((DBAnnotation.is_flagged == True, 1), else_=0)).desc(),  # noqa: E712
        "title": PaperRecord.title.asc(),
    }
    order = sort_map.get(sort_by, PaperRecord.created_at.desc())

    papers = (
        base.outerjoin(DBAnnotation, PaperRecord.doi == DBAnnotation.paper_doi)
        .group_by(PaperRecord.doi)
        .order_by(order)
        .offset(offset)
        .limit(limit)
        .all()
    )
    results = []
    for p in papers:
        votes = [a for a in p.annotations if a.proposed_label]
        flags = [a for a in p.annotations if a.is_flagged == True]
        consensus = get_consensus_label(p.annotations)
        has_human = any(
            a.annotator_type == "human" and a.proposed_label for a in p.annotations
        )
        has_llm = any(
            a.annotator_type == "llm" and a.proposed_label for a in p.annotations
        )

        results.append(
            {
                "doi": p.doi,
                "title": p.title,
                "abstract": p.abstract,
                "initial_intent": p.initial_intent,
                "source": p.source,
                "created_at": p.created_at.isoformat(),
                "consensus_label": consensus,
                "vote_count": len(votes),
                "flag_count": len(flags),
                "has_human": has_human,
                "has_llm": has_llm,
            }
        )
    return {"papers": results, "total": total, "limit": limit, "offset": offset}


@app.get("/api/annotations/papers/{doi:path}")
def get_paper_details(doi: str, db: Session = Depends(get_db)):
    clean_doi = sanitize_doi(doi)
    p = db.query(PaperRecord).filter(PaperRecord.doi == clean_doi).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper record not found")

    annotations_list = []
    for a in p.annotations:
        if a.annotator_type == "llm":
            actor = {
                "name": f"🤖 {a.llm_model or 'LLM Judge'}",
                "avatar_url": None,
                "institution": "LLM Judge",
                "first_name": None,
                "last_name": None,
            }
        else:
            actor = {
                "name": a.user.name if a.user else "Unknown",
                "avatar_url": a.user.avatar_url if a.user else None,
                "institution": a.user.institution if a.user else None,
                "first_name": a.user.first_name if a.user else None,
                "last_name": a.user.last_name if a.user else None,
            }
        annotations_list.append(
            {
                "id": a.id,
                "annotator_type": a.annotator_type or "human",
                "llm_model": a.llm_model,
                "user": actor,
                "proposed_label": a.proposed_label,
                "is_flagged": a.is_flagged,
                "flag_reason": a.flag_reason,
                "comment": a.comment,
                "updated_at": a.updated_at.isoformat() if a.updated_at else None,
            }
        )

    consensus = get_consensus_label(p.annotations)
    return {
        "doi": p.doi,
        "title": p.title,
        "abstract": p.abstract,
        "initial_intent": p.initial_intent,
        "source": p.source,
        "consensus_label": consensus,
        "annotations": annotations_list,
    }


@app.post("/api/annotations/papers")
def import_paper(req: PaperImportRequest, db: Session = Depends(get_db)):
    clean_doi = sanitize_doi(req.doi)
    if not clean_doi:
        raise HTTPException(
            status_code=400, detail="A valid DOI is required for annotation"
        )

    p = db.query(PaperRecord).filter(PaperRecord.doi == clean_doi).first()
    if not p:
        p = PaperRecord(
            doi=clean_doi,
            title=req.title,
            abstract=req.abstract,
            initial_intent=req.initial_intent,
            source=req.source,
        )
        db.add(p)
        db.commit()
        db.refresh(p)
    return {"message": "Paper imported successfully", "doi": p.doi}


@app.post("/api/annotations/vote")
def cast_vote(req: VoteRequest, request: Request, db: Session = Depends(get_db)):
    user_info = require_current_user(request)
    clean_doi = sanitize_doi(req.doi)

    p = db.query(PaperRecord).filter(PaperRecord.doi == clean_doi).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper record not found")

    # Get or create annotation for this user + paper
    a = (
        db.query(DBAnnotation)
        .filter(
            DBAnnotation.paper_doi == clean_doi, DBAnnotation.user_id == user_info["id"]
        )
        .first()
    )

    if not a:
        a = DBAnnotation(paper_doi=clean_doi, user_id=user_info["id"])
        db.add(a)

    a.proposed_label = req.proposed_label
    a.is_flagged = req.is_flagged
    a.flag_reason = req.flag_reason
    a.comment = req.comment

    db.commit()
    return {"message": "Vote registered successfully"}


class JudgeAnnotationRequest(BaseModel):
    doi: str
    title: str
    abstract: str
    initial_intent: Optional[str] = None
    source: str = "arxiv"
    proposed_label: Optional[str] = None
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    comment: Optional[str] = None
    llm_model: str


@app.post("/api/annotations/judge")
def judge_annotation(
    req: JudgeAnnotationRequest,
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    """Persist an LLM judge verdict. Bearer-token authenticated.

    Creates the PaperRecord if it doesn't exist and upserts the annotation.
    One annotation per (paper_doi, llm_model) — overwrites any existing.
    """
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    clean_doi = sanitize_doi(req.doi)
    if not clean_doi:
        raise HTTPException(status_code=400, detail="A valid DOI is required")

    # Ensure paper record exists
    p = db.query(PaperRecord).filter(PaperRecord.doi == clean_doi).first()
    if not p:
        p = PaperRecord(
            doi=clean_doi,
            title=req.title,
            abstract=req.abstract,
            initial_intent=req.initial_intent,
            source=req.source,
        )
        db.add(p)
        db.flush()

    # Upsert: overwrite existing annotation for this (doi, model)
    existing = (
        db.query(DBAnnotation)
        .filter(
            DBAnnotation.paper_doi == clean_doi,
            DBAnnotation.llm_model == req.llm_model,
        )
        .first()
    )
    if existing:
        existing.proposed_label = req.proposed_label
        existing.is_flagged = req.is_flagged
        existing.flag_reason = req.flag_reason
        existing.comment = req.comment
    else:
        annotation = DBAnnotation(
            paper_doi=clean_doi,
            user_id=None,
            llm_model=req.llm_model,
            annotator_type="llm",
            proposed_label=req.proposed_label,
            is_flagged=req.is_flagged,
            flag_reason=req.flag_reason,
            comment=req.comment,
        )
        db.add(annotation)
    db.commit()
    return {"status": "saved"}


# ── Stats cache ───────────────────────────────────────────────────────────────

_STATS_CACHE_TTL = 300  # 5 minutes

def _get_redis():
    """Lazy Redis connection, reusing Celery REDIS_URL."""
    try:
        from celery_app import REDIS_URL
        import redis
        return redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    except Exception:
        return None


def _cached(key: str, ttl: int, compute):
    """Return cached value or compute + store with TTL."""
    r = _get_redis()
    if r:
        try:
            cached = r.get(key)
            if cached is not None:
                return _json.loads(cached)
        except Exception:
            pass
    result = compute()
    if r:
        try:
            r.setex(key, ttl, _json.dumps(result))
        except Exception:
            pass
    return result


@app.get("/api/annotations/stats")
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func

    def _compute():
        base = db.query(PaperRecord).filter(PaperRecord.source != "dataset")

        total_papers = base.count()
        total_users = db.query(DBUser).count()

        # Flagged: papers with > 3 flag annotations — single subquery
        flag_sub = (
            db.query(DBAnnotation.paper_doi, func.count().label("flag_count"))
            .filter(DBAnnotation.is_flagged == True)  # noqa: E712
            .group_by(DBAnnotation.paper_doi)
            .having(func.count() > 3)
            .subquery()
        )
        flagged_count = base.filter(
            PaperRecord.doi.in_(db.query(flag_sub.c.paper_doi))
        ).count()

        # Consensus distribution — fetch (doi, label) pairs, compute mode per paper
        rows = (
            db.query(DBAnnotation.paper_doi, DBAnnotation.proposed_label)
            .filter(DBAnnotation.proposed_label.isnot(None))
            .all()
        )
        from collections import defaultdict
        by_paper: dict[str, list[str]] = defaultdict(list)
        for doi, label in rows:
            by_paper[doi].append(label)

        dist: dict[str, int] = defaultdict(int)
        for labels in by_paper.values():
            label_counts = Counter(labels)
            top = label_counts.most_common(2)
            if len(top) == 1 or (len(top) == 2 and top[0][1] > top[1][1]):
                dist[top[0][0]] += 1

        # Papers without annotations — use initial_intent
        unannotated = (
            db.query(PaperRecord.initial_intent, func.count())
            .outerjoin(DBAnnotation, PaperRecord.doi == DBAnnotation.paper_doi)
            .filter(PaperRecord.source != "dataset", DBAnnotation.id.is_(None), PaperRecord.initial_intent.isnot(None))
            .group_by(PaperRecord.initial_intent)
            .all()
        )
        for label, cnt in unannotated:
            dist[label] += cnt

        human_annotations = (
            db.query(DBAnnotation).filter(DBAnnotation.annotator_type == "human").count()
        )
        llm_annotations = (
            db.query(DBAnnotation).filter(DBAnnotation.annotator_type == "llm").count()
        )

        return {
            "total_papers": total_papers,
            "total_annotations": human_annotations + llm_annotations,
            "human_annotations": human_annotations,
            "llm_annotations": llm_annotations,
            "total_users": total_users,
            "flagged_papers": flagged_count,
            "consensus_distribution": dist,
        }

    return _cached("stats:annotations", _STATS_CACHE_TTL, _compute)


@app.get("/api/export/status")
def export_status():
    return {
        "enabled": EXPORT_ENABLED,
        "message": (
            None
            if EXPORT_ENABLED
            else "The community-curated dataset will be available once we have reached enough human annotations. Check back soon."
        ),
    }


@app.get("/api/annotations/export")
def export_dataset(db: Session = Depends(get_db)):
    if not EXPORT_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="The community-curated dataset will be available once we have reached enough human annotations.",
        )
    # Exclude internal evaluation records from exports
    papers = db.query(PaperRecord).filter(PaperRecord.source != "dataset").all()
    lines = []
    for p in papers:
        flags = [a for a in p.annotations if a.is_flagged == True]
        if len(flags) > 3:
            continue  # Skip quarantined papers

        consensus = get_consensus_label(p.annotations)
        final_intent = consensus if consensus else p.initial_intent

        # Build chat format sample matching prepared datasets
        lines.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research paper intent classifier. Answer with a single word label.",
                    },
                    {
                        "role": "user",
                        "content": f"Title: {p.title}\nAbstract: {p.abstract}",
                    },
                    {"role": "assistant", "content": final_intent},
                ],
                "metadata": {"doi": p.doi, "votes": len(p.annotations)},
            }
        )
    return lines


# Serve Static files if directory exists
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist"
)

if os.path.exists(STATIC_DIR):
    assets_dir = os.path.join(STATIC_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# =============================================================================
# MCP (Model Context Protocol) — agent-facing classification via SSE
# =============================================================================

from fastapi.responses import StreamingResponse
import asyncio


def _sse_event(event: str, data: dict | str) -> str:
    """Format an SSE event."""
    payload = _json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/api/mcp/classify")
async def mcp_classify(
    title: str,
    abstract: str,
    authorization: str = Header(default=""),
):
    """
    MCP SSE endpoint — classify a paper and stream the result.

    Query params:
      - title: paper title
      - abstract: paper abstract

    Auth: Bearer token via Authorization header.
    """
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        return JSONResponse(
            {"error": "invalid api_key"}, status_code=401
        )

    from tasks import classify_mcp as _task

    async def event_stream():
        yield _sse_event("status", {"state": "processing"})
        task = _task.delay(title=title, abstract=abstract)
        for _ in range(300):
            if task.ready():
                result = task.get(timeout=5)
                yield _sse_event("result", result)
                yield _sse_event("status", {"state": "done"})
                return
            await asyncio.sleep(0.1)
        yield _sse_event("error", {"message": "timeout"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── SPA fallback (must be last) ───────────────────────────────────────────────
@app.get("/{fallback_path:path}")
async def serve_spa(fallback_path: str):
    """Serve SPA index.html fallback for vue-router / history mode."""
    if os.path.exists(STATIC_DIR):
        file_path = os.path.join(STATIC_DIR, fallback_path)
        if fallback_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    return {"message": "Static assets not found. API is active."}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
