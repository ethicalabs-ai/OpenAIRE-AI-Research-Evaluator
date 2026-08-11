"""OpenAIRE Graph API client for the Echo-DSRN CLF telemetry app.

Fetching logic is adapted from the OpenAIRE-AI-Research-Evaluator project so
this app reuses the same proven parsing pipeline:

- ``backend/judge_cli.py::fetch_openaire``  (result parsing, DOI extraction)
- ``backend/server.py::openaire_random``    (random query/page strategy)
- ``backend/server.py::openaire_search``    (DOI search strategy)

The app only needs two entry points: a random paper (surprise me) and a
lookup by DOI.
"""

import logging
import os
import random
import re
import uuid
from typing import Optional

import requests

log = logging.getLogger(__name__)

OPENAIRE_API = "https://api.openaire.eu/search/publications"
TIMEOUT = 20

# The public OpenAIRE Graph API needs no token for basic search; a token only
# raises the rate limit. Send one as a Bearer header when OPENAIRE_TOKEN is set.
_OPENAIRE_TOKEN = os.environ.get("OPENAIRE_TOKEN", "").strip()

# Queries used by the "pick a random paper" endpoint (same pool as the evaluator).
RANDOM_QUERIES = [
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

_DOI_RE = re.compile(r"^10\.[0-9]{4,}[/.].+")


def _extract_title(title_obj) -> str:
    if isinstance(title_obj, list):
        return (title_obj[0].get("$", "") if title_obj else "").strip()
    return title_obj.get("$", "").strip() if title_obj else ""


def _extract_abstract(desc_obj) -> str:
    if isinstance(desc_obj, list):
        abstract = desc_obj[0].get("$", "") if desc_obj else ""
    elif isinstance(desc_obj, dict):
        abstract = desc_obj.get("$", "")
    else:
        abstract = str(desc_obj)
    return re.sub(r"<[^>]+>", "", abstract).strip().replace("\n", " ")


def _extract_authors(meta) -> list:
    creators = meta.get("creator", [])
    if not isinstance(creators, list):
        creators = [creators]
    authors = []
    for c in creators:
        if isinstance(c, dict) and c.get("$"):
            authors.append(c["$"])
        elif isinstance(c, str) and c:
            authors.append(c)
    return authors


def _extract_doi(meta, res) -> str:
    pid = meta.get("pid", [])
    if not isinstance(pid, list):
        pid = [pid]
    for p in pid:
        if isinstance(p, dict) and p.get("@classid") == "doi":
            doi = p.get("$", "").strip().removeprefix("https://doi.org/")
            if doi:
                return doi
    oa_id = res.get("header", {}).get("dri:objIdentifier", {}).get("$", "")
    return f"openaire-{oa_id}" if oa_id else f"openaire-{uuid.uuid4().hex[:8]}"


def _extract_link(meta, doi: str) -> str:
    if doi and not doi.startswith("openaire-"):
        return f"https://doi.org/{doi}"
    instance = meta.get("instance", {})
    if isinstance(instance, list) and instance:
        return instance[0].get("webresource", {}).get("url", {}).get("$", "")
    if isinstance(instance, dict):
        return instance.get("webresource", {}).get("url", {}).get("$", "")
    return "https://search.openaire.eu"


def parse_result(res) -> Optional[dict]:
    """Parse a single OpenAIRE ``result`` entry into a clean paper dict."""
    if not isinstance(res, dict):
        return None
    meta = res.get("metadata", {}) or {}
    meta = meta.get("oaf:entity", {}) or {}
    meta = meta.get("oaf:result", {}) or {}

    title = _extract_title(meta.get("title", {}))
    abstract = _extract_abstract(meta.get("description", ""))
    if not title or not abstract:
        return None

    doi = _extract_doi(meta, res)
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": _extract_authors(meta),
        "link": _extract_link(meta, doi),
        "source": "openaire",
    }


def parse_results(results_raw) -> list:
    """Parse the OpenAIRE ``response.results.result`` payload into papers."""
    if isinstance(results_raw, dict):
        results_raw = [results_raw]
    papers = []
    for res in results_raw:
        try:
            paper = parse_result(res)
            if paper:
                papers.append(paper)
        except Exception as e:
            log.warning("OpenAIRE parse error: %s", e)
    return papers


def _fetch(params: dict) -> list:
    """GET the OpenAIRE API with the given params and parse the results."""
    headers = {}
    if _OPENAIRE_TOKEN:
        headers["Authorization"] = f"Bearer {_OPENAIRE_TOKEN}"
    try:
        r = requests.get(OPENAIRE_API, params=params, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        log.warning("OpenAIRE fetch failed: %s", e)
        return []

    try:
        data = r.json()
    except Exception as e:
        log.warning("OpenAIRE JSON decode failed: %s", e)
        return []

    results_raw = data.get("response", {}).get("results", {}).get("result", []) or []
    return parse_results(results_raw)


def fetch_random_paper() -> Optional[dict]:
    """Return metadata for a pseudo-random OpenAIRE paper.

    Picks a random query keyword and random page, then returns one parsed
    paper preferring entries with a substantive abstract (>= 50 chars).
    """
    query = random.choice(RANDOM_QUERIES)
    for page in (random.randint(1, 20), 1):  # retry page 1 if a deep page is empty
        papers = _fetch({"keywords": query, "format": "json", "page": page, "size": 10})
        if not papers:
            continue
        random.shuffle(papers)
        good = [p for p in papers if len(p["abstract"]) >= 50]
        return random.choice(good) if good else papers[0]
    return None


def fetch_by_doi(doi: str) -> Optional[dict]:
    """Fetch a single publication by DOI (None if not found or unusable)."""
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    if not doi:
        return None
    papers = _fetch({"doi": doi, "format": "json", "page": 1, "size": 20})
    if not papers:
        return None
    # Prefer an exact DOI match with a real abstract.
    exact = [p for p in papers if p["doi"].lower() == doi.lower() and len(p["abstract"]) >= 50]
    return exact[0] if exact else papers[0]
