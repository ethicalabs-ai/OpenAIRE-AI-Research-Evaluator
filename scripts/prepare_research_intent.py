"""
datasets_scripts/research_intent/prepare_research_intent.py
─────────────────────────────────────────────────────────────────────────────
Build a 5-label research-paper intent classification dataset for fine-tuning
Echo-DSRN-v0.1.3-Intent-CLF via training/train_clf.py.

Output labels
─────────────
    Methodology  — introduces a new method, model, or algorithm
    Dataset      — introduces or documents a dataset or benchmark
    Review       — surveys or synthesises existing work
    Applied      — applies existing methods to a domain problem
    Theoretical  — mathematical or formal analysis, no empirical evaluation

Sources — priority order (gold > near-gold > silver)
──────────────────────────────────────────────────────
    1. PubMed E-utilities API      — GOLD labels for Review and Applied.
                                     Publication-type tags are curated by NLM
                                     indexers: [pt]Review → Review,
                                     [pt]"Systematic Review" → Review,
                                     [pt]"Clinical Trial" → Applied.
                                     Fetches title + abstract via esummary.
                                     Rate limit: 3 req/s without API key.

    2. Semantic Scholar /paper/search — GOLD labels for Dataset, Methodology,
                                     and Theoretical (three independent query
                                     banks, each cross-checked against a
                                     lightweight keyword filter).
                                     Rate limit: 1 req/s public tier.
                                     Breaks immediately on HTTP 400/403/429.

    3. Papers With Code API        — GOLD fallback for Dataset.
                                     The /datasets endpoint lists papers that
                                     introduce a dataset; every result is
                                     definitionally a Dataset paper.

    4. ccdv/arxiv-summarization    — Near-gold for all classes.
                                     Longer, cleaner abstracts than the
                                     streaming dataset; keyword silver-labeling
                                     + arXiv category prior.
                                     ⚠ Abstracts containing LaTeX/markup
                                     artifacts (@xmath, @xcite, etc.) are
                                     rejected by _is_clean_abstract().

    5. gfissore/arxiv-abstracts-2021 — Silver fallback (primary volume source).
                                     Streamed up to --arxiv_stream_limit rows.
                                     Use 1,000,000 when gold sources fall short
                                     (e.g. S2 API blocked, Dataset class
                                     starvation). Same keyword + category
                                     labeling as source 4.

    6. OpenAIRE Graph API          — Silver top-up per class (opt-in via
                                     --openaire). Low yield per query but
                                     useful when all other sources are
                                     exhausted.

    SciCite is intentionally excluded: it contains citation sentences
    (50–300 chars) with section headings as pseudo-titles, which are
    structurally incompatible with the Title+Abstract input format.
    It remains in the codebase behind --include_scicite for experiments.

Language filtering
──────────────────
    Each record receives a `lang` field (ISO 639-1) computed at ingestion via
    the pycld2 C extension (~70 k rec/s). The field is stored in the JSONL
    output and used by analyze_dataset.py. The classifier is English-only;
    non-English records that slip through are a negligible fraction (<0.05%).

Outputs
───────
    <output_dir>/train.jsonl      — 90% stratified split (for trainer)
    <output_dir>/val.jsonl        — 10% held-out validation split
    <output_dir>/stats.json       — per-label / per-source breakdown
    <output_dir>/DONE             — completion sentinel

Usage
─────
    # Dry-run (no files written):
    uv run python datasets_scripts/research_intent/prepare_research_intent.py \\
        --analyze_only

    # Standard build (~6 000 records, all sources):
    uv run python datasets_scripts/research_intent/prepare_research_intent.py \\
        --output_dir ~/.ethicalabs/datasets/research-intent \\
        --samples_per_class 1200

    # ✅ RECOMMENDED — deep arXiv scan for Dataset class recovery:
    #    Use when the Dataset class falls short of 1 200 due to S2 API blocks
    #    or scidocs/PWC failures. Scanning 1 M arXiv rows yields ~840 Dataset
    #    records via silver-label keyword rules (420 per 500 k rows observed).
    uv run python datasets_scripts/research_intent/prepare_research_intent.py \\
        --output_dir ~/.ethicalabs/datasets/research-intent \\
        --samples_per_class 1200 \\
        --arxiv_stream_limit 1000000

    # With OpenAIRE top-up (additional silver fallback):
    uv run python datasets_scripts/research_intent/prepare_research_intent.py \\
        --output_dir ~/.ethicalabs/datasets/research-intent \\
        --samples_per_class 1200 \\
        --arxiv_stream_limit 1000000 \\
        --openaire

Key CLI flags
─────────────
    --output_dir DIR          Where to write train.jsonl / val.jsonl / stats.json.
                              Default: ~/.ethicalabs/datasets/research-intent

    --samples_per_class N     Target records per label before balancing.
                              Default: 1200  (yields 6 000 raw → 5 400 train
                              + 600 val after 90/10 stratified split)

    --arxiv_stream_limit N    Maximum arXiv rows to scan from the streaming
                              gfissore/arxiv-abstracts-2021 dataset.
                              Default: 500 000. Set to 1 000 000 when Dataset
                              class starvation is observed (S2/PWC API dry).

    --openaire                Activate OpenAIRE Graph API top-up pass.
                              Queries each label's OpenAIRE query bank and
                              appends silver records until the per-class target
                              is met. Low yield; use as last resort.

    --skip_arxiv              Skip the arXiv streaming source entirely.
                              Useful for fast iteration when gold sources alone
                              are sufficient (Review + Applied via PubMed).

    --analyze_only            Run source loading and print stats without
                              writing any output files. Use for dry-run
                              validation of source availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Label set & prompt templates
# ─────────────────────────────────────────────────────────────────────────────

LABELS = ["Methodology", "Dataset", "Review", "Applied", "Theoretical"]

SYSTEM_PROMPT = (
    "You are a multilingual research paper intent classifier. "
    "Given a paper title and abstract, classify its primary research intent "
    "as exactly one of: Methodology, Dataset, Review, Applied, Theoretical."
)

USER_TEMPLATE = (
    "Classify the research intent of this paper:\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Source 1: SciCite label mapping
# ─────────────────────────────────────────────────────────────────────────────
# allenai/scicite integer labels:
#   0 = background   → Review   (citing paper uses this work as background)
#   1 = method       → Methodology
#   2 = result       → Applied  (citing paper compares/applies results)
SCICITE_LABEL_MAP: dict[int, str] = {
    0: "Review",
    1: "Methodology",
    2: "Applied",
}

# Section names that strongly indicate a specific override
SECTION_OVERRIDES: dict[str, str] = {
    "methods": "Methodology",
    "method": "Methodology",
    "methodology": "Methodology",
    "related work": "Review",
    "background": "Review",
    "introduction": "Review",
    "results": "Applied",
    "experiments": "Applied",
    "discussion": "Applied",
}


def scicite_label(int_label: int, section: str) -> str:
    base = SCICITE_LABEL_MAP.get(int_label, "Review")
    sec_key = (section or "").lower().strip()
    return SECTION_OVERRIDES.get(sec_key, base)


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: arXiv category → label mapping
# ─────────────────────────────────────────────────────────────────────────────
# Primary arXiv category prefix → candidate label.
# Used as a PRIOR; final label is determined by keyword silver-labeling
# with this as a fallback.
ARXIV_CATEGORY_PRIOR: dict[str, str] = {
    # cs.LG, cs.AI, cs.NE, stat.ML  → often Methodology
    "cs.lg": "Methodology",
    "cs.ai": "Methodology",
    "cs.ne": "Methodology",
    "stat.ml": "Methodology",
    # cs.IR, cs.DB → Applied
    "cs.ir": "Applied",
    "cs.db": "Applied",
    # math.ST, stat.TH, cs.IT, quant-ph  → Theoretical
    "math.st": "Theoretical",
    "stat.th": "Theoretical",
    "cs.it": "Theoretical",
    "quant-ph": "Theoretical",
    "math.pr": "Theoretical",
    "math.oc": "Theoretical",
    # q-bio, econ, cs.CY → Applied
    "q-bio": "Applied",
    "econ": "Applied",
    "cs.cy": "Applied",
    # Domain-specific Applied priors:
    "eess.sp": "Applied",
    "cs.hc": "Applied",
    "cs.ro": "Applied",
    "cs.se": "Applied",
    "cs.ni": "Applied",
    "q-fin": "Applied",
}


def arxiv_prior(categories: str | list) -> str | None:
    """Return a label prior from the primary arXiv category, or None."""
    if not categories:
        return None
    # categories may be a space-delimited string or a list
    if isinstance(categories, list):
        cats = categories
    else:
        cats = categories.strip().split()
    if not cats:
        return None
    primary = cats[0].lower()
    for prefix, label in ARXIV_CATEGORY_PRIOR.items():
        if primary.startswith(prefix):
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Keyword silver-labeling (applies to any text source)
# ─────────────────────────────────────────────────────────────────────────────

THEORETICAL_RULES = [
    re.compile(
        r"\bwe\s+(prove?|establish|derive|show\s+that|demonstrate\s+theoretically)\b"
    ),
    re.compile(
        r"\b(theoretically?|formally?|mathematically)\s+(analyz|show|characteriz|bound|prov)\w*\b"
    ),
    re.compile(
        r"\b(convergence\s+(rate|guarantee|bound)|regret\s+bound|sample\s+complexity|pac\s+learning)\b"
    ),
    re.compile(
        r"\b(information[- ]theoretic|generalization\s+bound|statistical\s+guarantee)\b"
    ),
]

REVIEW_RULES = [
    re.compile(
        r"\b(survey|systematic\s+review|literature\s+review|meta[- ]analysis|scoping\s+review)\b"
    ),
    re.compile(r"\bwe\s+(survey|review|summarize|synthesize|categorize|overview)\b"),
    re.compile(
        r"\bthis\s+(paper|work|article|study)\s+(surveys?|reviews?|provides?\s+(?:a\s+|an\s+)?(overview|survey|review|synthesis))\b"
    ),
    re.compile(r"\b(comprehensive|exhaustive)\s+(overview|review|survey|comparison)\b"),
]

APPLIED_RULES = [
    re.compile(
        r"\bwe\s+(apply|use|adapt|fine[- ]tune|deploy|test|evaluate)\b.{0,80}\b(clinical|medical|legal|financial|industrial|manufacturing|agricultural|educational|geological|archaeological|astronomical|traffic|climate|weather|biomedical|healthcare|biological|chemical|materials)\b"
    ),
    re.compile(
        r"\b(case\s+study|real[- ]world\s+application|system\s+demonstration|pilot\s+study|proof[- ]of[- ]concept\s+on|ablation\s+study\s+on)\b"
    ),
    re.compile(
        r"\b(off[- ]the[- ]shelf|pre[- ]trained\s+(?:model|backbone))\b.{0,60}\b(we\s+(apply|use|adapt|fine[- ]tune))\b"
    ),
    re.compile(
        r"\b(we\s+evaluate)\b.{0,60}\b(existing|baseline|state[- ]of[- ]the[- ]art|prior|previous)\b.{0,40}\b(method|model|approach|system)\b"
    ),
    re.compile(
        r"\bwe\s+(apply|use|employ|utilize|adapt|fine[- ]tune|deploy)\b.{0,60}\b(to|for|in)\b.{0,40}\b(domain|task|problem|application|clinical|medical|legal)\b"
    ),
    re.compile(
        r"\b(real[- ]world\s+(deployment|application|evaluation|setting)|production\s+system)\b"
    ),
    re.compile(
        r"\b(clinical|medical|legal|financial|industrial)\b.{0,60}\b(we\s+(apply|use|adapt|evaluate|test))\b"
    ),
]

DATASET_RELEASE_RE = re.compile(
    r"\b(we\s+(?:publicly\s+|freely\s+|openly\s+)?release|"
    r"make\s+(?:it\s+|our\s+|the\s+|this\s+)?(?:publicly\s+|freely\s+)?available|"
    r"available\s+(?:for\s+download|at|on|via\s+https?://)|"
    r"https?://|"
    r"github\.com|huggingface\.co|zenodo\.org|figshare|gitlab|bitbucket|"
    r"data\s+(?:statement|availability|sharing)|"
    r"download|source\s+code|"
    r"we\s+(?:share|provide|distribute)\s+(?:our|the|this)\s+(?:dataset|corpus|benchmark|annotations))\b",
    re.IGNORECASE,
)

DATASET_TITLE_RULES = [
    re.compile(
        r"\b(dataset|benchmark|corpus|testbed|evaluation\s+suite)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:new\s+|novel\s+|large[- ]scale\s+|multilingual\s+|open[- ]source\s+|comprehensive\s+)"
        r"(collection|knowledge\s+base)\b",
        re.IGNORECASE,
    ),
]

DATASET_BODY_RULES = [
    re.compile(
        r"\bwe\s+(introduce|present|release|publish|describe|contribute|propose)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:new\s+|novel\s+|large[- ]scale\s+|multilingual\s+|open[- ]source\s+)?"
        r"(dataset|benchmark|corpus|collection|resource|evaluation\s+suite|knowledge\s+base|ontology)"
    ),
    re.compile(
        r"\b(leaderboard|shared\s+task|challenge\s+dataset|test\s+bed|annotation\s+scheme)\b"
    ),
    re.compile(
        r"\bwe\s+collect(?:ed)?\b.{0,60}\b(samples?|examples?|documents?|utterances?|instances?)\b"
    ),
    re.compile(
        r"\b(manually\s+annotated|crowd[- ]sourced\s+annotation|inter[- ]annotator\s+agreement)\b"
    ),
    re.compile(
        r"\b(dataset|corpus|benchmark)\s+consists?\s+of\b.{0,60}\b(samples?|documents?|examples?|pairs?|sentences?)\b"
    ),
    re.compile(
        r"\bwe\s+(create|build|construct|compile)\b.{0,60}\b(dataset|corpus|benchmark|collection)\b"
    ),
    re.compile(
        r"\b(introduces?|presents?|releases?|describes?)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:new\s+|novel\s+|large[- ]scale\s+)?"
        r"(dataset|benchmark|corpus|collection)"
    ),
    re.compile(
        r"\b(\d[\d,]*\s+(?:thousand|million|k)\s+)?"
        r"(annotated|labelled?|labeled?)\s+"
        r"(samples?|examples?|documents?|images?|sentences?|utterances?)\b"
    ),
    re.compile(
        r"\bdata\s+(collection|gathering|curation|annotation)\s+(effort|process|pipeline|campaign)\b"
    ),
]

METHODOLOGY_RULES = [
    re.compile(
        r"\bwe\s+(propose|introduce|present|design|develop)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:new\s+|novel\s+)?"
        r"(method|model|architecture|algorithm|framework|approach|system|technique|module|layer|network|pipeline)\b"
    ),
    re.compile(
        r"\b(our\s+(model|method|approach|framework|system|architecture))\s+"
        r"(?:achieves?|outperforms?|surpasses?|sets?\s+(?:a\s+|the\s+)?(?:new\s+)?state[- ]of[- ]the[- ]art)\b"
    ),
    re.compile(r"\bstate[- ]of[- ]the[- ]art\b.{0,50}\bour\b"),
    re.compile(
        r"\bwe\s+(pre[- ]train|fine[- ]tune|distill)\b.{0,60}\b(model|encoder|decoder|transformer)\b"
    ),
]


def silver_label(title: str, abstract: str) -> str | None:
    """First-match keyword heuristic with release-gated Dataset classification."""
    title_lower = title.lower()
    combined = (title + " " + abstract).lower()

    # 1. Theoretical
    for pattern in THEORETICAL_RULES:
        if pattern.search(combined):
            return "Theoretical"

    # 2. Review
    for pattern in REVIEW_RULES:
        if pattern.search(combined):
            return "Review"

    # 3. Dataset (requires release signal or strong title signal, prioritized over Applied)
    is_dataset = False
    for pattern in DATASET_TITLE_RULES:
        if pattern.search(title_lower):
            is_dataset = True
            break
    if not is_dataset:
        for pattern in DATASET_BODY_RULES:
            if pattern.search(combined):
                if DATASET_RELEASE_RE.search(combined):
                    is_dataset = True
                    break

    if is_dataset:
        return "Dataset"

    # 4. Applied
    for pattern in APPLIED_RULES:
        if pattern.search(combined):
            return "Applied"

    # 5. Methodology
    for pattern in METHODOLOGY_RULES:
        if pattern.search(combined):
            return "Methodology"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Quality filters
# ─────────────────────────────────────────────────────────────────────────────

# Section headings that appear as pseudo-titles in SciCite and OpenAIRE dumps.
_BAD_TITLE_EXACT: frozenset[str] = frozenset(
    {
        "discussion",
        "introduction",
        "conclusion",
        "conclusions",
        "results",
        "abstract",
        "methods",
        "method",
        "background",
        "related work",
        "experiments",
        "experiment",
        "evaluation",
        "summary",
        "overview",
        "appendix",
        "acknowledgements",
        "acknowledgments",
        "references",
        "bibliography",
        "supplementary",
        "supplemental material",
        "research paper",
        "paper",
    }
)
_SECTION_NUM_RE = re.compile(r"^\d+[\.:)]")


def _is_clean_title(title: str) -> bool:
    """Return True only for plausible paper titles."""
    t = title.strip()
    if len(t) < 12:
        return False
    if _SECTION_NUM_RE.match(t):  # "4. Effect of…", "3.2 Methods"
        return False
    if t.lower() in _BAD_TITLE_EXACT:  # section-heading exact match
        return False
    # Reject if more than half the characters are digits / punctuation
    alnum = sum(c.isalpha() for c in t)
    if alnum < len(t) * 0.4:
        return False
    return True


# Compiled once — matches arxiv-summarization LaTeX/markup artifacts
_ARTIFACT_RE = re.compile(
    r"@xmath\d+|@xcite|@xref|\[\s*section\s*\]|\*\s*abstract\s*\*"
    r"|@x(math|cite|ref)|\{\{(section|figure|table)\}\}"
    r"|\\textit\{|\\textbf\{|\\cite\{|\\ref\{"
)


def _is_clean_abstract(abstract: str) -> bool:
    """Return True for plausible full-paper abstracts (not section snippets).

    Rejects:
    - Abstracts shorter than 120 chars (fragments)
    - Abstracts starting lower-case (mid-sentence fragments)
    - Abstracts containing arxiv-summarization markup artifacts
      (@xmath0, @xcite, [section], * abstract *, etc.)
    """
    a = abstract.strip()
    # Minimum length heuristic: real abstracts are ≥120 chars
    if len(a) < 120:
        return False
    # Reject fragments that start mid-sentence (lower-case first word is suspicious)
    first_char = a[0] if a else ""
    if first_char.islower():
        return False
    # Reject arxiv-summarization / markup artifacts
    if _ARTIFACT_RE.search(a):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────


def _fingerprint(title: str, abstract: str) -> str:
    norm = re.sub(r"\s+", " ", (title + abstract).lower().strip())
    return hashlib.md5(norm.encode()).hexdigest()


# ── Fasttext LID model (lazy-loaded singleton) ───────────────────────────────
_FT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_FT_MODEL_CACHE = Path.home() / ".cache" / "fasttext" / "lid.176.bin"
_ft_model = None  # module-level singleton, loaded once


def _get_fasttext_model():
    """Return the fasttext LID model, downloading it on first call."""
    global _ft_model
    if _ft_model is not None:
        return _ft_model
    try:
        import fasttext

        fasttext.FastText.eprint = lambda x: None  # suppress noisy stderr
        if not _FT_MODEL_CACHE.exists():
            _FT_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Downloading fasttext LID model → {_FT_MODEL_CACHE} …")
            import urllib.request

            urllib.request.urlretrieve(_FT_MODEL_URL, _FT_MODEL_CACHE)
        _ft_model = fasttext.load_model(str(_FT_MODEL_CACHE))
        return _ft_model
    except Exception:
        return None


def _detect_lang(text: str) -> str:
    """Return ISO 639-1 language code for *text*.

    Priority:
      1. pycld2      — C extension, no model file, microseconds per call.
                       Install:  uv add pycld2   (pre-built wheel available)
      2. fasttext    — C++, requires lid.176.bin (auto-downloaded).
                       Install:  uv add fasttext-wheel
      3. langid      — pure-Python fallback.
      4. langdetect  — last resort (slow Naive Bayes).
      5. 'en'        — best-effort default if nothing is available.

    Only the first 300 chars are inspected for speed.
    """
    snippet = text[:300].replace("\n", " ")

    # ── 1. pycld2 (preferred — pre-built C extension, no model needed) ───
    try:
        import pycld2

        _, _, details = pycld2.detect(snippet)
        return details[0][1]  # ISO 639-1 code
    except Exception:
        pass

    # ── 2. fasttext (if installed and model available) ────────────────────
    model = _get_fasttext_model()
    if model is not None:
        try:
            labels, _ = model.predict(snippet, k=1)
            return labels[0].replace("__label__", "")
        except Exception:
            pass

    # ── 3. langid ─────────────────────────────────────────────────────────
    try:
        import langid

        lang, _ = langid.classify(snippet)
        return lang
    except ImportError:
        pass

    # ── 4. langdetect ─────────────────────────────────────────────────────
    try:
        from langdetect import LangDetectException, detect

        try:
            return detect(snippet)
        except LangDetectException:
            return "??"
    except ImportError:
        pass

    return "en"  # best-effort default


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: SciCite
# ─────────────────────────────────────────────────────────────────────────────


def load_scicite(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Load allenai/scicite (via parquet). Each example is a citation context
    string (50–300 chars), not a full abstract.

    .. deprecated::
        SciCite citation sentences are structurally incompatible with the
        title+abstract format this classifier expects. This function is
        intentionally excluded from the default pipeline and is only callable
        via ``--include_scicite`` for experimental use.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed. Run: uv add datasets")
        sys.exit(1)

    log.info("Loading allenai/scicite (parquet) …")
    splits = ["train", "validation", "test"]
    records: list[dict] = []
    counts: Counter = Counter()

    for split in splits:
        try:
            ds = load_dataset(
                "allenai/scicite",
                revision="refs/convert/parquet",
                split=split,
            )
        except Exception as e:
            log.warning(f"  scicite split={split} failed: {e}")
            continue

        for ex in ds:
            int_label = ex.get("label")
            if int_label is None or int_label not in SCICITE_LABEL_MAP:
                continue
            section = ex.get("sectionName") or ""
            label = scicite_label(int_label, section)

            # The "string" field is the citing sentence; use it as abstract
            text = (ex.get("string") or "").strip()
            if not text or len(text) < 40:
                continue

            # No real title in scicite — use section name as pseudo-title
            title = section.strip() or "Research Paper"

            fp = _fingerprint(title, text)
            if fp in seen or counts[label] >= max_per_label:
                continue

            seen.add(fp)
            counts[label] += 1
            records.append(
                {
                    "title": title,
                    "abstract": text,
                    "label": label,
                    "source": "scicite",
                }
            )

    log.info(f"SciCite: {len(records)} records — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: arXiv abstracts
# ─────────────────────────────────────────────────────────────────────────────


def load_arxiv(
    max_per_label: int,
    seen: set[str],
    stream_limit: int = 200_000,
) -> list[dict]:
    """
    Stream gfissore/arxiv-abstracts-2021. For each paper:
    1. Try keyword silver-labeling (primary signal).
    2. Fall back to arXiv category prior if keyword gives no match.
    Stop streaming when every class has reached max_per_label or
    stream_limit rows are consumed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    log.info(f"Streaming gfissore/arxiv-abstracts-2021 (up to {stream_limit:,} rows) …")
    records: list[dict] = []
    counts: Counter = Counter()
    scanned = 0

    try:
        ds = load_dataset(
            "gfissore/arxiv-abstracts-2021",
            split="train",
            streaming=True,
        )
    except Exception as e:
        log.warning(f"Could not load arxiv dataset: {e}")
        return []

    skipped = 0
    from tqdm import tqdm

    pbar = tqdm(
        total=stream_limit, desc="arXiv-stream", unit="ex", ncols=80, mininterval=1.0
    )
    for ex in ds:
        if skipped < 1100000:
            skipped += 1
            continue

        scanned += 1
        pbar.update(1)
        if scanned > stream_limit:
            break
        if all(counts[label] >= max_per_label for label in LABELS):
            break

        title = (ex.get("title") or "").strip().replace("\n", " ")
        abstract = (ex.get("abstract") or "").strip().replace("\n", " ")
        cats_raw = ex.get("categories") or ""
        # Field is a list in some versions, a space-delimited string in others
        categories = cats_raw if isinstance(cats_raw, list) else cats_raw

        # Fast CS/ML domain filter: skip non-CS/AI/Applied-science papers to boost yield
        cats = categories if isinstance(categories, list) else categories.split()
        if not any(
            c.startswith("cs.")
            or c.startswith("stat.ml")
            or c.startswith("eess.")
            or c.startswith("q-fin")
            for c in cats
        ):
            continue

        if not _is_clean_title(title) or not _is_clean_abstract(abstract):
            continue

        # Label resolution
        label = silver_label(title, abstract)
        if label is None:
            label = arxiv_prior(categories)
        if label is None:
            continue

        if counts[label] >= max_per_label:
            continue

        fp = _fingerprint(title, abstract)
        if fp in seen:
            continue

        seen.add(fp)
        counts[label] += 1
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "label": label,
                "source": "arxiv",
            }
        )
        pbar.set_postfix({label[:3]: counts[label] for label in LABELS}, refresh=False)

    pbar.close()
    log.info(f"arXiv: {len(records)} records from {scanned:,} scanned — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: ccdv/arxiv-summarization (near-gold, cleaner abstracts)
# ─────────────────────────────────────────────────────────────────────────────


def load_arxiv_summarization(
    max_per_label: int,
    seen: set[str],
) -> list[dict]:
    """
    Load ccdv/arxiv-summarization (arxiv split).  Abstracts are typically
    longer and cleaner than the streaming gfissore dataset.  Label via the
    same keyword silver-labeler + category prior.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    log.info("Loading ccdv/arxiv-summarization …")
    records: list[dict] = []
    counts: Counter = Counter()

    try:
        ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    except Exception as e:
        log.warning(f"ccdv/arxiv-summarization unavailable: {e}")
        return []

    for ex in ds:
        if all(counts[label] >= max_per_label for label in LABELS):
            break

        # Prefer the dedicated "title" field; fall back to article first line
        # (unreliable — the article field is the full text, not just the title).
        if ex.get("title"):
            title = ex["title"].strip().replace("\n", " ")
        else:
            title = (ex.get("article") or "").split("\n")[0].strip()
            if title:
                log.debug(
                    "arxiv-sum: using first-line heuristic for title: %r", title[:60]
                )
        abstract = (ex.get("abstract") or "").strip().replace("\n", " ")

        if not _is_clean_title(title) or not _is_clean_abstract(abstract):
            continue

        label = silver_label(title, abstract)
        if label is None:
            continue
        if counts[label] >= max_per_label:
            continue

        fp = _fingerprint(title, abstract)
        if fp in seen:
            continue

        seen.add(fp)
        counts[label] += 1
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "label": label,
                "source": "arxiv_summarization",
            }
        )

    log.info(f"ccdv/arxiv-summarization: {len(records)} records — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: PubMed E-utilities API  (GOLD labels: Review, Applied)
# ─────────────────────────────────────────────────────────────────────────────

_PUBMED_QUERIES: dict[str, str] = {
    # NLM curated publication-type tags — these are gold labels.
    # Review: restrict to systematic review + meta-analysis ONLY.
    # Plain [pt]Review is too broad (includes editorials, narrative reviews
    # that share language with Applied / Methodology papers).
    "Review": (
        '("systematic review"[pt] OR "meta-analysis"[pt]) '
        "AND hasabstract[text] "
        'AND ("systematic review"[tiab] OR "meta-analysis"[tiab] OR '
        '"literature review"[tiab] OR "scoping review"[tiab] OR '
        '"narrative review"[tiab])'
    ),
    "Applied": (
        '("clinical trial"[pt] OR "randomized controlled trial"[pt]) '
        "AND hasabstract[text]"
    ),
}
_PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Words in a Review abstract that strongly suggest it is actually an Applied
# paper (clinical-trial language).  Reject Review abstracts containing these.
_REVIEW_REJECT_RE = re.compile(
    r"\b(randomized|randomised|placebo|double[- ]blind|control(?:led)?\s+trial"
    r"|patients\s+were\s+(randomly|assigned|allocated)"
    r"|compared\s+(the\s+)?efficacy|versus\s+placebo)\b",
    re.IGNORECASE,
)


def load_pubmed(max_per_label: int, seen: set[str]) -> list[dict]:
    """Fetch gold-labelled Review and Applied abstracts from PubMed."""
    import time

    try:
        import requests
    except ImportError:
        log.warning("requests not installed — skipping PubMed source.")
        return []

    from tqdm import tqdm

    records: list[dict] = []
    counts: Counter = Counter()

    for label, query in _PUBMED_QUERIES.items():
        if counts[label] >= max_per_label:
            continue

        log.info(f"  PubMed [{label}]: searching …")
        try:
            # Step 1 — get PMIDs
            r = requests.get(
                _PUBMED_ESEARCH,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": min(max_per_label * 4, 5000),
                    "retmode": "json",
                    "usehistory": "y",
                },
                timeout=20,
            )
            r.raise_for_status()
            result = r.json()["esearchresult"]
            pmids = result.get("idlist", [])

            if not pmids:
                log.warning(f"  PubMed [{label}]: no results")
                continue

            # Step 2 — fetch abstracts in batches of 200
            batch_size = 200
            pbar = tqdm(
                total=max_per_label, desc=f"PubMed-{label}", unit="rec", ncols=80
            )
            for start in range(0, min(len(pmids), max_per_label * 3), batch_size):
                if counts[label] >= max_per_label:
                    break
                batch = pmids[start : start + batch_size]
                try:
                    fr = requests.get(
                        _PUBMED_EFETCH,
                        params={
                            "db": "pubmed",
                            "id": ",".join(batch),
                            "rettype": "abstract",
                            "retmode": "xml",
                        },
                        timeout=30,
                    )
                    if fr.status_code != 200 or not fr.content.strip():
                        log.warning(
                            f"PubMed batch HTTP {fr.status_code} or empty payload. "
                            f"Skipping batch (start={start})."
                        )
                        continue
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(fr.content)  # .content avoids encoding bugs
                    for article in root.findall(".//PubmedArticle"):
                        if counts[label] >= max_per_label:
                            break
                        title_el = article.find(".//ArticleTitle")
                        abs_el = article.find(".//AbstractText")
                        title = (
                            (title_el.text or "").strip()
                            if title_el is not None
                            else ""
                        )
                        abstract = (
                            (abs_el.text or "").strip() if abs_el is not None else ""
                        )
                        if not _is_clean_title(title) or not _is_clean_abstract(
                            abstract
                        ):
                            continue
                        # For Review: reject abstracts with clinical-trial language
                        if label == "Review" and _REVIEW_REJECT_RE.search(abstract):
                            continue
                        fp = _fingerprint(title, abstract)
                        if fp in seen:
                            continue
                        seen.add(fp)
                        counts[label] += 1
                        pbar.update(1)
                        records.append(
                            {
                                "title": title,
                                "abstract": abstract,
                                "label": label,
                                "source": "pubmed",
                            }
                        )
                    time.sleep(0.4)  # NCBI rate limit: 3 req/s without API key
                except Exception as e:
                    log.debug(f"  PubMed batch error: {e}")
                    continue
            pbar.close()
        except Exception as e:
            log.warning(f"  PubMed [{label}] failed: {e}")
            continue

    log.info(f"PubMed: {len(records)} records — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: Semantic Scholar API  (GOLD labels: Dataset)
# ─────────────────────────────────────────────────────────────────────────────

_S2_DATASET_QUERIES = [
    "we introduce a new dataset",
    "we present a new benchmark",
    "we release a corpus",
    "large-scale annotated dataset",
    "we collect and annotate",
    "publicly available dataset benchmark",
    "inter-annotator agreement corpus",
    "crowdsourced annotation dataset",
]
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


S2_BLOCKED = False


def _s2_request_with_retry(query: str, offset: int) -> dict | None:
    """Helper to perform Semantic Scholar requests with exponential backoff on 429."""
    global S2_BLOCKED
    if S2_BLOCKED:
        return None

    import time

    try:
        import requests
    except ImportError:
        return None

    retries = 3
    backoff = 2.0
    for attempt in range(retries):
        try:
            r = requests.get(
                _S2_API,
                params={
                    "query": query,
                    "fields": "title,abstract,externalIds",
                    "limit": 100,
                    "offset": offset,
                },
                timeout=20,
            )
            if r.status_code == 429:
                if attempt == retries - 1:
                    log.warning(
                        "  S2 API persistently rate-limiting (HTTP 429). "
                        "Marking S2 as blocked and skipping subsequent S2 requests."
                    )
                    S2_BLOCKED = True
                    return None
                log.warning(
                    f"  S2 API 429 (rate limit) - retrying in {backoff}s (attempt {attempt+1}/{retries})..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (400, 403):
                log.warning(
                    f"  S2 API HTTP {r.status_code} for query '{query}' offset={offset}"
                )
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                log.debug(f"  S2 request failed after {retries} attempts: {e}")
                return None
            time.sleep(backoff)
            backoff *= 2
    return None


def load_semantic_scholar_datasets(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Query Semantic Scholar search API with dataset-announcement phrases.
    No API key needed (1 req/s public rate limit).
    Each result is cross-checked with the silver labeler to ensure label quality.
    """
    import time

    records: list[dict] = []
    fetched = 0
    s2_ids_seen: set[str] = set()

    from tqdm import tqdm

    log.info("Loading Semantic Scholar /paper/search (Dataset queries) …")
    pbar = tqdm(total=max_per_label, desc="S2-Dataset", unit="rec", ncols=80)
    for query in _S2_DATASET_QUERIES:
        if fetched >= max_per_label:
            break
        offset = 0
        while fetched < max_per_label:
            try:
                data = _s2_request_with_retry(query, offset)
                if not data:
                    break
                items = data.get("data", [])
                if not items:
                    break

                for paper in items:
                    if fetched >= max_per_label:
                        break
                    pid = paper.get("paperId", "")
                    title = (paper.get("title") or "").strip()
                    abstract = (paper.get("abstract") or "").strip()
                    if pid in s2_ids_seen:
                        continue
                    s2_ids_seen.add(pid)
                    if not _is_clean_title(title) or not _is_clean_abstract(abstract):
                        continue
                    # Verify using silver_label to reject dominant Methodology/Theoretical/Review/Applied papers.
                    kw = silver_label(title, abstract)
                    if kw in ("Methodology", "Theoretical", "Review", "Applied"):
                        continue
                    _combined = f"{title} {abstract}".lower()
                    if not any(
                        kw in _combined
                        for kw in (
                            "dataset",
                            "benchmark",
                            "corpus",
                            "annotation",
                            "evaluation",
                            "collection",
                        )
                    ):
                        continue
                    fp = _fingerprint(title, abstract)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    fetched += 1
                    pbar.update(1)
                    records.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "label": "Dataset",
                            "source": "semantic_scholar",
                        }
                    )

                offset += len(items)
                if len(items) < 100:
                    break
                time.sleep(1.1)  # stay under 1 req/s
            except Exception as e:
                log.debug(f"  S2 query '{query}' offset={offset}: {e}")
                break

    pbar.close()
    log.info(f"Semantic Scholar: {len(records)} Dataset records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: Semantic Scholar API  (GOLD labels: Methodology)
# ─────────────────────────────────────────────────────────────────────────────

_S2_METHODOLOGY_QUERIES = [
    "we propose a novel method",
    "we introduce a new model architecture",
    "we present a new algorithm",
    "we design a new framework",
    "we develop a new approach",
    "end-to-end deep learning method",
    "state-of-the-art neural network architecture",
    "we outperform previous methods",
]


def load_semantic_scholar_methodology(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Query Semantic Scholar with methodology-announcement phrases and accept only
    papers where the silver labeler agrees (silver_label == "Methodology").
    This provides high-precision gold labels for the Methodology class without
    relying on noisy arXiv silver labels alone.
    """
    import time

    records: list[dict] = []
    fetched = 0
    s2_ids_seen: set[str] = set()

    from tqdm import tqdm

    log.info("Loading Semantic Scholar /paper/search (Methodology queries) …")
    pbar = tqdm(total=max_per_label, desc="S2-Methodology", unit="rec", ncols=80)
    for query in _S2_METHODOLOGY_QUERIES:
        if fetched >= max_per_label:
            break
        offset = 0
        while fetched < max_per_label:
            try:
                data = _s2_request_with_retry(query, offset)
                if not data:
                    break
                items = data.get("data", [])
                if not items:
                    break

                for paper in items:
                    if fetched >= max_per_label:
                        break
                    pid = paper.get("paperId", "")
                    title = (paper.get("title") or "").strip()
                    abstract = (paper.get("abstract") or "").strip()
                    if pid in s2_ids_seen:
                        continue
                    s2_ids_seen.add(pid)
                    if not _is_clean_title(title) or not _is_clean_abstract(abstract):
                        continue
                    # Lightweight cross-check: S2 query is already Methodology-
                    # specific; require one methodological keyword to be present.
                    _combined = f"{title} {abstract}".lower()
                    if not any(
                        kw in _combined
                        for kw in (
                            "we propose",
                            "we introduce",
                            "we present",
                            "we design",
                            "we develop",
                            "we train",
                            "method",
                            "model",
                            "algorithm",
                            "architecture",
                            "framework",
                            "approach",
                            "network",
                        )
                    ):
                        continue
                    fp = _fingerprint(title, abstract)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    fetched += 1
                    pbar.update(1)
                    records.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "label": "Methodology",
                            "source": "semantic_scholar",
                        }
                    )

                offset += len(items)
                if len(items) < 100:
                    break
                time.sleep(1.1)  # stay under 1 req/s
            except Exception as e:
                log.debug(f"  S2 Methodology query '{query}' offset={offset}: {e}")
                break

    pbar.close()
    log.info(f"Semantic Scholar: {len(records)} Methodology records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: Semantic Scholar API  (GOLD labels: Theoretical)
# ─────────────────────────────────────────────────────────────────────────────

_S2_THEORETICAL_QUERIES = [
    "we prove that",
    "convergence analysis theoretical",
    "we derive a lower bound",
    "theoretical analysis generalization",
    "PAC learning sample complexity",
    "regret bound online learning",
    "we establish theoretical guarantees",
    "formal proof mathematical analysis",
]


def load_semantic_scholar_theoretical(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Query Semantic Scholar with formal/theoretical-analysis phrases and accept
    only papers where the silver labeler agrees (silver_label == "Theoretical").
    Provides clean gold signal for the Theoretical class, which previously had
    no dedicated gold source and was hurt by markup-artifact removal.
    """
    import time

    records: list[dict] = []
    fetched = 0
    s2_ids_seen: set[str] = set()

    from tqdm import tqdm

    log.info("Loading Semantic Scholar /paper/search (Theoretical queries) …")
    pbar = tqdm(total=max_per_label, desc="S2-Theoretical", unit="rec", ncols=80)
    for query in _S2_THEORETICAL_QUERIES:
        if fetched >= max_per_label:
            break
        offset = 0
        while fetched < max_per_label:
            try:
                data = _s2_request_with_retry(query, offset)
                if not data:
                    break
                items = data.get("data", [])
                if not items:
                    break

                for paper in items:
                    if fetched >= max_per_label:
                        break
                    pid = paper.get("paperId", "")
                    title = (paper.get("title") or "").strip()
                    abstract = (paper.get("abstract") or "").strip()
                    if pid in s2_ids_seen:
                        continue
                    s2_ids_seen.add(pid)
                    if not _is_clean_title(title) or not _is_clean_abstract(abstract):
                        continue
                    # Lightweight cross-check: require formal/mathematical signal.
                    _combined = f"{title} {abstract}".lower()
                    if not any(
                        kw in _combined
                        for kw in (
                            "we prove",
                            "proof",
                            "theorem",
                            "lemma",
                            "convergence",
                            "bound",
                            "analysis",
                            "complexity",
                            "theoretical",
                            "formal",
                            "sample complexity",
                            "regret",
                        )
                    ):
                        continue
                    fp = _fingerprint(title, abstract)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    fetched += 1
                    pbar.update(1)
                    records.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "label": "Theoretical",
                            "source": "semantic_scholar",
                        }
                    )

                offset += len(items)
                if len(items) < 100:
                    break
                time.sleep(1.1)  # stay under 1 req/s
            except Exception as e:
                log.debug(f"  S2 Theoretical query '{query}' offset={offset}: {e}")
                break

    pbar.close()
    log.info(f"Semantic Scholar: {len(records)} Theoretical records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: Dataset papers  (scidocs + Papers With Code fallback)
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset_papers(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Gold Dataset labels from two sources:

    A) allenai/scidocs (HF datasets) — large S2ORC-derived corpus with titles
       and abstracts.  We stream it and accept only examples where the silver
       labeler fires Dataset.

    B) Papers With Code REST API keyword search — /papers/?q= with
       dataset-specific queries, silver-labeler cross-check.
    """
    import time

    records: list[dict] = []
    fetched = 0

    # ── A: allenai/scidocs ────────────────────────────────────────────────
    # Skipped: allenai/scidocs has been deprecated on HF Hub.
    pass

    # ── B: Papers With Code keyword search ───────────────────────────────
    try:
        import requests

        BASE = "https://paperswithcode.com/api/v1/papers/"
        seen_pwc: set[str] = set()
        queries = [
            "dataset benchmark NLP",
            "annotated corpus evaluation",
            "large scale dataset collection",
            "we introduce a dataset",
            "we release a benchmark",
        ]
        log.info(
            f"  Dataset source B: Papers With Code keyword search "
            f"(have {fetched}, need {max_per_label}) …"
        )
        for kw in queries:
            if fetched >= max_per_label:
                break
            for page in (1, 2, 3):
                if fetched >= max_per_label:
                    break
                try:
                    r = requests.get(
                        BASE,
                        params={"q": kw, "page": page, "page_size": 50},
                        timeout=20,
                    )
                    r.raise_for_status()
                    results = r.json().get("results", [])
                    if not results:
                        break
                    for paper in results:
                        if fetched >= max_per_label:
                            break
                        pid = paper.get("id", "")
                        title = (paper.get("title") or "").strip()
                        abstract = (paper.get("abstract") or "").strip()
                        if (
                            pid in seen_pwc
                            or not _is_clean_title(title)
                            or not _is_clean_abstract(abstract)
                        ):
                            continue

                        # Relaxed check: exclude only if explicitly labeled as another intent.
                        # Since these come from Papers With Code (which only indexes papers introducing
                        # datasets/benchmarks), they are highly likely to be Dataset papers.
                        kw_label = silver_label(title, abstract)
                        if kw_label in (
                            "Methodology",
                            "Theoretical",
                            "Review",
                            "Applied",
                        ):
                            continue

                        fp = _fingerprint(title, abstract)
                        if fp in seen:
                            continue
                        seen_pwc.add(pid)
                        seen.add(fp)
                        fetched += 1
                        records.append(
                            {
                                "title": title,
                                "abstract": abstract,
                                "label": "Dataset",
                                "source": "paperswithcode",
                            }
                        )
                    time.sleep(0.3)
                except Exception as e:
                    log.debug(f"  PWC query '{kw}' page={page} failed: {e}")
                    break
    except ImportError:
        log.warning("  requests not installed — skipping Papers With Code source.")

    log.info(f"Dataset papers: {len(records)} records (scidocs + paperswithcode)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: OpenAIRE Graph API (optional, live)
# ─────────────────────────────────────────────────────────────────────────────

_OPENAIRE_QUERIES: dict[str, list[str]] = {
    "Methodology": [
        "we propose a new method deep learning",
        "novel architecture neural network",
        "we introduce a model language",
    ],
    "Dataset": [
        "we release a new dataset benchmark NLP",
        "annotated corpus multilingual evaluation",
        "new evaluation benchmark shared task",
    ],
    "Review": [
        "survey of deep learning NLP methods",
        "systematic review machine learning",
        "overview of transformer language models",
    ],
    "Applied": [
        "applying NLP to clinical text",
        "machine learning legal domain application",
        "deep learning biomedical information extraction",
    ],
    "Theoretical": [
        "theoretical analysis convergence neural networks",
        "generalization bounds PAC learning",
        "information theoretic machine learning proof",
    ],
}


def load_openaire(max_per_label: int, seen: set[str]) -> list[dict]:
    """Sample OpenAIRE Graph API with targeted queries per label."""
    import time

    try:
        import requests
    except ImportError:
        log.warning("requests not available — skipping OpenAIRE source.")
        return []

    BASE_URL = "https://api.openaire.eu/search/publications"
    counts: Counter = Counter()
    records: list[dict] = []

    for label, queries in _OPENAIRE_QUERIES.items():
        for query in queries:
            if counts[label] >= max_per_label:
                break
            try:
                resp = requests.get(
                    BASE_URL,
                    params={"keywords": query, "format": "json", "size": 20},
                    timeout=15,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                results = (
                    data.get("response", {}).get("results", {}).get("result", []) or []
                )
                for r in results:
                    try:
                        meta = r["metadata"]["oaf:entity"]["oaf:result"]
                        title_raw = meta.get("title", [{}])
                        title = (
                            title_raw[0].get("$", "")
                            if isinstance(title_raw, list)
                            else title_raw.get("$", "")
                        ).strip()
                        desc_raw = meta.get("description", [])
                        abstract = (
                            desc_raw[0].get("$", "") if desc_raw else ""
                        ).strip()

                        if not _is_clean_title(title) or not _is_clean_abstract(
                            abstract
                        ):
                            continue

                        # Verify with silver labeler — accept if consistent
                        inferred = silver_label(title, abstract)
                        if inferred is not None and inferred != label:
                            continue  # discard inconsistent labels

                        fp = _fingerprint(title, abstract)
                        if fp in seen or counts[label] >= max_per_label:
                            continue

                        seen.add(fp)
                        counts[label] += 1
                        records.append(
                            {
                                "title": title,
                                "abstract": abstract,
                                "label": label,
                                "source": "openaire",
                            }
                        )
                    except (KeyError, IndexError, TypeError):
                        continue

                time.sleep(0.3)
            except Exception as e:
                log.debug(f"OpenAIRE query '{query}' failed: {e}")
                continue

    log.info(f"OpenAIRE: {len(records)} records — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source: LLM Judge Corrections (High Confidence Gold Labels)
# ─────────────────────────────────────────────────────────────────────────────


def load_llm_judge_corrections(max_per_label: int, seen: set[str]) -> list[dict]:
    """
    Load high-confidence LLM judge corrections (where proposed_label != initial_intent)
    from the collaborative SQLite database.
    Only takes annotations with comment starting with "[HIGH confidence]"
    to act as gold training examples.
    """
    import sqlite3
    from pathlib import Path

    possible_paths = [
        Path("demo_apps/echo_dsrn_graph/data/collaborative.db"),
        Path(__file__).resolve().parent.parent.parent
        / "demo_apps/echo_dsrn_graph/data/collaborative.db",
        Path.home() / ".ethicalabs/datasets/collaborative.db",
    ]

    db_path = None
    for p in possible_paths:
        if p.exists():
            db_path = p
            break

    if db_path is None:
        log.warning(
            "Collaborative database not found. Skipping LLM corrections source."
        )
        return []

    log.info(f"Loading high-confidence LLM judge corrections from {db_path} …")
    records: list[dict] = []
    counts: Counter = Counter()

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.title, p.abstract, a.proposed_label
            FROM annotations a
            JOIN paper_records p ON a.paper_doi = p.doi
            WHERE a.annotator_type = 'llm'
              AND a.proposed_label IS NOT NULL
              AND a.comment LIKE '[HIGH confidence]%'
              AND (a.is_flagged = 0 OR a.is_flagged IS NULL)
        """
        )
        rows = cursor.fetchall()
        conn.close()

        for title, abstract, proposed_label in rows:
            if proposed_label not in LABELS:
                continue
            if counts[proposed_label] >= max_per_label:
                continue
            title = (title or "").strip()
            abstract = (abstract or "").strip()
            if not title or not abstract:
                continue
            fp = _fingerprint(title, abstract)
            if fp in seen:
                continue
            seen.add(fp)
            counts[proposed_label] += 1
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "label": proposed_label,
                    "source": "llm_judge_corrections",
                }
            )
    except Exception as e:
        log.warning(f"Failed to load LLM judge corrections: {e}")
        return []

    log.info(f"LLM Judge Corrections: {len(records)} records — {dict(counts)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Chat format helper
# ─────────────────────────────────────────────────────────────────────────────


def to_messages(title: str, abstract: str, label: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(title=title, abstract=abstract),
        },
        {"role": "assistant", "content": label},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Balancing
# ─────────────────────────────────────────────────────────────────────────────


def balance(records: list[dict], target: int, seed: int) -> list[dict]:
    """Downsample over-represented classes to `target` examples each."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    balanced = []
    log.info("Post-balance per-label counts:")
    for label in LABELS:
        bucket = by_label.get(label, [])
        rng.shuffle(bucket)
        if len(bucket) > target:
            bucket = bucket[:target]
        balanced.extend(bucket)
        bar = "█" * min(len(bucket) // 20, 40)
        log.info(f"  {label:12s}: {len(bucket):5d}  {bar}")

    rng.shuffle(balanced)
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 5-label research-paper intent dataset for Echo-DSRN SFT."
    )
    parser.add_argument(
        "--output_dir",
        default="~/.ethicalabs/datasets/research-intent",
        help="Output directory for train.jsonl + stats.json",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1200,
        help="Max examples per label after balancing (default: 1200 → 6 000 total)",
    )
    parser.add_argument(
        "--arxiv_stream_limit",
        type=int,
        default=500_000,
        help="Max arXiv rows to stream before stopping (default: 500 000)",
    )
    parser.add_argument(
        "--openaire",
        action="store_true",
        help="Also query the OpenAIRE Graph API for additional silver-labeled samples",
    )
    parser.add_argument(
        "--skip_arxiv",
        action="store_true",
        help="Skip arXiv source (use only scicite + optional OpenAIRE)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and balancing",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Print dataset statistics without writing any files (dry-run mode)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    if not args.analyze_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    all_records: list[dict] = []

    # ── Source 0 (GOLD): LLM Judge Corrections (High Confidence) ──────────
    llm_corr_records = load_llm_judge_corrections(args.samples_per_class, seen)
    all_records.extend(llm_corr_records)

    # ── Source 1 (GOLD): PubMed — Review + Applied ────────────────────────
    pubmed_records = load_pubmed(args.samples_per_class, seen)
    all_records.extend(pubmed_records)

    # ── Source 2 (GOLD): Semantic Scholar — Dataset ────────────────────────
    s2_records = load_semantic_scholar_datasets(args.samples_per_class, seen)
    all_records.extend(s2_records)

    # ── Source 2b (GOLD): Semantic Scholar — Methodology ─────────────────
    s2_meth_records = load_semantic_scholar_methodology(args.samples_per_class, seen)
    all_records.extend(s2_meth_records)

    # ── Source 2c (GOLD): Semantic Scholar — Theoretical ─────────────────
    s2_theo_records = load_semantic_scholar_theoretical(args.samples_per_class, seen)
    all_records.extend(s2_theo_records)

    # ── Source 3 (GOLD): Dataset papers — scidocs + Papers With Code ──────
    dataset_records = load_dataset_papers(args.samples_per_class, seen)
    all_records.extend(dataset_records)

    # ── Source 3 (near-gold): ccdv/arxiv-summarization ────────────────────
    arxiv_sum_records = load_arxiv_summarization(args.samples_per_class, seen)
    all_records.extend(arxiv_sum_records)

    # ── Source 4 (silver): gfissore/arxiv-abstracts-2021 ──────────────────
    if not args.skip_arxiv:
        arxiv_records = load_arxiv(
            args.samples_per_class,
            seen,
            stream_limit=args.arxiv_stream_limit,
        )
        all_records.extend(arxiv_records)

    # ── Source 5 (silver): OpenAIRE (optional top-up) ─────────────────────
    if args.openaire:
        oai_records = load_openaire(args.samples_per_class // 4, seen)
        all_records.extend(oai_records)

    # ── Language detection (parallel, fasttext-accelerated) ───────────────
    # fasttext releases the GIL so ThreadPoolExecutor gives real parallelism.
    # Falls back to langid / langdetect / 'en' if fasttext is not installed.
    import os
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    n_workers = min(os.cpu_count() or 4, 8)
    needs_lang = [r for r in all_records if "lang" not in r]
    if needs_lang:
        log.info(
            f"Detecting languages for {len(needs_lang):,} records "
            f"({n_workers} workers) …"
        )
        texts = [f"{r['title']} {r['abstract']}" for r in needs_lang]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            langs = list(
                tqdm(
                    pool.map(_detect_lang, texts),
                    total=len(texts),
                    desc="lang-detect",
                    unit="rec",
                    ncols=80,
                )
            )
        for r, lang in zip(needs_lang, langs):
            r["lang"] = lang
        log.info("Language detection complete.")

    # ── Analysis ──────────────────────────────────────────────────────────
    raw_counts: Counter = Counter(r["label"] for r in all_records)
    source_counts: dict[str, Counter] = defaultdict(Counter)
    for r in all_records:
        source_counts[r["source"]][r["label"]] += 1

    log.info(f"\n{'─'*62}")
    log.info(f"RAW DATASET: {len(all_records):,} records total")
    log.info(f"{'─'*62}")
    log.info("Per-label counts (raw, pre-balance):")
    for label in LABELS:
        n = raw_counts.get(label, 0)
        bar = "█" * min(n // 20, 40)
        log.info(f"  {label:12s}: {n:5d}  {bar}")

    log.info("\nPer-source breakdown:")
    for src, cnts in sorted(source_counts.items()):
        log.info(f"  {src:12s}: {dict(cnts)}")

    missing = [lbl for lbl in LABELS if raw_counts.get(lbl, 0) == 0]
    if missing:
        log.warning(f"\n⚠️  Labels with ZERO examples: {missing}")
        log.warning("   Consider adding --openaire or removing --skip_arxiv.")

    if args.analyze_only:
        log.info("\n[analyze_only] No files written.")
        return

    # ── Balance ───────────────────────────────────────────────────────────
    log.info(f"\nBalancing to {args.samples_per_class} per class …")
    balanced = balance(all_records, args.samples_per_class, args.seed)

    Counter(r["label"] for r in balanced)
    log.info(f"Final dataset: {len(balanced):,} records")

    # ── Stratified train / val split ──────────────────────────────────────
    # Hold out 10% per class as a fixed validation set (never seen by trainer).
    # The remaining 90% goes to train.jsonl; the trainer does its own
    # internal 5% split for eval monitoring during training.
    rng_split = random.Random(args.seed + 1)
    by_label_bal: dict[str, list[dict]] = defaultdict(list)
    for r in balanced:
        by_label_bal[r["label"]].append(r)

    train_records: list[dict] = []
    val_records: list[dict] = []
    for label in LABELS:
        bucket = by_label_bal.get(label, [])
        rng_split.shuffle(bucket)
        n_val = max(1, len(bucket) // 10)
        val_records.extend(bucket[:n_val])
        train_records.extend(bucket[n_val:])

    rng_split.shuffle(train_records)
    rng_split.shuffle(val_records)

    log.info(f"  train.jsonl : {len(train_records):,} records")
    log.info(f"  val.jsonl   : {len(val_records):,} records")

    # ── Write train.jsonl ─────────────────────────────────────────────────
    train_path = out_dir / "train.jsonl"
    log.info(f"Writing {train_path} …")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            row = {
                "messages": to_messages(r["title"], r["abstract"], r["label"]),
                "label": r["label"],
                "source": r["source"],
                "lang": r.get("lang", "en"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Write val.jsonl ───────────────────────────────────────────────────
    val_path = out_dir / "val.jsonl"
    log.info(f"Writing {val_path} …")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            row = {
                "messages": to_messages(r["title"], r["abstract"], r["label"]),
                "label": r["label"],
                "source": r["source"],
                "lang": r.get("lang", "en"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Write stats.json ──────────────────────────────────────────────────
    val_counts: Counter = Counter(r["label"] for r in val_records)
    stats = {
        "total": len(balanced),
        "train": len(train_records),
        "val": len(val_records),
        "samples_per_class_target": args.samples_per_class,
        "seed": args.seed,
        "labels": LABELS,
        "per_label_train": {
            lbl: Counter(r["label"] for r in train_records).get(lbl, 0)
            for lbl in LABELS
        },
        "per_label_val": {lbl: val_counts.get(lbl, 0) for lbl in LABELS},
        "per_source": {src: dict(cnts) for src, cnts in source_counts.items()},
        "raw_total": len(all_records),
        "raw_per_label": dict(raw_counts),
        "sources_used": list(source_counts.keys()),
        "openaire_enabled": args.openaire,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # ── Sentinel ──────────────────────────────────────────────────────────
    (out_dir / "DONE").write_text(
        f"train={len(train_records)}  val={len(val_records)}\n"
        f"train_path={train_path}\nval_path={val_path}\n"
    )

    log.info(
        f"\n✅ Done — {len(train_records):,} train + {len(val_records):,} val records"
    )
    log.info(f"   Stats → {stats_path}")
    log.info("\nNext step — launch classification fine-tune:")
    log.info("  uv run python training/train_clf.py \\")
    log.info("    --model_name_or_path ethicalabs/Echo-DSRN-v0.1.3-Intent-CLF \\")
    log.info(f"    --dataset_name {train_path} \\")
    log.info(
        "    --id2label '0:Methodology,1:Dataset,2:Review,3:Applied,4:Theoretical' \\"
    )
    log.info("    --backbone_lr 2e-5 \\")
    log.info("    --output_dir outputs/echo-research-intent-clf \\")
    log.info("    --num_train_epochs 5")


if __name__ == "__main__":
    main()
