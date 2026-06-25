"""
datasets_scripts/research_intent/expand_multilingual_openaire.py
─────────────────────────────────────────────────────────────────────────────
Expand the base research-intent dataset with multilingual paper abstracts
fetched from the OpenAIRE Graph API.

Why a separate script?
    The main prepare_research_intent.py uses keyword silver-labeling that is
    English-only. OpenAIRE indexes pan-European publications in 20+ languages,
    so English keyword matching silently skips French, German, Italian, Spanish,
    Portuguese, Polish, Dutch, Greek, etc. abstracts.

Strategy (language-agnostic)
    Instead of matching keywords inside the abstract, we assign the label from
    the *query* itself — the query is chosen to semantically target a specific
    research intent category and the API returns papers matching that intent.
    We keep all papers regardless of abstract language, then apply a light
    post-filter: discard papers where a *cross-lingual* marker strongly
    contradicts the query intent.

Cross-lingual contradiction markers (title-level only, very conservative)
    We only reject a paper if its title contains a clear cross-lingual signal
    that contradicts the label — e.g. a query for "Methodology" papers should
    not return papers whose title in any language is clearly a review/survey.
    The title-only filter keeps recall high while removing obvious errors.

Cross-lingual survey/review title signals (conservative discard list)
    These words in a title, regardless of language, strongly suggest the paper
    is a survey even if it was returned by a Methodology query:
    survey, review, overview, étude, überblick, revue, rassegna, revisión,
    przegląd, overzicht, ανασκόπηση
    (we discard from non-Review intents if these appear in title)

Output
    Appends to an existing train.jsonl (--base_file) or creates a new
    multilingual-only JSONL (--output).
    Also writes a stats breakdown per language.

Usage
    uv run python datasets_scripts/research_intent/expand_multilingual_openaire.py \\
        --base_file ~/.ethicalabs/datasets/research-intent/train.jsonl \\
        --output    ~/.ethicalabs/datasets/research-intent/train_multilingual.jsonl \\
        --per_label 300 \\
        --languages en,fr,de,it,es,pt,nl,pl

    # Merge both files for the final SFT dataset:
    cat train.jsonl train_multilingual.jsonl > train_full.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
# Multilingual query bank
# Each entry: (query_string, language_hint, target_label)
# Queries are chosen so that the API's semantic search returns papers whose
# primary intent matches the label — no abstract-level keyword matching needed.
# ─────────────────────────────────────────────────────────────────────────────

MULTILINGUAL_QUERIES: list[tuple[str, str, str]] = [
    # ── Methodology ──────────────────────────────────────────────────────
    ("we propose a new deep learning method", "en", "Methodology"),
    ("nous proposons une nouvelle méthode d'apprentissage", "fr", "Methodology"),
    ("wir schlagen eine neue Methode vor neuronales Netz", "de", "Methodology"),
    ("proponiamo un nuovo metodo di apprendimento automatico", "it", "Methodology"),
    ("proponemos un nuevo método de aprendizaje profundo", "es", "Methodology"),
    ("propomos um novo método de aprendizado de máquina", "pt", "Methodology"),
    ("proponujemy nową metodę głębokiego uczenia", "pl", "Methodology"),
    ("wij stellen een nieuwe methode voor machinaal leren", "nl", "Methodology"),
    ("novel architecture transformer language model", "en", "Methodology"),
    ("new algorithm optimization neural network", "en", "Methodology"),
    # ── Dataset ──────────────────────────────────────────────────────────
    ("we release a new annotated dataset benchmark", "en", "Dataset"),
    ("nous présentons un nouveau corpus annoté", "fr", "Dataset"),
    ("wir veröffentlichen einen neuen Datensatz", "de", "Dataset"),
    ("presentiamo un nuovo dataset annotato", "it", "Dataset"),
    ("presentamos un nuevo corpus anotado benchmark", "es", "Dataset"),
    ("apresentamos um novo conjunto de dados anotado", "pt", "Dataset"),
    ("publikujemy nowy zbiór danych adnotowany", "pl", "Dataset"),
    ("multilingual evaluation benchmark corpus NLP", "en", "Dataset"),
    ("crowdsourced annotation dataset shared task", "en", "Dataset"),
    ("open-source dataset publicly available evaluation", "en", "Dataset"),
    # ── Review ───────────────────────────────────────────────────────────
    ("survey of deep learning natural language processing", "en", "Review"),
    ("revue systématique apprentissage automatique", "fr", "Review"),
    ("Überblick über maschinelles Lernen Methoden", "de", "Review"),
    ("rassegna sistematica apprendimento automatico", "it", "Review"),
    ("revisión sistemática aprendizaje automático", "es", "Review"),
    ("revisão sistemática aprendizado de máquina", "pt", "Review"),
    ("przegląd systematyczny uczenie maszynowe", "pl", "Review"),
    ("systematic literature review machine learning methods", "en", "Review"),
    ("overview of transformer models NLP", "en", "Review"),
    ("comprehensive survey computer vision deep learning", "en", "Review"),
    # ── Applied ──────────────────────────────────────────────────────────
    ("applying machine learning to clinical healthcare", "en", "Applied"),
    ("application apprentissage automatique médecine", "fr", "Applied"),
    ("Anwendung maschinelles Lernen klinische Medizin", "de", "Applied"),
    ("applicazione apprendimento automatico medicina clinica", "it", "Applied"),
    ("aplicación aprendizaje automático medicina clínica", "es", "Applied"),
    ("aplicação aprendizado de máquina saúde clínica", "pt", "Applied"),
    ("zastosowanie uczenia maszynowego medycyna kliniczna", "pl", "Applied"),
    ("NLP legal domain information extraction", "en", "Applied"),
    ("deep learning financial forecasting real-world", "en", "Applied"),
    ("transfer learning biomedical named entity recognition", "en", "Applied"),
    # ── Theoretical ──────────────────────────────────────────────────────
    ("theoretical analysis convergence neural networks", "en", "Theoretical"),
    ("analyse théorique convergence réseaux de neurones", "fr", "Theoretical"),
    ("theoretische Analyse Konvergenz neuronale Netze", "de", "Theoretical"),
    ("analisi teorica convergenza reti neurali", "it", "Theoretical"),
    ("análisis teórico convergencia redes neuronales", "es", "Theoretical"),
    ("análise teórica convergência redes neurais", "pt", "Theoretical"),
    ("analiza teoretyczna zbieżność sieci neuronowe", "pl", "Theoretical"),
    ("generalization bounds PAC learning statistical", "en", "Theoretical"),
    ("information theoretic machine learning proofs", "en", "Theoretical"),
    ("sample complexity regret bounds online learning", "en", "Theoretical"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Cross-lingual contradiction filter (title-only, very conservative)
# ─────────────────────────────────────────────────────────────────────────────

# Words in any language that strongly signal a survey/review paper in the title
_REVIEW_TITLE_SIGNALS = re.compile(
    r"\b(survey|review|overview|revue|überblick|rassegna|revisión|revisão|"
    r"przegląd|overzicht|ανασκόπηση|état\s+de\s+l.art|stand\s+der\s+technik|"
    r"stato\s+dell.arte|estado\s+del\s+arte|panorama|bilan|synthèse)\b",
    re.IGNORECASE,
)

# Words in titles that strongly signal a dataset paper
_DATASET_TITLE_SIGNALS = re.compile(
    r"\b(dataset|corpus|benchmark|treebank|testbed|leaderboard|shared\s+task|"
    r"datensatz|insieme\s+di\s+dati|conjunto\s+de\s+datos|zbiór\s+danych|"
    r"dados|données|dati|данные)\b",
    re.IGNORECASE,
)


def _title_contradicts(title: str, intended_label: str) -> bool:
    """Return True if the title clearly contradicts the intended label."""
    t = title.lower()
    if intended_label != "Review" and _REVIEW_TITLE_SIGNALS.search(t):
        return True
    if intended_label != "Dataset" and _DATASET_TITLE_SIGNALS.search(t):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fingerprint(title: str, abstract: str) -> str:
    norm = re.sub(r"\s+", " ", (title + abstract).lower().strip())
    return hashlib.md5(norm.encode()).hexdigest()


def _load_seen_fps(base_file: Path) -> set[str]:
    """Load fingerprints from an existing JSONL to avoid duplicates."""
    seen = set()
    if not base_file.exists():
        return seen
    with open(base_file, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                msgs = row.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                # Extract title/abstract from user message
                title_match = re.search(r"Title: (.+?)\nAbstract:", user_msg, re.DOTALL)
                abstract_match = re.search(r"Abstract: (.+)$", user_msg, re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                abstract = abstract_match.group(1).strip() if abstract_match else ""
                seen.add(_fingerprint(title, abstract))
            except Exception:
                continue
    log.info(f"Loaded {len(seen):,} existing fingerprints from {base_file.name}")
    return seen


def _parse_openaire_result(r: dict) -> tuple[str, str] | None:
    """Extract (title, abstract) from an OpenAIRE result entry. Returns None on failure."""
    try:
        meta = r["metadata"]["oaf:entity"]["oaf:result"]
        title_raw = meta.get("title", [{}])
        title = (
            title_raw[0].get("$", "")
            if isinstance(title_raw, list)
            else title_raw.get("$", "")
        ).strip()
        desc_raw = meta.get("description", [])
        abstract = (desc_raw[0].get("$", "") if desc_raw else "").strip()
        if not title or not abstract or len(abstract) < 60:
            return None
        return title, abstract
    except (KeyError, IndexError, TypeError):
        return None


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
# Main fetch loop
# ─────────────────────────────────────────────────────────────────────────────


def fetch_multilingual(
    per_label: int,
    seen: set[str],
    languages: list[str],
    page_size: int = 25,
    sleep: float = 0.5,
) -> list[dict]:
    try:
        import requests
    except ImportError:
        log.error("requests not installed. Run: uv add requests")
        return []

    BASE_URL = "https://api.openaire.eu/search/publications"
    counts: Counter = Counter()
    lang_counts: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []

    log.info(
        f"Running {len(MULTILINGUAL_QUERIES)} multilingual queries "
        f"(langs={languages}, target={per_label}/label) …"
    )

    for query, query_lang, label in MULTILINGUAL_QUERIES:
        if counts[label] >= per_label:
            continue

        log.info(f"  [{label}] [{query_lang}] {query[:60]!r}")
        try:
            resp = requests.get(
                BASE_URL,
                # Note: OpenAIRE does NOT support a 'lang' filter param;
                # we use multilingual query strings to attract non-English results.
                params={"keywords": query, "format": "json", "size": page_size},
                timeout=20,
            )
            if resp.status_code != 200:
                log.warning(f"    HTTP {resp.status_code} — skipping")
                time.sleep(sleep)
                continue

            data = resp.json()
            results = (
                data.get("response", {}).get("results", {}).get("result", []) or []
            )

            added = 0
            for r in results:
                if counts[label] >= per_label:
                    break
                parsed = _parse_openaire_result(r)
                if parsed is None:
                    continue
                title, abstract = parsed

                # Detect language from the response metadata
                detected_lang = "?"
                try:
                    lang_obj = r["metadata"]["oaf:entity"]["oaf:result"].get(
                        "language", {}
                    )
                    classid = lang_obj.get("@classid", "").lower()  # e.g. 'eng', 'fra'
                    # Map 3-letter ISO-639-2 → 2-letter ISO-639-1
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
                        "und": "?",
                    }
                    detected_lang = _iso3_to_2.get(
                        classid, classid[:2] if len(classid) >= 2 else "?"
                    )
                except Exception:
                    pass

                # Conservative title-level contradiction filter
                if _title_contradicts(title, label):
                    continue

                fp = _fingerprint(title, abstract)
                if fp in seen:
                    continue

                seen.add(fp)
                counts[label] += 1
                lang_counts[detected_lang][label] += 1
                records.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "label": label,
                        "source": f"openaire_{detected_lang}",
                        "lang": detected_lang,
                    }
                )
                added += 1

            log.info(
                f"    → added {added} / {len(results)} results (running: {dict(counts)})"
            )
            time.sleep(sleep)

        except Exception as e:
            log.warning(f"    Query failed: {e}")
            time.sleep(sleep * 2)
            continue

    log.info(f"\nMultilingual fetch complete: {len(records)} records — {dict(counts)}")
    log.info("Per-language breakdown:")
    for lang, cnts in sorted(lang_counts.items()):
        log.info(f"  {lang}: {dict(cnts)}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Expand research-intent dataset with multilingual OpenAIRE papers."
    )
    parser.add_argument(
        "--base_file",
        default="~/.ethicalabs/datasets/research-intent/train.jsonl",
        help="Existing train.jsonl to load fingerprints from (avoids duplicates)",
    )
    parser.add_argument(
        "--output",
        default="~/.ethicalabs/datasets/research-intent/train_multilingual.jsonl",
        help="Output JSONL for multilingual records",
    )
    parser.add_argument(
        "--per_label",
        type=int,
        default=300,
        help="Target records per label from OpenAIRE (default: 300)",
    )
    parser.add_argument(
        "--languages",
        default="en,fr,de,it,es,pt,nl,pl",
        help="Comma-separated ISO-639-1 language codes to include",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After writing, merge base_file + output into train_full.jsonl",
    )
    args = parser.parse_args()

    base_file = Path(args.base_file).expanduser()
    out_file = Path(args.output).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    languages = [label.strip() for label in args.languages.split(",") if label.strip()]

    # Load fingerprints from the base dataset to avoid duplicates
    seen = _load_seen_fps(base_file)

    # Fetch multilingual records
    records = fetch_multilingual(
        per_label=args.per_label,
        seen=seen,
        languages=languages,
    )

    if not records:
        log.warning("No records fetched. Check network connectivity.")
        return

    # Write multilingual JSONL
    log.info(f"Writing {len(records)} records to {out_file} …")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "messages": to_messages(r["title"], r["abstract"], r["label"]),
                "label": r["label"],
                "source": r["source"],
                "lang": r.get("lang", "?"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Per-label / per-language stats
    label_counts = Counter(r["label"] for r in records)
    lang_label: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        lang_label[r.get("lang", "?")][r["label"]] += 1

    stats = {
        "total": len(records),
        "per_label": {lbl: label_counts.get(lbl, 0) for lbl in LABELS},
        "per_language": {lang: dict(cnts) for lang, cnts in sorted(lang_label.items())},
    }
    stats_path = out_file.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    log.info(f"\n✅ Multilingual expansion written to {out_file}")
    log.info(f"   Stats → {stats_path}")

    # Optional merge
    if args.merge and base_file.exists():
        merged_path = out_file.parent / "train_full.jsonl"
        log.info(f"\nMerging {base_file.name} + {out_file.name} → {merged_path.name} …")
        with open(merged_path, "w", encoding="utf-8") as out:
            for src in [base_file, out_file]:
                with open(src, encoding="utf-8") as inp:
                    for line in inp:
                        out.write(line)
        merged_count = sum(1 for _ in open(merged_path, encoding="utf-8"))
        log.info(f"   Merged: {merged_count:,} total records → {merged_path}")
        log.info("\nUse for SFT:")
        log.info(f"  --dataset_name {merged_path}")


if __name__ == "__main__":
    main()
