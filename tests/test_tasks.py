import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from tasks import ping_task


def test_ping_task():
    """Verify that ping_task returns 'pong'."""
    assert ping_task() == "pong"


# ── classify_and_judge tests ──────────────────────────────────────────────────


def test_classify_and_judge_happy_path(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """A new paper should be classified, judged, and annotated successfully."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.1",
        title="Test Paper",
        abstract="We propose a novel method.",
        model="test-model",
    )

    assert result["status"] == "ok"
    assert result["proposed_label"] == "Methodology"
    assert result["label_valid"] is True
    assert result["is_flagged"] is False


def test_classify_and_judge_already_judged(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """Calling the task twice with the same DOI + model should skip the second."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result1 = classify_and_judge(
        doi="10.1234/test.2",
        title="Test Paper 2",
        abstract="We propose another method.",
        model="test-model",
    )
    assert result1["status"] == "ok"

    result2 = classify_and_judge(
        doi="10.1234/test.2",
        title="Test Paper 2",
        abstract="We propose another method.",
        model="test-model",
    )
    assert result2["status"] == "skipped"
    assert result2["reason"] == "already-judged"


def test_classify_and_judge_llm_error(temp_db, mock_classifier, monkeypatch):
    """When the LLM judge raises, the task should return an error status."""

    def _raise_llm_error(*a, **kw):
        raise ValueError("LLM timeout")

    monkeypatch.setattr(
        "llm_judge.judge_paper",
        _raise_llm_error,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.3",
        title="Test Paper 3",
        abstract="An abstract that will cause an error.",
        model="test-model",
    )

    assert result["status"] == "error"
    assert "llm-error" in result["reason"]


def test_classify_and_judge_different_models_same_paper(
    temp_db, mock_judge_verdict, mock_classifier, monkeypatch
):
    """Same paper with different models should create separate annotations."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result1 = classify_and_judge(
        doi="10.1234/test.4",
        title="Test Paper 4",
        abstract="Multi-model test.",
        model="model-a",
    )
    assert result1["status"] == "ok"

    result2 = classify_and_judge(
        doi="10.1234/test.4",
        title="Test Paper 4",
        abstract="Multi-model test.",
        model="model-b",
    )
    assert result2["status"] == "ok"


def test_classify_and_judge_with_initial_intent(
    temp_db, mock_judge_verdict, monkeypatch
):
    """When initial_intent is provided, the classifier should not be called."""
    monkeypatch.setattr(
        "llm_judge.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    from tasks import classify_and_judge

    result = classify_and_judge(
        doi="10.1234/test.5",
        title="Test Paper 5",
        abstract="Pre-labeled paper.",
        model="test-model",
        initial_intent="Review",
        source="dataset",
    )

    assert result["status"] == "ok"
    # Verify the paper record was created with the given initial_intent
    from database import SessionLocal
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = db.query(PaperRecord).filter(PaperRecord.doi == "10.1234/test.5").first()
        assert p is not None
        assert p.initial_intent == "Review"
        assert p.source == "dataset"
    finally:
        db.close()


# ── classify_mcp tests ────────────────────────────────────────────────────────


class FakeMCPResult:
    label = "Methodology"
    probabilities = {
        "Methodology": 0.87,
        "Dataset": 0.03,
        "Review": 0.02,
        "Applied": 0.06,
        "Theoretical": 0.02,
    }


def test_classify_mcp_returns_label_and_probabilities(monkeypatch):
    """MCP task should return label + all 5 probabilities without DB writes."""
    monkeypatch.setattr(
        "intent_classifier.classify_paper",
        lambda title, abstract: FakeMCPResult(),
    )

    from tasks import classify_mcp

    result = classify_mcp("Test Title", "Test Abstract")

    assert result["label"] == "Methodology"
    assert "probabilities" in result
    assert len(result["probabilities"]) == 5
    assert result["probabilities"]["Methodology"] == 0.87


# ── search queries loader tests ───────────────────────────────────────────────


def test_load_search_queries_from_file(tmp_path):
    """_load_search_queries reads queries from a topics file."""
    topics_file = tmp_path / "topics.txt"
    topics_file.write_text("query one\nquery two\n\n# comment\nquery three\n")

    # Patch the candidate paths to point at our temp file
    import judge_cli

    original = judge_cli._load_search_queries

    def _tmp_loader():
        with open(topics_file) as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    judge_cli._load_search_queries = _tmp_loader
    result = judge_cli._load_search_queries()
    judge_cli._load_search_queries = original

    assert result == ["query one", "query two", "query three"]


def test_load_search_queries_missing_file():
    """_load_search_queries returns empty list when no file found."""

    import judge_cli

    original = judge_cli._load_search_queries

    def _tmp_loader():
        return []

    judge_cli._load_search_queries = _tmp_loader
    result = judge_cli._load_search_queries()
    judge_cli._load_search_queries = original

    assert result == []


# ── process_paper remote mode tests ───────────────────────────────────────────


def test_process_paper_remote_calls_local_classifier(
    temp_db, mock_judge_verdict, monkeypatch
):
    """Remote mode must call local Echo classifier, ignoring stale server initial_intent."""
    import judge_cli

    # Track whether classify_paper was called
    classify_called = [False]

    def fake_classify(title, abstract):
        classify_called[0] = True
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            label: str = "Applied"
            probabilities: dict = None

        return FakeResult()

    monkeypatch.setattr("judge_cli.classify_paper", fake_classify, raising=False)
    monkeypatch.setattr(
        "judge_cli.judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )
    monkeypatch.setattr(
        "judge_cli.post_annotation_remote",
        lambda paper, verdict, model, url, key, initial_intent="": "saved",
    )

    paper = {
        "doi": "10.1234/remote-test",
        "title": "Remote Test",
        "abstract": "Testing remote mode.",
        "initial_intent": "Methodology",  # stale server value
        "source": "catalog",
    }

    judge_cli.process_paper(
        paper,
        "test-model",
        dry_run=False,
        db=None,
        remote_url="https://example.com",
        remote_api_key="test-key",
    )

    assert classify_called[0], "Echo classifier must be called locally in remote mode"


# ── judge_batch tests ─────────────────────────────────────────────────────────


def test_post_annotation_remote_sends_computed_initial_intent(
    mock_judge_verdict, monkeypatch
):
    """Remote payload must carry the locally computed Echo prediction, never the
    hardcoded 'Methodology' default for live-fetched papers."""
    import judge_cli

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"status": "saved"}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(judge_cli.requests, "post", fake_post)

    paper = {
        "doi": "10.1234/remote.1",
        "title": "T",
        "abstract": "a",
        "source": "openaire",
    }  # no initial_intent key — as fetched from OpenAIRE/arXiv
    judge_cli.post_annotation_remote(
        paper,
        mock_judge_verdict,
        "test-model",
        "https://example.com",
        "test-key",
        initial_intent="Theoretical",
    )
    assert captured["payload"]["initial_intent"] == "Theoretical"


def test_process_paper_remote_passes_computed_prediction(
    temp_db, mock_judge_verdict, monkeypatch
):
    """process_paper sends the locally classified label as initial_intent."""
    from types import SimpleNamespace

    import judge_cli

    monkeypatch.setattr(
        judge_cli,
        "classify_paper",
        lambda title, abstract: SimpleNamespace(label="Theoretical"),
    )
    monkeypatch.setattr(
        judge_cli,
        "judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )

    received = {}

    def fake_post(paper, verdict, model, url, key, initial_intent=""):
        received["intent"] = initial_intent
        return "saved"

    monkeypatch.setattr(judge_cli, "post_annotation_remote", fake_post)

    paper = {
        "doi": "10.1234/remote.2",
        "title": "T",
        "abstract": "A short abstract.",
        "source": "openaire",
    }
    ok = judge_cli.process_paper(
        paper,
        "test-model",
        dry_run=False,
        db=None,
        remote_url="https://example.com",
        remote_api_key="test-key",
    )
    assert ok is True
    assert received["intent"] == "Theoretical"


def test_process_paper_remote_skips_catalogued_papers(monkeypatch):
    """Live sources skip papers already in the remote catalog before the LLM call."""
    import judge_cli

    judged = []
    monkeypatch.setattr(judge_cli, "_remote_paper_exists", lambda doi, url: True)
    monkeypatch.setattr(
        judge_cli, "judge_paper", lambda *a, **k: judged.append(1) or None
    )

    paper = {
        "doi": "10.1234/cat.1",
        "title": "T",
        "abstract": "A",
        "source": "openaire",
    }
    ok = judge_cli.process_paper(
        paper,
        "test-model",
        dry_run=False,
        db=None,
        remote_url="https://example.com",
        remote_api_key="test-key",
    )
    assert ok is False
    assert judged == []  # LLM never called


def test_process_paper_remote_judges_new_paper(
    temp_db, mock_judge_verdict, monkeypatch
):
    """A paper not yet in the remote catalog gets judged normally."""
    from types import SimpleNamespace

    import judge_cli

    monkeypatch.setattr(judge_cli, "_remote_paper_exists", lambda doi, url: False)
    monkeypatch.setattr(
        judge_cli,
        "classify_paper",
        lambda title, abstract: SimpleNamespace(label="Applied"),
    )
    monkeypatch.setattr(
        judge_cli,
        "judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )
    captured = {}

    def fake_post(paper, verdict, model, url, key, initial_intent=""):
        captured["intent"] = initial_intent
        return "saved"

    monkeypatch.setattr(judge_cli, "post_annotation_remote", fake_post)

    paper = {
        "doi": "10.1234/new.1",
        "title": "T",
        "abstract": "A",
        "source": "openaire",
    }
    ok = judge_cli.process_paper(
        paper,
        "test-model",
        dry_run=False,
        db=None,
        remote_url="https://example.com",
        remote_api_key="test-key",
    )
    assert ok is True
    assert captured["intent"] == "Applied"


def test_process_paper_remote_catalog_bypasses_exist_check(
    monkeypatch, mock_judge_verdict
):
    """Catalog source (skip_catalogued=False) must not run the exists check —
    exclude_model already filters judged papers."""
    from types import SimpleNamespace

    import judge_cli

    check_called = []
    monkeypatch.setattr(
        judge_cli,
        "_remote_paper_exists",
        lambda doi, url: check_called.append(doi) or True,
    )
    monkeypatch.setattr(
        judge_cli,
        "classify_paper",
        lambda title, abstract: SimpleNamespace(label="Applied"),
    )
    monkeypatch.setattr(
        judge_cli,
        "judge_paper",
        lambda title, abstract, prediction, model, timeout=300: mock_judge_verdict,
    )
    monkeypatch.setattr(judge_cli, "post_annotation_remote", lambda *a, **k: "saved")

    paper = {
        "doi": "10.1234/cat.2",
        "title": "T",
        "abstract": "A",
        "source": "catalog",
    }
    ok = judge_cli.process_paper(
        paper,
        "test-model",
        dry_run=False,
        db=None,
        remote_url="https://example.com",
        remote_api_key="test-key",
        skip_catalogued=False,
    )
    assert ok is True
    assert check_called == []


def test_run_catalog_remote_summary_no_crash(monkeypatch):
    """Catalog remote runs must reach the final summary — total_duplicates /
    total_errors are initialized for non-live sources too (regression:
    UnboundLocalError after --source catalog completed)."""
    from types import SimpleNamespace

    import judge_cli

    args = SimpleNamespace(
        source="catalog",
        n=3,
        delay=0.0,
        dry_run=False,
        model="test-model",
        sort="default",
        api_url="https://example.com",
        api_key="test-key",
        remote=True,
        force=False,
        async_mode=False,
        llm_url="",
        llm_key="",
        version="",
    )

    def fake_fetch(api_url, api_key, model, limit, *, force=False, version=""):
        return [
            {
                "doi": f"10.1234/cat.{i}",
                "title": "T",
                "abstract": "A",
                "source": "catalog",
            }
            for i in range(3)
        ]

    monkeypatch.setattr(judge_cli, "fetch_remote_catalog", fake_fetch)
    monkeypatch.setattr(
        judge_cli,
        "process_paper",
        lambda paper, model, dry_run, db, **kw: True,
    )

    judge_cli.run(args)  # must not raise UnboundLocalError


def test_judge_batch_dedup_and_counts(monkeypatch):
    """Duplicate DOIs within a run are skipped; success/failure counted."""
    import judge_cli

    calls = []

    def fake_process(paper, model, dry_run, db, **kwargs):
        calls.append(paper["doi"])
        return paper["doi"] != "10.1234/err"

    monkeypatch.setattr(judge_cli, "process_paper", fake_process)

    papers = [
        {"doi": "10.1234/dup", "title": "D1", "abstract": "a", "source": "openaire"},
        {"doi": "10.1234/dup", "title": "D1", "abstract": "a", "source": "openaire"},
        {"doi": "10.1234/new", "title": "N", "abstract": "a", "source": "openaire"},
        {"doi": "10.1234/err", "title": "E", "abstract": "a", "source": "openaire"},
    ]
    seen: set[str] = set()
    newly, duplicates, errors = judge_cli.judge_batch(
        papers, "test-model", False, None, seen=seen, target=10
    )

    assert newly == 2  # first occurrence of dup + new
    assert duplicates == 1  # second occurrence of 10.1234/dup
    assert errors == 1  # 10.1234/err returned False
    assert seen == {"10.1234/dup", "10.1234/new", "10.1234/err"}
    assert len(calls) == 3  # duplicate never re-judged


def test_judge_batch_respects_target(monkeypatch):
    """judge_batch stops once the target of new annotations is reached."""
    import judge_cli

    calls = []

    def fake_process(paper, model, dry_run, db, **kwargs):
        calls.append(paper["doi"])
        return True

    monkeypatch.setattr(judge_cli, "process_paper", fake_process)

    papers = [
        {"doi": f"10.1234/p{i}", "title": f"T{i}", "abstract": "a", "source": "arxiv"}
        for i in range(5)
    ]
    newly, _, _ = judge_cli.judge_batch(
        papers, "test-model", False, None, seen=set(), target=2
    )

    assert newly == 2
    assert len(calls) == 2  # stopped after reaching the target


# ── live-source run() discovery tests ─────────────────────────────────────────


def _live_source_args(**overrides):
    """Minimal args namespace for a remote openaire run (no local DB needed)."""
    from types import SimpleNamespace

    defaults = {
        "source": "openaire",
        "n": 100,
        "delay": 0.0,
        "dry_run": True,
        "model": "test-model",
        "sort": "default",
        "api_url": "https://example.com",
        "api_key": "test-key",
        "remote": True,
        "force": False,
        "async_mode": False,
        "llm_url": "",
        "llm_key": "",
        "version": "",
        "no_early_stop": False,
        "no_shuffle": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_run_live_source_stops_after_three_empty_pages(monkeypatch):
    """Default sync run stops after 3 consecutive pages with no new papers."""
    import judge_cli

    monkeypatch.setattr(judge_cli, "SEARCH_QUERIES", ["query a", "query b"])

    fetch_calls = []

    def fake_fetch(query, max_results=25, *, sort_fresh=False, page=1):
        fetch_calls.append((query, page))
        return [
            {
                "doi": f"10.1234/{query[6]}.{page}",
                "title": "T",
                "abstract": "A",
                "source": "openaire",
            }
        ]

    monkeypatch.setattr(judge_cli, "fetch_openaire", fake_fetch)
    monkeypatch.setattr(judge_cli, "process_paper", lambda *a, **k: False)

    judge_cli.run(_live_source_args(no_shuffle=True))

    # Query 'a' pages 1-3 yield nothing -> early stop; query 'b' never starts.
    assert fetch_calls == [("query a", 1), ("query a", 2), ("query a", 3)]


def test_run_live_source_no_early_stop_pages_all_queries(monkeypatch):
    """--no-early-stop keeps paging past empty pages to reach more topics."""
    import judge_cli

    monkeypatch.setattr(judge_cli, "SEARCH_QUERIES", ["query a", "query b"])

    fetch_calls = []

    def fake_fetch(query, max_results=25, *, sort_fresh=False, page=1):
        fetch_calls.append((query, page))
        return [
            {
                "doi": f"10.1234/{query[6]}.{page}",
                "title": "T",
                "abstract": "A",
                "source": "openaire",
            }
        ]

    monkeypatch.setattr(judge_cli, "fetch_openaire", fake_fetch)
    monkeypatch.setattr(judge_cli, "process_paper", lambda *a, **k: False)

    judge_cli.run(_live_source_args(no_early_stop=True, no_shuffle=True))

    # Both queries paged to the max (5 pages each) despite zero progress.
    assert fetch_calls == [("query a", p) for p in range(1, 6)] + [
        ("query b", p) for p in range(1, 6)
    ]


def test_run_live_source_no_shuffle_preserves_order(monkeypatch):
    """--no-shuffle processes topics top-to-bottom and papers in API order."""
    import judge_cli

    monkeypatch.setattr(judge_cli, "SEARCH_QUERIES", ["query a", "query b", "query c"])

    seen_queries = []
    processed_dois = []

    def fake_fetch(query, max_results=25, *, sort_fresh=False, page=1):
        if page >= 2:
            return []
        seen_queries.append(query)
        return [
            {
                "doi": f"10.1234/{query[6]}.1",
                "title": "T1",
                "abstract": "A",
                "source": "openaire",
            },
            {
                "doi": f"10.1234/{query[6]}.2",
                "title": "T2",
                "abstract": "A",
                "source": "openaire",
            },
        ]

    def fake_process(paper, model, dry_run, db, **kwargs):
        processed_dois.append(paper["doi"])
        return False

    monkeypatch.setattr(judge_cli, "fetch_openaire", fake_fetch)
    monkeypatch.setattr(judge_cli, "process_paper", fake_process)

    judge_cli.run(_live_source_args(no_early_stop=True, no_shuffle=True))

    assert seen_queries == ["query a", "query b", "query c"]
    assert processed_dois == [
        "10.1234/a.1",
        "10.1234/a.2",
        "10.1234/b.1",
        "10.1234/b.2",
        "10.1234/c.1",
        "10.1234/c.2",
    ]
