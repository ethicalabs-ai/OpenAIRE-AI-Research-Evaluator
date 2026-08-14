import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import server


@pytest.fixture
def client():
    with TestClient(server.app) as client:
        yield client


def test_health_check_healthy(client):
    """Verify health endpoint when redis connection is healthy."""
    with patch("server.celery_app.connection"):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


def test_health_check_error(client):
    """Verify health endpoint when redis broker is unreachable."""
    with patch(
        "server.celery_app.connection",
        side_effect=Exception("Redis connection refused"),
    ):
        response = client.get("/api/health")
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["status"] == "error"
        assert (
            "connection" in data["detail"]["error"].lower()
            or "redis" in data["detail"]["error"].lower()
        )


def test_fallback_static_assets(client):
    """Verify fallback and not found handlers return 200 for single page application index routing."""
    with patch("fastapi.responses.FileResponse") as mock_response:
        mock_response.return_value = "IndexHTML"
        response = client.get("/some-random-route")
        assert response.status_code == 200


def test_mcp_requires_bearer_auth(client):
    """MCP endpoint must reject requests without valid Bearer token."""
    response = client.get(
        "/api/mcp/classify",
        params={"title": "Test", "abstract": "Test"},
    )
    assert response.status_code == 401


def test_mcp_rejects_wrong_token(client):
    """MCP endpoint must reject wrong Bearer token."""
    response = client.get(
        "/api/mcp/classify",
        params={"title": "Test", "abstract": "Test"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_mcp_accepts_valid_token(client, monkeypatch):
    """MCP endpoint accepts valid Bearer token."""
    monkeypatch.setattr("server.API_KEY", "test-key")

    # Mock the Celery task to avoid Redis dependency
    with patch("tasks.classify_mcp") as mock_task:
        mock_task.delay.return_value.get.return_value = {
            "label": "Methodology",
            "probabilities": {"Methodology": 1.0},
        }
        mock_task.delay.return_value.ready.return_value = True
        response = client.get(
            "/api/mcp/classify",
            params={"title": "Test", "abstract": "Test"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


# ── Stats endpoint tests ──────────────────────────────────────────────────────


def test_stats_empty_db(temp_db, monkeypatch):
    """Stats on empty DB returns zeros."""
    monkeypatch.setattr("server._get_redis", lambda: None)  # bypass Redis cache
    import server as _srv
    from database import SessionLocal

    db = SessionLocal()
    try:
        result = _srv.get_stats(db=db)
        assert isinstance(result, dict)
        assert result["total_papers"] == 0
        assert result["total_annotations"] == 0
        assert result["human_annotations"] == 0
        assert result["llm_annotations"] == 0
        assert result["flagged_papers"] == 0
        assert result["consensus_distribution"] == {}
    finally:
        db.close()


def test_stats_with_data(temp_db, monkeypatch):
    """Stats reflect actual paper and annotation counts."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        # Create papers
        p1 = PaperRecord(
            doi="10.1234/stats.1",
            title="Paper 1",
            abstract="abs",
            initial_intent="Methodology",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        p2 = PaperRecord(
            doi="10.1234/stats.2",
            title="Paper 2",
            abstract="abs",
            initial_intent="Review",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([p1, p2])
        db.flush()

        # Create annotations
        a1 = DBAnnotation(
            paper_doi="10.1234/stats.1",
            proposed_label="Methodology",
            annotator_type="human",
            user_id="test-user",
            model_version=server.MODEL_VERSION,
        )
        a2 = DBAnnotation(
            paper_doi="10.1234/stats.2",
            proposed_label="Review",
            annotator_type="llm",
            llm_model="test-model",
            model_version=server.MODEL_VERSION,
        )
        a3 = DBAnnotation(
            paper_doi="10.1234/stats.1",
            proposed_label="Dataset",
            annotator_type="llm",
            llm_model="test-model-2",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([a1, a2, a3])
        db.commit()

        result = _srv.get_stats(db=db)

        assert result["total_papers"] == 2
        assert result["total_annotations"] == 3
        assert result["human_annotations"] == 1
        assert result["llm_annotations"] == 2
        assert result["flagged_papers"] == 0
    finally:
        db.close()


def test_stats_flagged(temp_db, monkeypatch):
    """Papers with at least one flag annotation count as flagged."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = PaperRecord(
            doi="10.1234/stats.flagged",
            title="F",
            abstract="x",
            initial_intent="Methodology",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        p_single = PaperRecord(
            doi="10.1234/stats.flagged-single",
            title="F1",
            abstract="x",
            initial_intent="Review",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([p, p_single])
        db.flush()
        for i in range(4):
            db.add(
                DBAnnotation(
                    paper_doi="10.1234/stats.flagged",
                    is_flagged=True,
                    annotator_type="llm",
                    llm_model=f"test-model-{i}",
                    model_version=server.MODEL_VERSION,
                )
            )
        # A single flag must count too (the old >3 rule hid these).
        db.add(
            DBAnnotation(
                paper_doi="10.1234/stats.flagged-single",
                is_flagged=True,
                annotator_type="llm",
                llm_model="single-model",
                model_version=server.MODEL_VERSION,
            )
        )
        db.commit()

        result = _srv.get_stats(db=db)
        assert result["flagged_papers"] == 2
    finally:
        db.close()


def test_classify_min_chars_guard(monkeypatch):
    """Default classify guard rejects <50-char abstracts; min_chars=1 accepts them
    (the collab detail panel passes min_chars=1 for short-but-real abstracts)."""
    from dataclasses import dataclass, field

    from fastapi.testclient import TestClient

    import server as _srv

    @dataclass
    class FakeResult:
        label: str = "Applied"
        probabilities: dict = field(default_factory=lambda: {"Applied": 0.9, "Methodology": 0.1})

    monkeypatch.setattr(_srv, "classify_paper", lambda title, abstract: FakeResult())
    client = TestClient(_srv.app)

    short_abstract = "National Science Foundation's URExSRN;SES-1444755"  # 49 chars
    assert len(short_abstract) < 50
    payload = {"title": "T", "abstract": short_abstract}

    # Default guard (free-text/streaming tabs) still rejects <50 chars
    r = client.post("/api/classify/intent", json=payload)
    assert r.status_code == 400

    # Collab panel path accepts it and returns the prediction
    r = client.post("/api/classify/intent?min_chars=1", json=payload)
    assert r.status_code == 200
    assert r.json()["label"] == "Applied"


def test_stats_consensus(temp_db, monkeypatch):
    """Consensus distribution uses majority vote per paper."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        # Paper 1: two Methodology votes → Methodology
        p1 = PaperRecord(
            doi="10.1234/stats.c1",
            title="C1",
            abstract="x",
            initial_intent="Dataset",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        # Paper 2: no annotations → initial_intent
        p2 = PaperRecord(
            doi="10.1234/stats.c2",
            title="C2",
            abstract="x",
            initial_intent="Review",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        # Paper 3: tie (Methodology + Dataset) → excluded
        p3 = PaperRecord(
            doi="10.1234/stats.c3",
            title="C3",
            abstract="x",
            initial_intent="Applied",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([p1, p2, p3])
        db.flush()

        db.add_all(
            [
                DBAnnotation(
                    paper_doi="10.1234/stats.c1",
                    proposed_label="Methodology",
                    annotator_type="human",
                    user_id="u1",
                    model_version=server.MODEL_VERSION,
                ),
                DBAnnotation(
                    paper_doi="10.1234/stats.c1",
                    proposed_label="Methodology",
                    annotator_type="llm",
                    llm_model="m",
                    model_version=server.MODEL_VERSION,
                ),
                DBAnnotation(
                    paper_doi="10.1234/stats.c3",
                    proposed_label="Methodology",
                    annotator_type="human",
                    user_id="u2",
                    model_version=server.MODEL_VERSION,
                ),
                DBAnnotation(
                    paper_doi="10.1234/stats.c3",
                    proposed_label="Dataset",
                    annotator_type="llm",
                    llm_model="m",
                    model_version=server.MODEL_VERSION,
                ),
            ]
        )
        db.commit()

        result = _srv.get_stats(db=db)
        dist = result["consensus_distribution"]
        assert dist.get("Methodology") == 1  # paper 1
        assert dist.get("Review") == 1  # paper 2 (initial_intent)
        assert "Applied" not in dist  # paper 3: tie, excluded
    finally:
        db.close()


def test_stats_cache_hit(temp_db, monkeypatch):
    """_cached returns cached value without calling compute."""
    monkeypatch.setattr("server._get_redis", lambda: None)  # no Redis
    import server as _srv

    call_count = [0]

    def _compute():
        call_count[0] += 1
        return {"x": 1}

    # No Redis → compute always called
    r1 = _srv._cached("test-key", 60, _compute)
    r2 = _srv._cached("test-key", 60, _compute)
    assert r1 == {"x": 1}
    assert r2 == {"x": 1}
    assert call_count[0] == 2  # no cache without Redis


# ── Judge endpoint tests ──────────────────────────────────────────────────────


def test_judge_rejects_no_auth(client):
    """POST /api/annotations/judge requires Bearer token."""
    response = client.post(
        "/api/annotations/judge",
        json={
            "doi": "10.1234/test",
            "title": "T",
            "abstract": "A",
            "llm_model": "test-model",
        },
    )
    assert response.status_code == 401


def test_judge_rejects_wrong_auth(client, monkeypatch):
    """POST /api/annotations/judge rejects wrong token."""
    monkeypatch.setattr("server.API_KEY", "correct-key")
    response = client.post(
        "/api/annotations/judge",
        json={
            "doi": "10.1234/test",
            "title": "T",
            "abstract": "A",
            "llm_model": "test-model",
        },
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_judge_creates_paper_and_annotation(temp_db, monkeypatch):
    """Judge endpoint creates PaperRecord + Annotation in one call."""
    monkeypatch.setattr("server.API_KEY", "test-key")
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from fastapi.testclient import TestClient

    client = TestClient(_srv.app)
    response = client.post(
        "/api/annotations/judge",
        json={
            "doi": "10.1234/judge-test",
            "title": "Judge Test",
            "abstract": "A novel approach to testing.",
            "llm_model": "test-model",
            "initial_intent": "Applied",
            "proposed_label": "Methodology",
            "is_flagged": True,
            "flag_reason": "garbled text",
            "comment": "[HIGH confidence] Looks good.",
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    # Verify in DB
    from database import SessionLocal
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = (
            db.query(PaperRecord)
            .filter(PaperRecord.doi == "10.1234/judge-test")
            .first()
        )
        assert p is not None
        assert p.initial_intent == "Applied"

        a = (
            db.query(DBAnnotation)
            .filter(
                DBAnnotation.paper_doi == "10.1234/judge-test",
                DBAnnotation.llm_model == "test-model",
            )
            .first()
        )
        assert a is not None
        assert a.proposed_label == "Methodology"
        assert a.is_flagged is True
        assert a.flag_reason == "garbled text"
        # Version stamp: judge annotations belong to the active round
        assert p.model_version == server.MODEL_VERSION
        assert a.model_version == server.MODEL_VERSION
    finally:
        db.close()


def test_judge_upserts_existing(temp_db, monkeypatch):
    """Second POST with same doi+model overwrites, doesn't duplicate."""
    monkeypatch.setattr("server.API_KEY", "test-key")
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from fastapi.testclient import TestClient

    client = TestClient(_srv.app)
    payload = {
        "doi": "10.1234/judge-upsert",
        "title": "T",
        "abstract": "A",
        "llm_model": "test-model",
    }
    headers = {"Authorization": "Bearer test-key"}

    r1 = client.post("/api/annotations/judge", json=payload, headers=headers)
    assert r1.status_code == 200

    r2 = client.post(
        "/api/annotations/judge",
        json={
            **payload,
            "proposed_label": "Review",
            "comment": "updated",
        },
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "saved"

    from database import SessionLocal
    from models import Annotation as DBAnnotation

    db = SessionLocal()
    try:
        annotations = (
            db.query(DBAnnotation)
            .filter(
                DBAnnotation.paper_doi == "10.1234/judge-upsert",
                DBAnnotation.llm_model == "test-model",
            )
            .all()
        )
        assert len(annotations) == 1
        assert annotations[0].proposed_label == "Review"
        assert annotations[0].comment == "updated"
    finally:
        db.close()


def test_hub_read_only_blocks_writes(temp_db, monkeypatch):
    """When HUB_READ_ONLY is set, vote/judge/import return 503."""
    monkeypatch.setattr("server.HUB_READ_ONLY", True)
    monkeypatch.setattr("server.API_KEY", "test-key")
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from fastapi.testclient import TestClient

    client = TestClient(_srv.app)

    # Paper import blocked
    r = client.post(
        "/api/annotations/papers",
        json={
            "doi": "10.1234/ro-test",
            "title": "T",
            "abstract": "A",
            "initial_intent": "Applied",
        },
    )
    assert r.status_code == 503

    # Vote blocked
    r = client.post(
        "/api/annotations/vote",
        json={
            "doi": "10.1234/ro-test",
            "proposed_label": "Applied",
        },
    )
    assert r.status_code == 503

    # Judge blocked
    r = client.post(
        "/api/annotations/judge",
        json={
            "doi": "10.1234/ro-test",
            "title": "T",
            "abstract": "A",
            "llm_model": "m",
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 503

    # Login (GET) should still work — not a mutation
    r = client.get("/api/auth/login")
    assert r.status_code == 200  # redirect to HF OAuth


# ── Versioned rounds tests ────────────────────────────────────────────────────


def test_import_sets_model_version(temp_db, monkeypatch):
    """Papers imported through the hub are stamped with the active round."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from fastapi.testclient import TestClient

    client = TestClient(_srv.app)
    r = client.post(
        "/api/annotations/papers",
        json={
            "doi": "10.1234/import-ver",
            "title": "T",
            "abstract": "A novel method.",
            "initial_intent": "Applied",
            "source": "arxiv",
        },
    )
    assert r.status_code == 200

    from database import SessionLocal
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = (
            db.query(PaperRecord)
            .filter(PaperRecord.doi == "10.1234/import-ver")
            .first()
        )
        assert p is not None
        assert p.model_version == server.MODEL_VERSION
    finally:
        db.close()


def test_vote_sets_model_version(temp_db, monkeypatch):
    """Human votes are stamped with the active round."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from auth_utils import create_session_token
    from fastapi.testclient import TestClient

    client = TestClient(_srv.app)
    r = client.post(
        "/api/annotations/papers",
        json={
            "doi": "10.1234/vote-ver",
            "title": "T",
            "abstract": "A novel approach.",
            "initial_intent": "Methodology",
            "source": "arxiv",
        },
    )
    assert r.status_code == 200

    token = create_session_token("voter", "Voter")
    r = client.post(
        "/api/annotations/vote",
        json={"doi": "10.1234/vote-ver", "proposed_label": "Review"},
        cookies={"session_token": token},
    )
    assert r.status_code == 200

    from database import SessionLocal
    from models import Annotation as DBAnnotation

    db = SessionLocal()
    try:
        a = (
            db.query(DBAnnotation)
            .filter(
                DBAnnotation.paper_doi == "10.1234/vote-ver",
                DBAnnotation.user_id == "voter",
            )
            .first()
        )
        assert a is not None
        assert a.model_version == server.MODEL_VERSION
    finally:
        db.close()


def test_list_papers_version_filter(temp_db, monkeypatch):
    """Paper list is scoped to the round; default view = active version."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from fastapi.testclient import TestClient
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        p_old = PaperRecord(
            doi="10.1234/ver.legacy",
            title="Legacy",
            abstract="abs",
            initial_intent="Methodology",
            source="arxiv",
            model_version="v0.1.3",
        )
        p_new = PaperRecord(
            doi="10.1234/ver.active",
            title="Active",
            abstract="abs",
            initial_intent="Review",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([p_old, p_new])
        db.flush()
        db.add(
            DBAnnotation(
                paper_doi=p_old.doi,
                proposed_label="Applied",
                annotator_type="human",
                user_id="u1",
                model_version="v0.1.3",
            )
        )
        db.add(
            DBAnnotation(
                paper_doi=p_new.doi,
                proposed_label="Dataset",
                annotator_type="human",
                user_id="u2",
                model_version=server.MODEL_VERSION,
            )
        )
        db.commit()
    finally:
        db.close()

    client = TestClient(_srv.app)

    # Default: active round only
    r = client.get("/api/annotations/papers")
    data = r.json()
    assert data["total"] == 1
    assert data["papers"][0]["doi"] == "10.1234/ver.active"
    assert data["papers"][0]["consensus_label"] == "Dataset"
    assert data["papers"][0]["vote_count"] == 1

    # Explicit legacy round
    r = client.get("/api/annotations/papers", params={"version": "v0.1.3"})
    data = r.json()
    assert data["total"] == 1
    assert data["papers"][0]["doi"] == "10.1234/ver.legacy"
    assert data["papers"][0]["consensus_label"] == "Applied"
    assert data["papers"][0]["vote_count"] == 1


def test_get_paper_details_version_filter(temp_db, monkeypatch):
    """Details show only the requested round's annotations + consensus."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from fastapi.testclient import TestClient
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        p = PaperRecord(
            doi="10.1234/ver.detail",
            title="D",
            abstract="abs",
            initial_intent="Methodology",
            source="arxiv",
            model_version="v0.1.3",
        )
        db.add(p)
        db.flush()
        db.add(
            DBAnnotation(
                paper_doi=p.doi,
                proposed_label="Applied",
                annotator_type="human",
                user_id="u1",
                model_version="v0.1.3",
            )
        )
        db.add(
            DBAnnotation(
                paper_doi=p.doi,
                proposed_label="Review",
                annotator_type="human",
                user_id="u2",
                model_version=server.MODEL_VERSION,
            )
        )
        db.commit()
    finally:
        db.close()

    client = TestClient(_srv.app)

    # Default (active round): only the active annotation
    r = client.get("/api/annotations/papers/10.1234/ver.detail")
    data = r.json()
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["proposed_label"] == "Review"
    assert data["consensus_label"] == "Review"

    # Legacy round: only the legacy annotation
    r = client.get(
        "/api/annotations/papers/10.1234/ver.detail",
        params={"version": "v0.1.3"},
    )
    data = r.json()
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["proposed_label"] == "Applied"
    assert data["consensus_label"] == "Applied"


def test_stats_version_filter(temp_db, monkeypatch):
    """Stats are scoped to the round; archive_versions and labels exposed."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import Annotation as DBAnnotation
    from models import PaperRecord

    db = SessionLocal()
    try:
        p_old = PaperRecord(
            doi="10.1234/vstats.legacy",
            title="L",
            abstract="x",
            initial_intent="Methodology",
            source="arxiv",
            model_version="v0.1.3",
        )
        p_new = PaperRecord(
            doi="10.1234/vstats.active",
            title="A",
            abstract="x",
            initial_intent="Review",
            source="arxiv",
            model_version=server.MODEL_VERSION,
        )
        db.add_all([p_old, p_new])
        db.flush()
        db.add(
            DBAnnotation(
                paper_doi=p_old.doi,
                proposed_label="Applied",
                annotator_type="human",
                user_id="u1",
                model_version="v0.1.3",
            )
        )
        db.add(
            DBAnnotation(
                paper_doi=p_new.doi,
                proposed_label="Dataset",
                annotator_type="human",
                user_id="u2",
                model_version=server.MODEL_VERSION,
            )
        )
        db.commit()

        active = _srv.get_stats(db=db)
        assert active["total_papers"] == 1
        assert active["human_annotations"] == 1
        assert active["model_version"] == server.MODEL_VERSION
        assert "v0.1.3" in active["archive_versions"]
        assert server.MODEL_VERSION not in active["archive_versions"]
        assert len(active["labels"]) == 6
        assert "Unclassifiable" in active["labels"]

        legacy = _srv.get_stats(db=db, version="v0.1.3")
        assert legacy["total_papers"] == 1
        assert legacy["human_annotations"] == 1
        # archive_versions is relative to the ACTIVE round, not the queried one
        assert legacy["archive_versions"] == active["archive_versions"]
    finally:
        db.close()
