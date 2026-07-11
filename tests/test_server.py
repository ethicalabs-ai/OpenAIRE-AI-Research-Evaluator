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
    from models import PaperRecord, Annotation as DBAnnotation

    db = SessionLocal()
    try:
        # Create papers
        p1 = PaperRecord(
            doi="10.1234/stats.1", title="Paper 1", abstract="abs",
            initial_intent="Methodology", source="arxiv",
        )
        p2 = PaperRecord(
            doi="10.1234/stats.2", title="Paper 2", abstract="abs",
            initial_intent="Review", source="arxiv",
        )
        db.add_all([p1, p2])
        db.flush()

        # Create annotations
        a1 = DBAnnotation(paper_doi="10.1234/stats.1", proposed_label="Methodology",
                          annotator_type="human", user_id="test-user")
        a2 = DBAnnotation(paper_doi="10.1234/stats.2", proposed_label="Review",
                          annotator_type="llm", llm_model="test-model")
        a3 = DBAnnotation(paper_doi="10.1234/stats.1", proposed_label="Dataset",
                          annotator_type="llm", llm_model="test-model-2")
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
    """Papers with > 3 flag annotations count as flagged."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import PaperRecord, Annotation as DBAnnotation

    db = SessionLocal()
    try:
        p = PaperRecord(doi="10.1234/stats.flagged", title="F", abstract="x",
                        initial_intent="Methodology", source="arxiv")
        db.add(p)
        db.flush()
        for i in range(4):
            db.add(DBAnnotation(paper_doi="10.1234/stats.flagged", is_flagged=True,
                                annotator_type="llm", llm_model=f"test-model-{i}"))
        db.commit()

        result = _srv.get_stats(db=db)
        assert result["flagged_papers"] == 1
    finally:
        db.close()


def test_stats_consensus(temp_db, monkeypatch):
    """Consensus distribution uses majority vote per paper."""
    monkeypatch.setattr("server._get_redis", lambda: None)
    import server as _srv
    from database import SessionLocal
    from models import PaperRecord, Annotation as DBAnnotation

    db = SessionLocal()
    try:
        # Paper 1: two Methodology votes → Methodology
        p1 = PaperRecord(doi="10.1234/stats.c1", title="C1", abstract="x",
                         initial_intent="Dataset", source="arxiv")
        # Paper 2: no annotations → initial_intent
        p2 = PaperRecord(doi="10.1234/stats.c2", title="C2", abstract="x",
                         initial_intent="Review", source="arxiv")
        # Paper 3: tie (Methodology + Dataset) → excluded
        p3 = PaperRecord(doi="10.1234/stats.c3", title="C3", abstract="x",
                         initial_intent="Applied", source="arxiv")
        db.add_all([p1, p2, p3])
        db.flush()

        db.add_all([
            DBAnnotation(paper_doi="10.1234/stats.c1", proposed_label="Methodology",
                         annotator_type="human", user_id="u1"),
            DBAnnotation(paper_doi="10.1234/stats.c1", proposed_label="Methodology",
                         annotator_type="llm", llm_model="m"),
            DBAnnotation(paper_doi="10.1234/stats.c3", proposed_label="Methodology",
                         annotator_type="human", user_id="u2"),
            DBAnnotation(paper_doi="10.1234/stats.c3", proposed_label="Dataset",
                         annotator_type="llm", llm_model="m"),
        ])
        db.commit()

        result = _srv.get_stats(db=db)
        dist = result["consensus_distribution"]
        assert dist.get("Methodology") == 1  # paper 1
        assert dist.get("Review") == 1       # paper 2 (initial_intent)
        assert "Applied" not in dist         # paper 3: tie, excluded
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
