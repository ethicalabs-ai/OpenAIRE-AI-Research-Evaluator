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
    monkeypatch.setattr("server.MCP_API_KEY", "test-key")

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
