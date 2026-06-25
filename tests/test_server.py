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
