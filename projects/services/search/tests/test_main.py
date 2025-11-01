"""Tests for search service."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app


@pytest.fixture
def client():
    """Test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check."""
    with patch("app.main.qdrant_client") as mock_qdrant:
        mock_qdrant.get_collections.return_value = MagicMock()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]


def test_metrics(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"search_requests_total" in response.content

