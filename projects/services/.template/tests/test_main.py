"""Tests for main application."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_metrics(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content

