"""Tests for metadata service."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app


@pytest.fixture
def client():
    """Test client."""
    return TestClient(app)


def test_metrics(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"db_operations_total" in response.content

