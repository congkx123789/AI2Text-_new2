"""Contract tests for Search REST API."""
import pytest
from contract.shared.client import get_search_client
from contract.shared.validators import validate_openapi_response


@pytest.fixture
def search_client():
    """Search HTTP client."""
    return get_search_client()


def test_search_health_endpoint(search_client):
    """Test Search /health endpoint."""
    response = search_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert validate_openapi_response(
        data,
        service="search",
        endpoint="/health",
        method="get",
        status_code=200,
    )


def test_search_endpoint(search_client):
    """Test /search endpoint matches OpenAPI spec."""
    response = search_client.get("/search?q=test&limit=10&threshold=0.7")
    
    assert response.status_code == 200
    data = response.json()
    
    assert validate_openapi_response(
        data,
        service="search",
        endpoint="/search",
        method="get",
        status_code=200,
    )
    
    # Validate required fields
    assert "hits" in data
    assert "total" in data
    assert "query_time_ms" in data
    
    # Validate hit structure
    if data["hits"]:
        hit = data["hits"][0]
        assert "id" in hit
        assert "score" in hit
        assert "text" in hit


def test_search_endpoint_validation(search_client):
    """Test /search endpoint parameter validation."""
    # Invalid limit (too high)
    response = search_client.get("/search?q=test&limit=200")
    assert response.status_code == 422  # Validation error
    
    # Invalid threshold (out of range)
    response = search_client.get("/search?q=test&threshold=1.5")
    assert response.status_code == 422
    
    # Missing required parameter
    response = search_client.get("/search")
    assert response.status_code == 422

