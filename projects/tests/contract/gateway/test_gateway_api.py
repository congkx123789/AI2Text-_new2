"""Contract tests for Gateway REST API."""
import pytest
from contract.shared.client import get_gateway_client
from contract.shared.validators import validate_openapi_response


@pytest.fixture
def gateway_client():
    """Gateway HTTP client."""
    return get_gateway_client()


def test_health_endpoint(gateway_client):
    """Test /health endpoint matches OpenAPI spec."""
    response = gateway_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response schema
    assert validate_openapi_response(
        data,
        service="gateway",
        endpoint="/health",
        method="get",
        status_code=200,
    )
    
    # Validate required fields
    assert "status" in data
    assert data["status"] in ["ok", "unhealthy"]


def test_search_endpoint(gateway_client):
    """Test /search endpoint matches OpenAPI spec."""
    # Mock auth token
    headers = {"Authorization": "Bearer test-token"}
    
    response = gateway_client.get("/search?q=test&limit=10", headers=headers)
    
    # Should be 200 or 401 (if auth required)
    assert response.status_code in [200, 401]
    
    if response.status_code == 200:
        data = response.json()
        assert validate_openapi_response(
            data,
            service="gateway",
            endpoint="/search",
            method="get",
            status_code=200,
        )
        assert "hits" in data
        assert "total" in data


def test_ingest_endpoint_schema(gateway_client):
    """Test /ingest endpoint schema (without actual upload)."""
    # Just validate endpoint exists in spec
    from contract.shared.validators import load_openapi_spec
    
    spec = load_openapi_spec("gateway")
    assert "/ingest" in spec["paths"]
    assert "post" in spec["paths"]["/ingest"]


def test_metadata_endpoint_schema(gateway_client):
    """Test /metadata/{recording_id} endpoint schema."""
    from contract.shared.validators import load_openapi_spec
    
    spec = load_openapi_spec("gateway")
    assert "/metadata/{recording_id}" in spec["paths"]
    assert "get" in spec["paths"]["/metadata/{recording_id}"]

