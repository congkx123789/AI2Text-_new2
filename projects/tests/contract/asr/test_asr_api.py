"""Contract tests for ASR REST API."""
import pytest
from contract.shared.client import get_asr_client
from contract.shared.validators import validate_openapi_response


@pytest.fixture
def asr_client():
    """ASR HTTP client."""
    return get_asr_client()


def test_asr_health_endpoint(asr_client):
    """Test ASR /health endpoint."""
    response = asr_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert validate_openapi_response(
        data,
        service="asr",
        endpoint="/health",
        method="get",
        status_code=200,
    )


def test_asr_transcribe_endpoint_schema(asr_client):
    """Test /transcribe endpoint schema."""
    from contract.shared.validators import load_openapi_spec
    
    spec = load_openapi_spec("asr")
    assert "/transcribe" in spec["paths"]
    
    # Validate request schema
    post_def = spec["paths"]["/transcribe"]["post"]
    request_body = post_def.get("requestBody", {})
    assert request_body  # Should have request body


def test_asr_stream_endpoint_schema(asr_client):
    """Test /stream WebSocket endpoint schema."""
    from contract.shared.validators import load_openapi_spec
    
    spec = load_openapi_spec("asr")
    assert "/stream" in spec["paths"]
    assert "get" in spec["paths"]["/stream"]
    
    # Should be WebSocket
    get_def = spec["paths"]["/stream"]["get"]
    assert get_def.get("responses", {}).get("101")  # 101 Switching Protocols

