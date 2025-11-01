"""HTTP clients for contract testing."""
import os
from typing import Optional
import httpx


def get_base_url(service: str) -> str:
    """Get base URL for service."""
    env_var = f"{service.upper()}_URL"
    return os.getenv(env_var, f"http://{service}:8080")


def get_gateway_client(base_url: Optional[str] = None) -> httpx.Client:
    """Get HTTP client for Gateway service."""
    url = base_url or get_base_url("gateway")
    return httpx.Client(
        base_url=url,
        timeout=30.0,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )


def get_search_client(base_url: Optional[str] = None) -> httpx.Client:
    """Get HTTP client for Search service."""
    url = base_url or get_base_url("search")
    return httpx.Client(
        base_url=url,
        timeout=30.0,
    )


def get_metadata_client(base_url: Optional[str] = None) -> httpx.Client:
    """Get HTTP client for Metadata service."""
    url = base_url or get_base_url("metadata")
    return httpx.Client(
        base_url=url,
        timeout=30.0,
    )


def get_asr_client(base_url: Optional[str] = None) -> httpx.Client:
    """Get HTTP client for ASR service."""
    url = base_url or get_base_url("asr")
    return httpx.Client(
        base_url=url,
        timeout=60.0,  # Longer timeout for ASR
    )

