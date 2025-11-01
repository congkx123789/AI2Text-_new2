"""
Prometheus metrics setup
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Optional


def setup_metrics(
    service_name: str,
) -> dict:
    """
    Setup Prometheus metrics for a service

    Args:
        service_name: Name of the service

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "requests_total": Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        ),
        "request_duration": Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
        ),
        "errors_total": Counter(
            "http_errors_total",
            "Total HTTP errors",
            ["method", "endpoint", "error_type"],
        ),
        "service_info": Gauge(
            "service_info",
            "Service information",
            ["service_name", "version"],
        ),
    }

    metrics["service_info"].labels(service_name=service_name, version="0.1.0").set(1)

    return metrics


def generate_metrics_response() -> bytes:
    """Generate Prometheus metrics response"""
    return generate_latest()


