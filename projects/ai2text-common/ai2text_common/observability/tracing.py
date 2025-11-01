"""
OpenTelemetry tracing setup
"""

from typing import Optional
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


def setup_tracing(
    service_name: str,
    endpoint: Optional[str] = None,
    sampling_rate: float = 1.0,
) -> trace.Tracer:
    """
    Setup OpenTelemetry tracing

    Args:
        service_name: Name of the service
        endpoint: OTLP endpoint URL (e.g., "http://jaeger:4317")
        sampling_rate: Sampling rate (0.0 to 1.0)

    Returns:
        Tracer instance
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(service_name)
    return tracer


