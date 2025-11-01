"""Observability helpers (OpenTelemetry wiring)."""

from typing import Optional


def wire_otel(app, service_name: str) -> None:
    """Attach OpenTelemetry instrumentation if packages exist."""

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        # OpenTelemetry is optional. Skip silently if not installed.
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


__all__ = ["wire_otel"]

