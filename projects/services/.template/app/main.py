"""Main FastAPI application for AI2Text service."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from ai2text_common.observability import setup_tracing, setup_logging
from ai2text_common.schemas import HealthResponse

# Setup logging
setup_logging("service-name")
logger = logging.getLogger(__name__)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting service-name service...")
    setup_tracing("service-name")
    
    # TODO: Initialize connections (NATS, DB, etc.)
    
    logger.info("Service ready")
    yield
    
    # Shutdown
    logger.info("Shutting down service-name service...")
    # TODO: Close connections


app = FastAPI(
    title="AI2Text Service Name",
    version="1.0.0",
    description="Microservice template for AI2Text",
    lifespan=lifespan
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record metrics for all requests."""
    method = request.method
    path = request.url.path
    
    with http_request_duration_seconds.labels(method=method, endpoint=path).time():
        response = await call_next(request)
    
    http_requests_total.labels(
        method=method,
        endpoint=path,
        status=response.status_code
    ).inc()
    
    return response


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    # TODO: Add actual health checks (DB, NATS, etc.)
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        checks={}
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest().decode())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

