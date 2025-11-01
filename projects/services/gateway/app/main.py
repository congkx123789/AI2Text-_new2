"""
API Gateway - Edge service for authentication, rate limiting, and routing
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import os
import time
from typing import Optional

from ai2text_common import setup_logging, setup_metrics, HealthResponse
from ai2text_common.observability.metrics import generate_metrics_response

logger = setup_logging("gateway")
metrics = setup_metrics("gateway")

app = FastAPI(
    title="AI2Text API Gateway",
    version="1.0.0",
    description="Edge service for authentication and routing",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Backend service URLs
INGESTION_URL = os.getenv("INGESTION_SERVICE_URL", "http://ingestion:8080")
SEARCH_URL = os.getenv("SEARCH_SERVICE_URL", "http://search:8080")
METADATA_URL = os.getenv("METADATA_SERVICE_URL", "http://metadata:8080")

# Rate limiting (simple in-memory, use Redis in production)
rate_limits = {}
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        path = request.url.path

        # Skip rate limiting for health checks
        if path == "/health" or path == "/metrics":
            return await call_next(request)

        # Simple rate limiting (use Redis in production)
        key = f"{client_ip}:{path}"
        current_time = int(time.time())
        
        if key not in rate_limits:
            rate_limits[key] = {"count": 1, "window": current_time}
        elif rate_limits[key]["window"] < current_time - 60:
            rate_limits[key] = {"count": 1, "window": current_time}
        else:
            rate_limits[key]["count"] += 1
        
        if rate_limits[key]["count"] > RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": "Too many requests"}
            )

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


def verify_jwt(token: str) -> dict:
    """Verify JWT token (simplified - implement RS256 verification)"""
    # TODO: Implement proper JWT verification with RS256
    # For now, return a dummy payload
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    # In production: verify with public key
    return {"sub": "user", "email": "user@example.com"}


async def get_current_user(request: Request) -> dict:
    """Extract and verify JWT from request"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    return verify_jwt(token)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy")


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(content=generate_metrics_response(), media_type="text/plain")


@app.post("/v1/ingest")
async def ingest(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Route to ingestion service"""
    async with httpx.AsyncClient() as client:
        body = await request.body()
        response = await client.post(
            f"{INGESTION_URL}/ingest",
            content=body,
            headers={
                "Content-Type": request.headers.get("Content-Type", "multipart/form-data"),
                "X-User-ID": user.get("sub", "unknown"),
            },
            timeout=30.0,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )


@app.get("/v1/search")
async def search(
    request: Request,
    q: str,
    user: dict = Depends(get_current_user)
):
    """Route to search service"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SEARCH_URL}/search",
            params={"q": q, **request.query_params},
            headers={"X-User-ID": user.get("sub", "unknown")},
            timeout=10.0,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )


@app.get("/v1/metadata/{recording_id}")
async def get_metadata(
    recording_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Route to metadata service"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{METADATA_URL}/metadata/{recording_id}",
            headers={"X-User-ID": user.get("sub", "unknown")},
            timeout=5.0,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )


if __name__ == "__main__":
    import uvicorn
    import time
    uvicorn.run(app, host="0.0.0.0", port=8080)

