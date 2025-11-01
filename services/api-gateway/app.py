"""
API Gateway - Single entry point, routing, authentication, rate limiting.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import httpx
import os
import jwt

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="api-gateway", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

try:
    from libs.common.observability import wire_otel
    wire_otel(app, "api-gateway")
except ImportError:
    pass

ROUTES = {
    "/v1/audio": {
        "base": os.getenv("INGESTION_URL", "http://ingestion:8001"),
        "target": "/v1/ingest",
    },
    "/v1/ingest": {
        "base": os.getenv("INGESTION_URL", "http://ingestion:8001"),
        "target": "/v1/ingest",
    },
    "/v1/transcripts": {
        "base": os.getenv("METADATA_URL", "http://metadata:8002"),
        "target": None,
    },
    "/v1/search": {
        "base": os.getenv("SEARCH_URL", "http://search:8005"),
        "target": None,
    },
}

JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "dev")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")  # switch to RS256 in prod

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _authenticate(request: Request) -> None:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="missing authorization header")

    token = auth_header.split()[-1]
    try:
        jwt.decode(token, JWT_PUBLIC_KEY, algorithms=[JWT_ALGO], options={"verify_aud": False})
    except Exception as exc:  # keep failure verbose to caller
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@limiter.limit(os.getenv("RATE_LIMIT_PER_MINUTE", "60/minute"))
async def proxy(full_path: str, request: Request):
    """JWT-protected reverse proxy to internal services with rate limiting."""
    if request.url.path == "/health":
        return {"status": "healthy", "service": "api-gateway"}

    if request.url.path == "/":
        return {
            "service": "api-gateway",
            "version": "0.1.0",
            "routes": list(ROUTES.keys()),
        }

    _authenticate(request)

    upstream = None
    matched_prefix = None
    for prefix, info in ROUTES.items():
        if request.url.path.startswith(prefix):
            upstream = info
            matched_prefix = prefix
            break

    if not upstream:
        raise HTTPException(status_code=404, detail="no upstream for path")

    async with httpx.AsyncClient(timeout=30.0) as client:
        body = await request.body()
        headers = dict(request.headers)
        # remove hop headers that confuse upstream
        headers.pop("host", None)
        target_path = upstream["target"]
        if target_path:
            url_path = target_path + request.url.path[len(matched_prefix):]
        else:
            url_path = request.url.path
        try:
            response = await client.request(
                request.method,
                f"{upstream['base']}{url_path}",
                params=request.query_params,
                content=body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() != "transfer-encoding"},
            background=BackgroundTask(response.aclose),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
