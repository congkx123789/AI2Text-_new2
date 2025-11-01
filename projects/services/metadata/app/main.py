"""AI2Text Metadata Service - Recording metadata management."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, List
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ai2text_common.observability import setup_tracing, setup_logging
from app.deps import get_settings

# Setup
setup_logging("metadata")
logger = logging.getLogger(__name__)

# Metrics
db_operations_total = Counter("db_operations_total", "Total DB operations", ["operation"])
db_duration_seconds = Histogram("db_duration_seconds", "DB operation duration", ["operation"])

# Database
class Base(DeclarativeBase):
    pass


class RecordingStatus(str, Enum):
    UPLOADED = "uploaded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Recording(Base):
    __tablename__ = "recordings"
    
    recording_id: Mapped[UUID] = mapped_column(primary_key=True)
    status: Mapped[str]
    audio_url: Mapped[str]
    transcript: Mapped[str | None]
    language: Mapped[str]
    duration_sec: Mapped[float | None]
    error: Mapped[str | None]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class CreateRecordingRequest(BaseModel):
    audio_url: str
    language: str
    duration_sec: float | None = None
    metadata: dict = {}


class UpdateRecordingRequest(BaseModel):
    status: RecordingStatus | None = None
    transcript: str | None = None
    error: str | None = None
    metadata: dict = {}


class RecordingResponse(BaseModel):
    recording_id: str
    status: str
    audio_url: str
    transcript: str | None
    language: str
    duration_sec: float | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class RecordingListResponse(BaseModel):
    recordings: List[RecordingResponse]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    checks: dict = {}


# Global engine
engine = None
SessionLocal = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    global engine, SessionLocal
    
    logger.info("Starting metadata service...")
    setup_tracing("metadata")
    
    settings = get_settings()
    
    # Initialize database
    engine = create_async_engine(settings.database_url, echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Metadata service ready")
    yield
    
    logger.info("Shutting down metadata service...")
    await engine.dispose()


app = FastAPI(
    title="AI2Text Metadata API",
    version="1.0.0",
    description="Recording metadata management service",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    checks = {}
    
    # Check database
    try:
        async with SessionLocal() as session:
            await session.execute(select(1))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = "error"
    
    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        checks=checks
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics."""
    return PlainTextResponse(generate_latest().decode())


@app.post("/recordings", response_model=RecordingResponse, status_code=201, tags=["recordings"])
async def create_recording(request: CreateRecordingRequest) -> RecordingResponse:
    """Create recording metadata."""
    db_operations_total.labels(operation="create").inc()
    
    recording_id = uuid4()
    now = datetime.utcnow()
    
    with db_duration_seconds.labels(operation="create").time():
        async with SessionLocal() as session:
            recording = Recording(
                recording_id=recording_id,
                status=RecordingStatus.UPLOADED.value,
                audio_url=request.audio_url,
                language=request.language,
                duration_sec=request.duration_sec,
                created_at=now,
                updated_at=now
            )
            session.add(recording)
            await session.commit()
            await session.refresh(recording)
    
    return RecordingResponse(
        recording_id=str(recording.recording_id),
        status=recording.status,
        audio_url=recording.audio_url,
        transcript=recording.transcript,
        language=recording.language,
        duration_sec=recording.duration_sec,
        error=recording.error,
        created_at=recording.created_at,
        updated_at=recording.updated_at
    )


@app.get("/recordings", response_model=RecordingListResponse, tags=["recordings"])
async def list_recordings(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: RecordingStatus | None = None
) -> RecordingListResponse:
    """List recordings."""
    db_operations_total.labels(operation="list").inc()
    
    with db_duration_seconds.labels(operation="list").time():
        async with SessionLocal() as session:
            query = select(Recording).offset(offset).limit(limit)
            if status:
                query = query.where(Recording.status == status.value)
            
            result = await session.execute(query)
            recordings = result.scalars().all()
    
    return RecordingListResponse(
        recordings=[
            RecordingResponse(
                recording_id=str(r.recording_id),
                status=r.status,
                audio_url=r.audio_url,
                transcript=r.transcript,
                language=r.language,
                duration_sec=r.duration_sec,
                error=r.error,
                created_at=r.created_at,
                updated_at=r.updated_at
            )
            for r in recordings
        ],
        total=len(recordings),
        limit=limit,
        offset=offset
    )


@app.get("/recordings/{recording_id}", response_model=RecordingResponse, tags=["recordings"])
async def get_recording(recording_id: UUID) -> RecordingResponse:
    """Get recording metadata."""
    db_operations_total.labels(operation="get").inc()
    
    with db_duration_seconds.labels(operation="get").time():
        async with SessionLocal() as session:
            result = await session.execute(
                select(Recording).where(Recording.recording_id == recording_id)
            )
            recording = result.scalar_one_or_none()
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return RecordingResponse(
        recording_id=str(recording.recording_id),
        status=recording.status,
        audio_url=recording.audio_url,
        transcript=recording.transcript,
        language=recording.language,
        duration_sec=recording.duration_sec,
        error=recording.error,
        created_at=recording.created_at,
        updated_at=recording.updated_at
    )


@app.patch("/recordings/{recording_id}", response_model=RecordingResponse, tags=["recordings"])
async def update_recording(
    recording_id: UUID,
    request: UpdateRecordingRequest
) -> RecordingResponse:
    """Update recording metadata."""
    db_operations_total.labels(operation="update").inc()
    
    with db_duration_seconds.labels(operation="update").time():
        async with SessionLocal() as session:
            result = await session.execute(
                select(Recording).where(Recording.recording_id == recording_id)
            )
            recording = result.scalar_one_or_none()
            
            if not recording:
                raise HTTPException(status_code=404, detail="Recording not found")
            
            if request.status:
                recording.status = request.status.value
            if request.transcript:
                recording.transcript = request.transcript
            if request.error:
                recording.error = request.error
            
            recording.updated_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(recording)
    
    return RecordingResponse(
        recording_id=str(recording.recording_id),
        status=recording.status,
        audio_url=recording.audio_url,
        transcript=recording.transcript,
        language=recording.language,
        duration_sec=recording.duration_sec,
        error=recording.error,
        created_at=recording.created_at,
        updated_at=recording.updated_at
    )


@app.delete("/recordings/{recording_id}", status_code=204, tags=["recordings"])
async def delete_recording(recording_id: UUID):
    """Delete recording."""
    db_operations_total.labels(operation="delete").inc()
    
    with db_duration_seconds.labels(operation="delete").time():
        async with SessionLocal() as session:
            result = await session.execute(
                select(Recording).where(Recording.recording_id == recording_id)
            )
            recording = result.scalar_one_or_none()
            
            if not recording:
                raise HTTPException(status_code=404, detail="Recording not found")
            
            await session.delete(recording)
            await session.commit()
    
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

