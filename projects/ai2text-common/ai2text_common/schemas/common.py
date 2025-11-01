"""
Common API schemas
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status", examples=["healthy", "unhealthy"])
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: Optional[str] = Field(None, description="Service version")
    details: Optional[dict] = Field(None, description="Additional health details")


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    code: Optional[int] = Field(None, description="HTTP status code")
    details: Optional[dict] = Field(None, description="Additional error details")


