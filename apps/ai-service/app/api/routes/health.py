"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import Settings, get_settings
from app.models.schemas import HealthResponse
from app.services.media_service import MediaService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return live application health and version information."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version or __version__,
        environment=settings.environment,
        ffmpeg_available=MediaService.ffmpeg_available(),
        ffprobe_available=MediaService.ffprobe_available(),
    )
