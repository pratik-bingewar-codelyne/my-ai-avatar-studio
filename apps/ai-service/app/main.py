"""FastAPI application entrypoint for MyAI Avatar Studio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.api.routes import avatars, health, system
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.media_service import MediaService


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(
        "Starting %s v%s (%s)",
        settings.app_name,
        settings.app_version or __version__,
        settings.environment,
    )
    logger.info(
        "FFmpeg available=%s ffprobe available=%s",
        MediaService.ffmpeg_available(),
        MediaService.ffprobe_available(),
    )
    if not MediaService.ffmpeg_available():
        logger.warning(
            "FFmpeg is not on PATH. The API will start, but media operations "
            "that require FFmpeg/ffprobe will report clear errors instead of "
            "fabricating metadata."
        )
    # Ensure storage directories exist; do not touch the GPU here.
    settings.raw_media_dir.mkdir(parents=True, exist_ok=True)
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version or __version__,
        description="Personal AI Avatar Video Generation — Phase 1 backend",
        lifespan=lifespan,
    )
    application.include_router(health.router)
    application.include_router(system.router)
    application.include_router(avatars.router)
    return application


app = create_app()
