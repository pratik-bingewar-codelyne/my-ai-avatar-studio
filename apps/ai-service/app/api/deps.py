"""FastAPI dependency injection helpers."""

from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.avatar_service import AvatarService
from app.services.gpu_service import GpuService
from app.services.media_service import MediaService


@lru_cache
def get_gpu_service() -> GpuService:
    return GpuService()


def get_media_service(
    settings: Settings = Depends(get_settings),
) -> MediaService:
    return MediaService(settings)


def get_avatar_service(
    settings: Settings = Depends(get_settings),
    media_service: MediaService = Depends(get_media_service),
) -> AvatarService:
    return AvatarService(settings, media_service)
