"""Shared pytest fixtures for the AI service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the ai-service package root is importable
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api.deps import get_avatar_service, get_gpu_service  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.avatar_service import AvatarService  # noqa: E402
from app.services.gpu_service import GpuService  # noqa: E402
from app.services.media_service import MediaService  # noqa: E402


@pytest.fixture()
def temp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated settings pointing at a temporary data directory."""
    get_settings.cache_clear()
    data_dir = tmp_path / "data"
    settings = Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        raw_media_dir=data_dir / "raw",
        avatars_dir=data_dir / "avatars",
        processed_dir=data_dir / "processed",
        environment="test",
        max_video_upload_bytes=5 * 1024 * 1024,
        max_audio_upload_bytes=2 * 1024 * 1024,
    )
    settings.raw_media_dir.mkdir(parents=True, exist_ok=True)
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("ENVIRONMENT", "test")

    def _override_settings() -> Settings:
        return settings

    # Patch the cached getter used by the app
    monkeypatch.setattr("app.core.config.get_settings", _override_settings)
    monkeypatch.setattr("app.api.deps.get_settings", _override_settings)
    monkeypatch.setattr("app.main.get_settings", _override_settings)
    return settings


@pytest.fixture()
def client(temp_settings: Settings) -> TestClient:
    application = create_app()

    def _settings() -> Settings:
        return temp_settings

    def _avatar_service() -> AvatarService:
        return AvatarService(temp_settings, MediaService(temp_settings))

    application.dependency_overrides[get_settings] = _settings
    application.dependency_overrides[get_avatar_service] = _avatar_service
    application.dependency_overrides[get_gpu_service] = lambda: GpuService()

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
    get_settings.cache_clear()
