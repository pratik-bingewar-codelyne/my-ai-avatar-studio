"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Resolve repository root from this file location."""
    return Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MyAI Avatar Studio"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Storage roots (relative to project root unless absolute)
    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path | None = None
    raw_media_dir: Path | None = None
    avatars_dir: Path | None = None
    processed_dir: Path | None = None

    # Upload limits
    max_video_upload_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_audio_upload_bytes: int = 100 * 1024 * 1024  # 100 MB

    # Allowed media types
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    )
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    )
    # Client MIME is never proof of content; paired with extension + decode checks.
    # application/octet-stream is intentionally excluded (Phase 1.1 MIME policy).
    allowed_video_mime_types: list[str] = Field(
        default_factory=lambda: [
            "video/mp4",
            "video/quicktime",
            "video/x-msvideo",
            "video/x-matroska",
            "video/webm",
            "video/avi",
        ]
    )
    allowed_audio_mime_types: list[str] = Field(
        default_factory=lambda: [
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/mp4",
            "audio/m4a",
            "audio/x-m4a",
            "audio/flac",
            "audio/ogg",
            "audio/x-flac",
        ]
    )
    require_upload_content_type: bool = True

    host: str = "127.0.0.1"
    port: int = 8000

    def model_post_init(self, __context: object) -> None:
        root = self.project_root
        if self.data_dir is None:
            self.data_dir = root / "data"
        if self.raw_media_dir is None:
            self.raw_media_dir = self.data_dir / "raw"
        if self.avatars_dir is None:
            self.avatars_dir = self.data_dir / "avatars"
        if self.processed_dir is None:
            self.processed_dir = self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for dependency injection."""
    settings = Settings()
    settings.raw_media_dir.mkdir(parents=True, exist_ok=True)
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    return settings
