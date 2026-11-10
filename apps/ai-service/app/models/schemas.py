"""API and domain schemas for avatar studio."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DatasetStatus(str, Enum):
    """Lifecycle status of an avatar dataset.

    Transitions (deterministic):
    - EMPTY: no media files registered.
    - INCOMPLETE: one media modality missing (video or audio).
    - VALIDATING: both modalities present, but one or more files are still
      PENDING validation (tooling unavailable / incomplete probe). PENDING is
      never treated as VALID.
    - READY: at least one VALID video and one VALID audio, required metadata
      present on those files, and no INVALID or PENDING files.
    - INVALID: one or more files failed content validation (INVALID).
    """

    EMPTY = "empty"
    INCOMPLETE = "incomplete"
    VALIDATING = "validating"
    READY = "ready"
    INVALID = "invalid"


class ValidationStatus(str, Enum):
    """Aggregate or per-file validation outcome."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    VALIDATING = "validating"


class HealthResponse(BaseModel):
    """Application health payload."""

    status: str
    app_name: str
    version: str
    environment: str
    ffmpeg_available: bool
    ffprobe_available: bool


class GpuInfoResponse(BaseModel):
    """Live GPU / CUDA status from PyTorch."""

    pytorch_version: str
    cuda_available: bool
    gpu_name: str | None = None
    total_vram_gb: float | None = None
    allocated_vram_gb: float | None = None
    free_vram_gb: float | None = None
    device_count: int = 0
    cuda_version: str | None = None


class AvatarCreateRequest(BaseModel):
    """Request body for creating an avatar profile."""

    name: str = Field(..., min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Avatar name must not be empty")
        return cleaned


class MediaFileInfo(BaseModel):
    """Metadata for an uploaded media file (paths are relative storage keys)."""

    filename: str
    original_filename: str
    size_bytes: int
    content_type: str | None = None
    uploaded_at: datetime
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    frame_count: int | None = None
    metadata_source: str | None = None
    metadata_notes: str | None = None
    validation_status: ValidationStatus = ValidationStatus.PENDING


class AvatarProfile(BaseModel):
    """Registered avatar profile summary."""

    avatar_id: str
    name: str
    created_at: datetime
    dataset_status: DatasetStatus
    video_count: int = 0
    audio_count: int = 0


class AvatarCreateResponse(BaseModel):
    """Response after creating an avatar."""

    avatar_id: str
    name: str
    created_at: datetime
    dataset_status: DatasetStatus


class AvatarListResponse(BaseModel):
    """List of registered avatars."""

    avatars: list[AvatarProfile]
    total: int


class DatasetManifest(BaseModel):
    """Per-avatar JSON dataset manifest (never stores media binaries)."""

    avatar_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    original_video_files: list[MediaFileInfo] = Field(default_factory=list)
    original_audio_files: list[MediaFileInfo] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    dataset_status: DatasetStatus = DatasetStatus.EMPTY
    validation_notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetStatusResponse(BaseModel):
    """Dataset status API response."""

    avatar_id: str
    name: str
    dataset_status: DatasetStatus
    validation_status: ValidationStatus
    video_files: list[MediaFileInfo]
    audio_files: list[MediaFileInfo]
    validation_notes: list[str]
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    """Response after a successful media upload."""

    avatar_id: str
    media_type: str
    file: MediaFileInfo
    dataset_status: DatasetStatus
    validation_status: ValidationStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str
