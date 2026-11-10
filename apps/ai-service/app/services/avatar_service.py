"""Avatar profile and dataset manifest management.

Concurrency notes
-----------------
Per-avatar ``threading.Lock`` instances serialize manifest read-modify-write
and the matching media persist/rollback within a **single process**.

Limitation: these locks are in-process only. If multiple Uvicorn worker
processes (or multiple OS processes) are introduced later, they will not share
locks and can still race on the same manifest file. Use a cross-process lock
(file lock / external store) before scaling beyond one worker.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core.avatar_id import InvalidAvatarIdError, validate_avatar_id
from app.core.config import Settings
from app.models.schemas import (
    AvatarCreateResponse,
    AvatarProfile,
    DatasetManifest,
    DatasetStatus,
    DatasetStatusResponse,
    MediaFileInfo,
    UploadResponse,
    ValidationStatus,
)
from app.services.media_service import MediaService, MediaValidationError

logger = logging.getLogger(__name__)

# Module-level registry so locks are shared across AvatarService instances
# created per-request by FastAPI dependency injection (single process).
_LOCKS_GUARD = threading.Lock()
_AVATAR_LOCKS: dict[str, threading.Lock] = {}


class AvatarNotFoundError(Exception):
    """Raised when an avatar ID does not exist."""


class AvatarService:
    """CRUD for avatar profiles backed by per-avatar JSON manifests."""

    def __init__(self, settings: Settings, media_service: MediaService) -> None:
        self.settings = settings
        self.media_service = media_service
        self.settings.avatars_dir.mkdir(parents=True, exist_ok=True)

    def _lock_for(self, avatar_id: str) -> threading.Lock:
        with _LOCKS_GUARD:
            lock = _AVATAR_LOCKS.get(avatar_id)
            if lock is None:
                lock = threading.Lock()
                _AVATAR_LOCKS[avatar_id] = lock
            return lock

    def _manifest_path(self, avatar_id: str) -> Path:
        validate_avatar_id(avatar_id)
        return self.settings.avatars_dir / avatar_id / "manifest.json"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _write_manifest(self, manifest: DatasetManifest) -> None:
        """Atomically replace manifest.json via temp file + os.replace."""
        validate_avatar_id(manifest.avatar_id)
        avatar_dir = self.settings.avatars_dir / manifest.avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._manifest_path(manifest.avatar_id)
        tmp_path = avatar_dir / f".manifest_{uuid.uuid4().hex}.tmp"
        try:
            tmp_path.write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _read_manifest(self, avatar_id: str) -> DatasetManifest:
        path = self._manifest_path(avatar_id)
        if not path.is_file():
            raise AvatarNotFoundError(f"Avatar '{avatar_id}' not found")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(raw)

    @staticmethod
    def _file_has_required_video_metadata(media: MediaFileInfo) -> bool:
        return (
            media.validation_status == ValidationStatus.VALID
            and media.width is not None
            and media.width > 0
            and media.height is not None
            and media.height > 0
            and media.frame_count is not None
            and media.frame_count > 0
        )

    @staticmethod
    def _file_has_required_audio_metadata(media: MediaFileInfo) -> bool:
        return (
            media.validation_status == ValidationStatus.VALID
            and media.sample_rate is not None
            and media.sample_rate > 0
            and media.duration_seconds is not None
            and media.duration_seconds > 0
        )

    def _recompute_status(self, manifest: DatasetManifest) -> DatasetManifest:
        """Deterministic dataset status from per-file validation outcomes."""
        videos = manifest.original_video_files
        audios = manifest.original_audio_files
        notes: list[str] = []

        all_files = videos + audios
        if not all_files:
            manifest.dataset_status = DatasetStatus.EMPTY
            manifest.validation_status = ValidationStatus.PENDING
            manifest.validation_notes = ["No media uploaded yet"]
            manifest.updated_at = self._now()
            return manifest

        has_invalid = any(
            f.validation_status == ValidationStatus.INVALID for f in all_files
        )
        has_pending = any(
            f.validation_status == ValidationStatus.PENDING for f in all_files
        )
        valid_videos = [v for v in videos if self._file_has_required_video_metadata(v)]
        valid_audios = [a for a in audios if self._file_has_required_audio_metadata(a)]

        if has_invalid:
            dataset_status = DatasetStatus.INVALID
            validation_status = ValidationStatus.INVALID
            notes.append("One or more media files failed content validation")
            for media in all_files:
                if media.validation_status == ValidationStatus.INVALID:
                    notes.append(f"Invalid media: {media.filename}")
        elif not videos or not audios:
            dataset_status = DatasetStatus.INCOMPLETE
            validation_status = ValidationStatus.INCOMPLETE
            if not videos:
                notes.append("Missing video recording")
            if not audios:
                notes.append("Missing audio recording")
        elif has_pending or not valid_videos or not valid_audios:
            # Both modalities present, but PENDING must never count as VALID.
            dataset_status = DatasetStatus.VALIDATING
            validation_status = ValidationStatus.VALIDATING
            if has_pending:
                notes.append(
                    "One or more files are PENDING validation (tooling unavailable "
                    "or incomplete probe); PENDING is not treated as VALID"
                )
            if not valid_videos:
                notes.append("No successfully validated video with required metadata")
            if not valid_audios:
                notes.append("No successfully validated audio with required metadata")
        else:
            dataset_status = DatasetStatus.READY
            validation_status = ValidationStatus.VALID
            notes.append(
                "At least one validated video and one validated audio with "
                "required metadata; no PENDING or INVALID files"
            )

        manifest.dataset_status = dataset_status
        manifest.validation_status = validation_status
        manifest.validation_notes = notes
        manifest.updated_at = self._now()
        return manifest

    def create_avatar(self, name: str) -> AvatarCreateResponse:
        avatar_id = uuid.uuid4().hex
        now = self._now()
        manifest = DatasetManifest(
            avatar_id=avatar_id,
            name=name,
            created_at=now,
            updated_at=now,
            dataset_status=DatasetStatus.EMPTY,
            validation_status=ValidationStatus.PENDING,
            validation_notes=["No media uploaded yet"],
        )
        with self._lock_for(avatar_id):
            self._write_manifest(manifest)
            self.media_service.avatar_raw_dir(avatar_id)
        logger.info("Created avatar %s (%s)", avatar_id, name)
        return AvatarCreateResponse(
            avatar_id=avatar_id,
            name=name,
            created_at=now,
            dataset_status=DatasetStatus.EMPTY,
        )

    def list_avatars(self) -> list[AvatarProfile]:
        profiles: list[AvatarProfile] = []
        if not self.settings.avatars_dir.exists():
            return profiles

        for child in sorted(self.settings.avatars_dir.iterdir()):
            if not child.is_dir():
                continue
            # Skip directories that are not valid avatar IDs
            try:
                validate_avatar_id(child.name)
            except InvalidAvatarIdError:
                continue
            manifest_path = child / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = DatasetManifest.model_validate(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.warning(
                    "Skipping corrupt manifest for avatar_id=%s: %s",
                    child.name,
                    exc,
                )
                continue
            profiles.append(
                AvatarProfile(
                    avatar_id=manifest.avatar_id,
                    name=manifest.name,
                    created_at=manifest.created_at,
                    dataset_status=manifest.dataset_status,
                    video_count=len(manifest.original_video_files),
                    audio_count=len(manifest.original_audio_files),
                )
            )
        return profiles

    def get_dataset_status(self, avatar_id: str) -> DatasetStatusResponse:
        validate_avatar_id(avatar_id)
        with self._lock_for(avatar_id):
            manifest = self._read_manifest(avatar_id)
        return DatasetStatusResponse(
            avatar_id=manifest.avatar_id,
            name=manifest.name,
            dataset_status=manifest.dataset_status,
            validation_status=manifest.validation_status,
            video_files=manifest.original_video_files,
            audio_files=manifest.original_audio_files,
            validation_notes=manifest.validation_notes,
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
        )

    def get_manifest(self, avatar_id: str) -> DatasetManifest:
        validate_avatar_id(avatar_id)
        with self._lock_for(avatar_id):
            return self._read_manifest(avatar_id)

    async def upload_video(
        self, avatar_id: str, upload: UploadFile
    ) -> UploadResponse:
        validate_avatar_id(avatar_id)
        with self._lock_for(avatar_id):
            # Ensure avatar exists before accepting bytes.
            self._read_manifest(avatar_id)
            stored = None
            try:
                stored = await self.media_service.save_video(avatar_id, upload)
                manifest = self._read_manifest(avatar_id)
                manifest.original_video_files.append(stored.info)
                manifest = self._recompute_status(manifest)
                self._write_manifest(manifest)
            except Exception:
                if stored is not None and stored.absolute_path.exists():
                    stored.absolute_path.unlink(missing_ok=True)
                    logger.warning(
                        "Rolled back new video after manifest failure: %s",
                        stored.info.filename,
                    )
                raise

            return UploadResponse(
                avatar_id=avatar_id,
                media_type="video",
                file=stored.info,
                dataset_status=manifest.dataset_status,
                validation_status=manifest.validation_status,
                message="Video uploaded and original recording preserved",
            )

    async def upload_audio(
        self, avatar_id: str, upload: UploadFile
    ) -> UploadResponse:
        validate_avatar_id(avatar_id)
        with self._lock_for(avatar_id):
            self._read_manifest(avatar_id)
            stored = None
            try:
                stored = await self.media_service.save_audio(avatar_id, upload)
                manifest = self._read_manifest(avatar_id)
                manifest.original_audio_files.append(stored.info)
                manifest = self._recompute_status(manifest)
                self._write_manifest(manifest)
            except Exception:
                if stored is not None and stored.absolute_path.exists():
                    stored.absolute_path.unlink(missing_ok=True)
                    logger.warning(
                        "Rolled back new audio after manifest failure: %s",
                        stored.info.filename,
                    )
                raise

            return UploadResponse(
                avatar_id=avatar_id,
                media_type="audio",
                file=stored.info,
                dataset_status=manifest.dataset_status,
                validation_status=manifest.validation_status,
                message="Audio uploaded and original recording preserved",
            )
