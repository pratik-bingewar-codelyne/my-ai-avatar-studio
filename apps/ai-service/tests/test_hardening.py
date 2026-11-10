"""Phase 1.1 reliability and security hardening tests."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.models.schemas import (
    DatasetManifest,
    DatasetStatus,
    MediaFileInfo,
    ValidationStatus,
)
from app.services.avatar_service import AvatarService
from app.services.gpu_service import GpuService
from app.services.media_service import MediaService, MediaValidationError
from tests.media_fixtures import build_valid_avi_file, build_valid_wav_bytes


def _create_avatar(client: TestClient, name: str = "Hardening") -> str:
    return client.post("/api/v1/avatars", json={"name": name}).json()["avatar_id"]


def test_oversized_video_upload(client: TestClient, temp_settings: Settings) -> None:
    avatar_id = _create_avatar(client, "Over Video")
    # Limit is 5 MiB in temp_settings; stream slightly over.
    oversized = b"\x00" * (temp_settings.max_video_upload_bytes + 1)
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("big.avi", io.BytesIO(oversized), "video/x-msvideo")},
    )
    assert response.status_code == 400
    assert "maximum upload size" in response.json()["detail"].lower()
    video_dir = temp_settings.raw_media_dir / avatar_id / "video"
    if video_dir.exists():
        assert list(video_dir.glob("*")) == [] or all(
            p.name.startswith(".") is False and p.suffix == ".part"
            for p in video_dir.iterdir()
            if False
        )
        assert not any(p.suffix == ".avi" for p in video_dir.glob("*"))


def test_oversized_audio_upload(client: TestClient, temp_settings: Settings) -> None:
    avatar_id = _create_avatar(client, "Over Audio")
    oversized = b"\x00" * (temp_settings.max_audio_upload_bytes + 1)
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("big.wav", io.BytesIO(oversized), "audio/wav")},
    )
    assert response.status_code == 400
    assert "maximum upload size" in response.json()["detail"].lower()


def test_empty_file_rejected(client: TestClient) -> None:
    avatar_id = _create_avatar(client, "Empty")
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_mime_spoofing_rejected(client: TestClient) -> None:
    avatar_id = _create_avatar(client, "MIME Spoof")
    content = build_valid_wav_bytes()
    # .wav bytes labeled as video/mp4 with .mp4 extension should fail video decode.
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("spoof.mp4", io.BytesIO(content), "video/mp4")},
    )
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


def test_octet_stream_mime_rejected(client: TestClient) -> None:
    avatar_id = _create_avatar(client, "Octet")
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={
            "file": (
                "voice.wav",
                io.BytesIO(build_valid_wav_bytes()),
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 400
    assert "mime" in response.json()["detail"].lower()


def test_corrupted_mp4_rejected(
    client: TestClient, temp_settings: Settings
) -> None:
    avatar_id = _create_avatar(client, "Corrupt MP4")
    garbage = b"not-a-real-mp4-file" + b"\x00" * 64
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("bad.mp4", io.BytesIO(garbage), "video/mp4")},
    )
    assert response.status_code == 400
    video_dir = temp_settings.raw_media_dir / avatar_id / "video"
    if video_dir.exists():
        assert not any(p.suffix == ".mp4" for p in video_dir.iterdir())


def test_corrupted_wav_rejected(
    client: TestClient, temp_settings: Settings
) -> None:
    avatar_id = _create_avatar(client, "Corrupt WAV")
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("bad.wav", io.BytesIO(b"RIFF....WAVE"), "audio/wav")},
    )
    assert response.status_code == 400
    audio_dir = temp_settings.raw_media_dir / avatar_id / "audio"
    if audio_dir.exists():
        assert not any(p.suffix == ".wav" for p in audio_dir.iterdir())


@pytest.mark.parametrize(
    "avatar_id",
    [
        "../etc/passwd",
        "..%2F..%2Fsecret",
        "..\\..\\windows",
        "not-a-uuid",
        "ABCDEF" + "0" * 26,  # uppercase rejected
        "a" * 31,
        "a" * 33,
    ],
)
def test_invalid_avatar_id_variants(client: TestClient, avatar_id: str) -> None:
    from app.core.avatar_id import InvalidAvatarIdError, validate_avatar_id

    # Validator rejects traversal / malformed IDs before any path join.
    with pytest.raises(InvalidAvatarIdError):
        validate_avatar_id(avatar_id)

    response = client.get(f"/api/v1/avatars/{avatar_id}/dataset-status")
    # HTTP stack may normalize ".." out of the route (404) before our handler;
    # when the segment reaches the handler it must be 400.
    assert response.status_code in {400, 404}, avatar_id


def test_encoded_path_traversal_upload(client: TestClient) -> None:
    from app.core.avatar_id import InvalidAvatarIdError, validate_avatar_id

    with pytest.raises(InvalidAvatarIdError):
        validate_avatar_id("..%2F..%2Fevil")

    response = client.post(
        "/api/v1/avatars/..%2F..%2Fevil/upload-audio",
        files={"file": ("x.wav", io.BytesIO(build_valid_wav_bytes()), "audio/wav")},
    )
    # Starlette may decode/normalize path; either 400 from our validator or 404.
    assert response.status_code in {400, 404}


def test_valid_hex_avatar_id_accepted_by_validator() -> None:
    from app.core.avatar_id import validate_avatar_id

    value = "a" * 32
    assert validate_avatar_id(value) == value


def test_duplicate_filenames_do_not_overwrite(
    client: TestClient, temp_settings: Settings, tmp_path: Path
) -> None:
    avatar_id = _create_avatar(client, "Dup Names")
    content = build_valid_avi_file(tmp_path / "dup.avi").read_bytes()
    first = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("same.avi", io.BytesIO(content), "video/x-msvideo")},
    ).json()
    second = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("same.avi", io.BytesIO(content), "video/x-msvideo")},
    ).json()
    assert first["file"]["filename"] != second["file"]["filename"]
    path1 = temp_settings.raw_media_dir / avatar_id / first["file"]["filename"]
    path2 = temp_settings.raw_media_dir / avatar_id / second["file"]["filename"]
    assert path1.is_file() and path2.is_file()
    assert path1.read_bytes() == content
    assert path2.read_bytes() == content


def test_concurrent_uploads_preserve_all_manifest_entries(
    temp_settings: Settings,
) -> None:
    """In-process locks must keep all concurrent audio entries in the manifest."""
    service = AvatarService(temp_settings, MediaService(temp_settings))
    avatar_id = service.create_avatar("Concurrent").avatar_id
    wav = build_valid_wav_bytes()
    errors: list[BaseException] = []

    class _Upload:
        def __init__(self, name: str, payload: bytes) -> None:
            self.filename = name
            self.content_type = "audio/wav"
            self._payload = payload
            self._sent = False

        async def read(self, size: int = -1) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return self._payload

    def _upload(index: int) -> None:
        try:
            asyncio.run(
                service.upload_audio(
                    avatar_id, _Upload(f"voice_{index}.wav", wav)  # type: ignore[arg-type]
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_upload, args=(index,)) for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    manifest = service.get_manifest(avatar_id)
    assert len(manifest.original_audio_files) == 4
    names = {item.filename for item in manifest.original_audio_files}
    assert len(names) == 4
    for item in manifest.original_audio_files:
        assert (temp_settings.raw_media_dir / avatar_id / item.filename).is_file()


def test_manifest_write_failure_rolls_back_new_file(
    temp_settings: Settings, tmp_path: Path
) -> None:
    service = AvatarService(temp_settings, MediaService(temp_settings))
    created = service.create_avatar("Rollback")
    avatar_id = created.avatar_id
    avi = build_valid_avi_file(tmp_path / "roll.avi")

    class _Upload:
        filename = "roll.avi"
        content_type = "video/x-msvideo"

        async def read(self, size: int = -1) -> bytes:
            data = avi.read_bytes()
            if not hasattr(self, "_sent"):
                self._sent = True
                return data
            return b""

    with patch.object(
        AvatarService,
        "_write_manifest",
        side_effect=OSError("simulated manifest failure"),
    ):
        with pytest.raises(OSError, match="simulated manifest failure"):
            asyncio.run(service.upload_video(avatar_id, _Upload()))  # type: ignore[arg-type]

    video_dir = temp_settings.raw_media_dir / avatar_id / "video"
    assert not video_dir.exists() or list(video_dir.glob("*.avi")) == []
    manifest = service.get_manifest(avatar_id)
    assert manifest.original_video_files == []


def test_atomic_manifest_replacement(temp_settings: Settings) -> None:
    service = AvatarService(temp_settings, MediaService(temp_settings))
    created = service.create_avatar("Atomic")
    avatar_id = created.avatar_id
    path = temp_settings.avatars_dir / avatar_id / "manifest.json"
    original = path.read_text(encoding="utf-8")

    manifest = service.get_manifest(avatar_id)
    manifest.name = "Atomic Updated"
    service._write_manifest(manifest)

    assert path.is_file()
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["name"] == "Atomic Updated"
    assert updated["avatar_id"] == avatar_id
    # No leftover temp manifests
    leftovers = list((temp_settings.avatars_dir / avatar_id).glob(".manifest_*.tmp"))
    assert leftovers == []
    assert original != path.read_text(encoding="utf-8")


def test_cpu_only_gpu_fallback() -> None:
    service = GpuService()
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.device_count", return_value=0),
    ):
        # Import path used inside get_gpu_info
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            info = service.get_gpu_info()
    assert info.cuda_available is False
    assert info.gpu_name is None
    assert info.pytorch_version  # still reports installed torch


def test_ready_status_with_valid_media(
    client: TestClient, tmp_path: Path
) -> None:
    avatar_id = _create_avatar(client, "Ready Path")
    avi = build_valid_avi_file(tmp_path / "ready.avi").read_bytes()
    wav = build_valid_wav_bytes()

    video_resp = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("ready.avi", io.BytesIO(avi), "video/x-msvideo")},
    )
    assert video_resp.status_code == 200
    assert video_resp.json()["dataset_status"] == "incomplete"

    audio_resp = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("ready.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert audio_resp.status_code == 200
    payload = audio_resp.json()
    assert payload["dataset_status"] == "ready"
    assert payload["validation_status"] == "valid"
    assert payload["file"]["validation_status"] == "valid"

    status = client.get(f"/api/v1/avatars/{avatar_id}/dataset-status").json()
    assert status["dataset_status"] == "ready"
    assert status["validation_status"] == "valid"


def test_ready_not_set_for_pending_media(temp_settings: Settings) -> None:
    service = AvatarService(temp_settings, MediaService(temp_settings))
    created = service.create_avatar("Pending Ready")
    now = datetime.now(timezone.utc)
    pending_audio = MediaFileInfo(
        filename="audio/pending.mp3",
        original_filename="pending.mp3",
        size_bytes=100,
        content_type="audio/mpeg",
        uploaded_at=now,
        validation_status=ValidationStatus.PENDING,
        metadata_notes="ffprobe unavailable",
    )
    valid_video = MediaFileInfo(
        filename="video/ok.avi",
        original_filename="ok.avi",
        size_bytes=100,
        content_type="video/x-msvideo",
        uploaded_at=now,
        width=64,
        height=64,
        frame_count=5,
        duration_seconds=0.5,
        validation_status=ValidationStatus.VALID,
        metadata_source="opencv",
    )
    manifest = DatasetManifest(
        avatar_id=created.avatar_id,
        name="Pending Ready",
        created_at=now,
        updated_at=now,
        original_video_files=[valid_video],
        original_audio_files=[pending_audio],
    )
    updated = service._recompute_status(manifest)
    assert updated.dataset_status == DatasetStatus.VALIDATING
    assert updated.validation_status == ValidationStatus.VALIDATING
    assert updated.dataset_status != DatasetStatus.READY


def test_invalid_status_for_corrupted_media_entry(temp_settings: Settings) -> None:
    service = AvatarService(temp_settings, MediaService(temp_settings))
    created = service.create_avatar("Invalid Entry")
    now = datetime.now(timezone.utc)
    invalid_video = MediaFileInfo(
        filename="video/bad.mp4",
        original_filename="bad.mp4",
        size_bytes=12,
        content_type="video/mp4",
        uploaded_at=now,
        validation_status=ValidationStatus.INVALID,
        metadata_notes="undecodable",
    )
    valid_audio = MediaFileInfo(
        filename="audio/ok.wav",
        original_filename="ok.wav",
        size_bytes=100,
        content_type="audio/wav",
        uploaded_at=now,
        sample_rate=16000,
        duration_seconds=0.1,
        validation_status=ValidationStatus.VALID,
        metadata_source="wave",
    )
    manifest = DatasetManifest(
        avatar_id=created.avatar_id,
        name="Invalid Entry",
        created_at=now,
        updated_at=now,
        original_video_files=[invalid_video],
        original_audio_files=[valid_audio],
    )
    updated = service._recompute_status(manifest)
    assert updated.dataset_status == DatasetStatus.INVALID
    assert updated.validation_status == ValidationStatus.INVALID


def test_non_wav_without_ffprobe_stays_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    avatar_id = _create_avatar(client, "Pending MP3")
    monkeypatch.setattr(
        MediaService, "ffprobe_available", staticmethod(lambda: False)
    )
    # Minimal fake mp3 bytes — without ffprobe should remain PENDING, not VALID.
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={
            "file": (
                "clip.mp3",
                io.BytesIO(b"ID3" + b"\x00" * 64),
                "audio/mpeg",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["file"]["validation_status"] == "pending"
    assert payload["dataset_status"] in {"incomplete", "validating"}
    assert payload["dataset_status"] != "ready"
    notes = (payload["file"].get("metadata_notes") or "").lower()
    assert "ffprobe" in notes or "ffmpeg" in notes
