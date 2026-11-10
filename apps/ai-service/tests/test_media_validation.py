"""Upload validation and media handling tests."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from tests.media_fixtures import build_valid_avi_file, build_valid_wav_bytes


def _create_avatar(client: TestClient, name: str = "Upload User") -> str:
    return client.post("/api/v1/avatars", json={"name": name}).json()["avatar_id"]


def test_reject_invalid_video_extension(client: TestClient) -> None:
    avatar_id = _create_avatar(client)
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("notes.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert response.status_code == 400
    assert "extension" in response.json()["detail"].lower()


def test_reject_invalid_audio_extension(client: TestClient) -> None:
    avatar_id = _create_avatar(client, "Audio User")
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("clip.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "extension" in detail or "mime" in detail


def test_upload_missing_avatar_valid_id(client: TestClient) -> None:
    missing = "b" * 32
    response = client.post(
        f"/api/v1/avatars/{missing}/upload-video",
        files={"file": ("clip.avi", io.BytesIO(b"\x00\x00"), "video/x-msvideo")},
    )
    assert response.status_code == 404


def test_valid_video_upload_preserves_original(
    client: TestClient, temp_settings: Settings, tmp_path: Path
) -> None:
    avatar_id = _create_avatar(client, "Video Keep")
    fixture = build_valid_avi_file(tmp_path / "take_one.avi")
    content = fixture.read_bytes()
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("take_one.avi", io.BytesIO(content), "video/x-msvideo")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["media_type"] == "video"
    assert payload["file"]["original_filename"] == "take_one.avi"
    assert payload["file"]["size_bytes"] == len(content)
    assert payload["file"]["validation_status"] == "valid"
    assert payload["dataset_status"] in {"incomplete", "validating", "ready"}

    stored = temp_settings.raw_media_dir / avatar_id / payload["file"]["filename"]
    assert stored.is_file()
    assert stored.read_bytes() == content

    response2 = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-video",
        files={"file": ("take_one.avi", io.BytesIO(content), "video/x-msvideo")},
    )
    assert response2.status_code == 200
    stored2 = (
        temp_settings.raw_media_dir / avatar_id / response2.json()["file"]["filename"]
    )
    assert stored2 != stored
    assert stored.is_file()
    assert stored.read_bytes() == content


def test_valid_audio_upload_updates_manifest(
    client: TestClient, temp_settings: Settings
) -> None:
    avatar_id = _create_avatar(client, "Audio Keep")
    content = build_valid_wav_bytes()
    response = client.post(
        f"/api/v1/avatars/{avatar_id}/upload-audio",
        files={"file": ("voice.wav", io.BytesIO(content), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["media_type"] == "audio"
    assert payload["file"]["validation_status"] == "valid"
    assert payload["file"]["sample_rate"] == 16000
    assert payload["file"]["duration_seconds"] is not None

    status = client.get(f"/api/v1/avatars/{avatar_id}/dataset-status").json()
    assert len(status["audio_files"]) == 1
    stored = temp_settings.raw_media_dir / avatar_id / payload["file"]["filename"]
    assert Path(stored).is_file()
