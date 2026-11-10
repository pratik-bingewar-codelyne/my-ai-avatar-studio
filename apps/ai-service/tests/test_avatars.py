"""Avatar profile and dataset tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.core.config import Settings


def test_create_avatar(client: TestClient) -> None:
    response = client.post("/api/v1/avatars", json={"name": "Primary Self"})
    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "Primary Self"
    assert payload["avatar_id"]
    assert len(payload["avatar_id"]) == 32
    assert payload["dataset_status"] == "empty"
    assert payload["created_at"]


def test_list_avatars(client: TestClient) -> None:
    created = client.post("/api/v1/avatars", json={"name": "List Me"}).json()
    response = client.get("/api/v1/avatars")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 1
    ids = {item["avatar_id"] for item in payload["avatars"]}
    assert created["avatar_id"] in ids


def test_invalid_avatar_id_format_dataset_status(client: TestClient) -> None:
    response = client.get("/api/v1/avatars/does-not-exist/dataset-status")
    assert response.status_code == 400
    assert "invalid avatar_id" in response.json()["detail"].lower()


def test_valid_format_missing_avatar_returns_404(client: TestClient) -> None:
    missing = "a" * 32
    response = client.get(f"/api/v1/avatars/{missing}/dataset-status")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_dataset_manifest_generation(
    client: TestClient, temp_settings: Settings
) -> None:
    created = client.post("/api/v1/avatars", json={"name": "Manifest User"}).json()
    avatar_id = created["avatar_id"]

    status_response = client.get(f"/api/v1/avatars/{avatar_id}/dataset-status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["dataset_status"] == "empty"
    assert status_payload["validation_status"] == "pending"
    assert status_payload["video_files"] == []
    assert status_payload["audio_files"] == []

    manifest_path = temp_settings.avatars_dir / avatar_id / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["avatar_id"] == avatar_id
    assert manifest["name"] == "Manifest User"
    assert "original_video_files" in manifest
    assert "original_audio_files" in manifest
    assert manifest["validation_status"] == "pending"
