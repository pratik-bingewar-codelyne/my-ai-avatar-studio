"""Health endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["app_name"]
    assert "ffmpeg_available" in payload
    assert "ffprobe_available" in payload
    assert isinstance(payload["ffmpeg_available"], bool)
    assert isinstance(payload["ffprobe_available"], bool)
