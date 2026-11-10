"""GPU endpoint tests."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient


def test_gpu_endpoint_returns_real_pytorch_info(client: TestClient) -> None:
    response = client.get("/api/v1/system/gpu")
    assert response.status_code == 200
    payload = response.json()

    assert payload["pytorch_version"] == torch.__version__
    assert payload["cuda_available"] is torch.cuda.is_available()

    if torch.cuda.is_available():
        assert payload["gpu_name"] == torch.cuda.get_device_name(0)
        assert payload["total_vram_gb"] is not None
        assert payload["total_vram_gb"] > 0
        assert payload["allocated_vram_gb"] is not None
        assert payload["device_count"] >= 1
    else:
        assert payload["gpu_name"] is None
