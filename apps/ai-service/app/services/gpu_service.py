"""GPU introspection via the installed PyTorch runtime.

No models are downloaded or loaded. No heavy GPU work runs at import time.
"""

from __future__ import annotations

import logging

from app.models.schemas import GpuInfoResponse

logger = logging.getLogger(__name__)


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


class GpuService:
    """Query live CUDA / VRAM state through PyTorch."""

    def get_gpu_info(self) -> GpuInfoResponse:
        try:
            import torch
        except ImportError as exc:
            logger.error("PyTorch is not installed: %s", exc)
            return GpuInfoResponse(
                pytorch_version="unavailable",
                cuda_available=False,
                device_count=0,
            )

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_version = getattr(torch.version, "cuda", None)

        if not cuda_available or device_count < 1:
            return GpuInfoResponse(
                pytorch_version=torch.__version__,
                cuda_available=False,
                device_count=device_count,
                cuda_version=cuda_version,
            )

        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        allocated = int(torch.cuda.memory_allocated(0))

        free_vram_gb: float | None = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_vram_gb = _bytes_to_gb(free_bytes)
            # Prefer driver-reported total when available
            total = int(total_bytes)
        except Exception as exc:  # noqa: BLE001 — mem_get_info may be unavailable
            logger.info(
                "torch.cuda.mem_get_info unavailable; free VRAM omitted: %s",
                exc,
            )
            free_vram_gb = _bytes_to_gb(max(total - allocated, 0))

        return GpuInfoResponse(
            pytorch_version=torch.__version__,
            cuda_available=True,
            gpu_name=torch.cuda.get_device_name(0),
            total_vram_gb=_bytes_to_gb(total),
            allocated_vram_gb=_bytes_to_gb(allocated),
            free_vram_gb=free_vram_gb,
            device_count=device_count,
            cuda_version=cuda_version,
        )
