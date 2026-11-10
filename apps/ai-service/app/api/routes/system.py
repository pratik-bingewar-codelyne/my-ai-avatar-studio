"""System / hardware introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_gpu_service
from app.models.schemas import GpuInfoResponse
from app.services.gpu_service import GpuService

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/gpu", response_model=GpuInfoResponse)
def gpu_info(gpu_service: GpuService = Depends(get_gpu_service)) -> GpuInfoResponse:
    """Return actual PyTorch / CUDA / VRAM information."""
    return gpu_service.get_gpu_info()
