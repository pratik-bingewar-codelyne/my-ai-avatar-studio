"""Avatar profile and media upload endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_avatar_service
from app.core.avatar_id import InvalidAvatarIdError
from app.models.schemas import (
    AvatarCreateRequest,
    AvatarCreateResponse,
    AvatarListResponse,
    DatasetStatusResponse,
    ErrorResponse,
    UploadResponse,
)
from app.services.avatar_service import AvatarNotFoundError, AvatarService
from app.services.media_service import MediaValidationError

router = APIRouter(prefix="/api/v1/avatars", tags=["avatars"])


def _map_avatar_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidAvatarIdError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    if isinstance(exc, AvatarNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    if isinstance(exc, MediaValidationError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    raise exc


@router.post(
    "",
    response_model=AvatarCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
def create_avatar(
    payload: AvatarCreateRequest,
    avatar_service: AvatarService = Depends(get_avatar_service),
) -> AvatarCreateResponse:
    """Create a new avatar profile and empty dataset manifest."""
    return avatar_service.create_avatar(payload.name)


@router.get("", response_model=AvatarListResponse)
def list_avatars(
    avatar_service: AvatarService = Depends(get_avatar_service),
) -> AvatarListResponse:
    """Return all registered avatar profiles."""
    avatars = avatar_service.list_avatars()
    return AvatarListResponse(avatars=avatars, total=len(avatars))


@router.get(
    "/{avatar_id}/dataset-status",
    response_model=DatasetStatusResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def dataset_status(
    avatar_id: str,
    avatar_service: AvatarService = Depends(get_avatar_service),
) -> DatasetStatusResponse:
    """Return actual dataset information and validation status."""
    try:
        return avatar_service.get_dataset_status(avatar_id)
    except (InvalidAvatarIdError, AvatarNotFoundError) as exc:
        raise _map_avatar_errors(exc) from exc


@router.post(
    "/{avatar_id}/upload-video",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_video(
    avatar_id: str,
    file: UploadFile = File(...),
    avatar_service: AvatarService = Depends(get_avatar_service),
) -> UploadResponse:
    """Upload and preserve an original video recording for an avatar."""
    try:
        return await avatar_service.upload_video(avatar_id, file)
    except (InvalidAvatarIdError, AvatarNotFoundError, MediaValidationError) as exc:
        raise _map_avatar_errors(exc) from exc


@router.post(
    "/{avatar_id}/upload-audio",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def upload_audio(
    avatar_id: str,
    file: UploadFile = File(...),
    avatar_service: AvatarService = Depends(get_avatar_service),
) -> UploadResponse:
    """Upload and preserve an original audio recording for an avatar."""
    try:
        return await avatar_service.upload_audio(avatar_id, file)
    except (InvalidAvatarIdError, AvatarNotFoundError, MediaValidationError) as exc:
        raise _map_avatar_errors(exc) from exc
