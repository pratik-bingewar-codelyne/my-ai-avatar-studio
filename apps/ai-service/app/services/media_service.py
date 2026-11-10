"""Media upload validation, streaming storage, and content inspection.

Client MIME and file extensions are advisory gates only. Decodability and
required metadata are established by OpenCV (video), the stdlib wave module
(WAV), and ffprobe when available (non-WAV audio). Values are never fabricated.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.models.schemas import MediaFileInfo, ValidationStatus

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK_SIZE = 1024 * 1024


class MediaValidationError(Exception):
    """Raised when an uploaded file fails validation."""


@dataclass
class StoredMedia:
    """Result of a successful media persist (absolute path for rollback)."""

    info: MediaFileInfo
    absolute_path: Path


class MediaService:
    """Handles safe media persistence and content-level validation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def ffprobe_available() -> bool:
        return shutil.which("ffprobe") is not None

    def avatar_raw_dir(self, avatar_id: str) -> Path:
        path = self.settings.raw_media_dir / avatar_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sanitize_stem(self, filename: str) -> str:
        stem = Path(filename).stem
        cleaned = _SAFE_NAME_RE.sub("_", stem).strip("._")
        return cleaned[:64] or "upload"

    def _unique_filename(self, original_filename: str, destination_dir: Path) -> str:
        ext = Path(original_filename).suffix.lower()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        token = uuid.uuid4().hex[:8]
        stem = self._sanitize_stem(original_filename)
        candidate = f"{stamp}_{token}_{stem}{ext}"
        while (destination_dir / candidate).exists():
            token = uuid.uuid4().hex[:8]
            candidate = f"{stamp}_{token}_{stem}{ext}"
        return candidate

    def _validate_extension_and_mime(
        self,
        filename: str,
        content_type: str | None,
        *,
        allowed_extensions: list[str],
        allowed_mimes: list[str],
        media_kind: str,
    ) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in allowed_extensions:
            raise MediaValidationError(
                f"Unsupported {media_kind} extension '{ext}'. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )

        if not content_type or not content_type.strip():
            if self.settings.require_upload_content_type:
                raise MediaValidationError(
                    f"Content-Type header is required for {media_kind} uploads. "
                    "Client MIME is checked together with extension and content "
                    "validation; it is never treated as proof of file contents."
                )
        else:
            mime = content_type.split(";")[0].strip().lower()
            if mime not in allowed_mimes:
                raise MediaValidationError(
                    f"Unsupported {media_kind} MIME type '{mime}'. "
                    f"Allowed: {', '.join(sorted(set(allowed_mimes)))}"
                )
        return ext

    async def _stream_to_temp(
        self,
        upload: UploadFile,
        max_bytes: int,
        media_kind: str,
        destination_dir: Path,
    ) -> tuple[Path, int]:
        """Stream upload chunks to a temp file with a running size counter."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = destination_dir / f".upload_{uuid.uuid4().hex}.part"
        total = 0
        try:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = await upload.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise MediaValidationError(
                            f"{media_kind.capitalize()} exceeds maximum upload size "
                            f"of {max_bytes} bytes"
                        )
                    handle.write(chunk)
            if total == 0:
                raise MediaValidationError(
                    f"{media_kind.capitalize()} file is empty"
                )
            return tmp_path, total
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _finalize_temp(self, tmp_path: Path, final_path: Path) -> None:
        """Atomically move a validated temp upload into its unique destination."""
        if final_path.exists():
            # Never overwrite an existing original recording.
            tmp_path.unlink(missing_ok=True)
            raise MediaValidationError(
                "Refusing to overwrite an existing media file; choose a new upload"
            )
        tmp_path.replace(final_path)

    def _assert_video_container_signature(
        self, named_path: Path, data_path: Path | None = None
    ) -> None:
        """Cheap magic-byte gate before OpenCV (avoids hangs on MIME spoofing)."""
        ext = named_path.suffix.lower()
        source = data_path or named_path
        header = source.read_bytes()[:64]
        if len(header) < 12:
            raise MediaValidationError(
                "Video file is too small to be a valid media container"
            )

        if ext in {".mp4", ".mov"}:
            # ISO BMFF: bytes 4..8 == 'ftyp'
            if header[4:8] != b"ftyp":
                raise MediaValidationError(
                    "Video content does not match MP4/MOV container signature "
                    "(ftyp box missing)"
                )
        elif ext == ".avi":
            if not (header[0:4] == b"RIFF" and header[8:12] == b"AVI "):
                raise MediaValidationError(
                    "Video content does not match AVI container signature"
                )
        elif ext == ".mkv":
            # EBML header
            if header[0:4] != b"\x1a\x45\xdf\xa3":
                raise MediaValidationError(
                    "Video content does not match Matroska/MKV signature"
                )
        elif ext == ".webm":
            if header[0:4] != b"\x1a\x45\xdf\xa3":
                raise MediaValidationError(
                    "Video content does not match WebM signature"
                )

    def _opencv_inspect(self, path: Path) -> dict[str, object]:
        """OpenCV probe executed in a worker thread with a hard timeout."""
        import threading

        outcome: dict[str, object] = {}

        def worker() -> None:
            try:
                import cv2

                capture = cv2.VideoCapture(str(path))
                try:
                    if not capture.isOpened():
                        outcome["error"] = MediaValidationError(
                            "Video file could not be opened or decoded by OpenCV "
                            "(corrupted or unsupported container)"
                        )
                        return

                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                    reported_frames = int(
                        capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    )
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        outcome["error"] = MediaValidationError(
                            "Video file opened but no frames could be decoded "
                            "(corrupted or empty video stream)"
                        )
                        return

                    if width <= 0 or height <= 0:
                        height, width = int(frame.shape[0]), int(frame.shape[1])
                    if width <= 0 or height <= 0:
                        outcome["error"] = MediaValidationError(
                            "Video frame dimensions are invalid"
                        )
                        return

                    frame_count = reported_frames if reported_frames > 0 else 1
                    duration: float | None = None
                    duration_note: str | None = None
                    if fps > 0 and reported_frames > 0:
                        duration = round(reported_frames / fps, 3)
                    else:
                        duration_note = (
                            "Duration could not be determined reliably from "
                            "OpenCV frame count/FPS; frame decode still succeeded"
                        )

                    outcome["data"] = {
                        "width": width,
                        "height": height,
                        "frame_count": frame_count,
                        "duration_seconds": duration,
                        "duration_note": duration_note,
                    }
                finally:
                    capture.release()
            except MediaValidationError as exc:
                outcome["error"] = exc
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = MediaValidationError(
                    f"OpenCV video validation failed: {exc}"
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=8.0)
        if thread.is_alive():
            raise MediaValidationError(
                "Video validation timed out while decoding with OpenCV "
                "(likely corrupted or unsupported content)"
            )
        if "error" in outcome:
            raise outcome["error"]  # type: ignore[misc]
        if "data" not in outcome:
            raise MediaValidationError("Video validation produced no result")
        return outcome["data"]  # type: ignore[return-value]

    def validate_video_content(
        self, path: Path, *, original_extension: str | None = None
    ) -> dict[str, object]:
        """Decode video with OpenCV. Reject undecodable / empty streams."""
        notes: list[str] = []
        result: dict[str, object] = {
            "duration_seconds": None,
            "width": None,
            "height": None,
            "frame_count": None,
            "metadata_source": None,
            "metadata_notes": None,
            "validation_status": ValidationStatus.PENDING,
        }

        # Signature checks key off the intended extension (temp files are .part).
        check_path = path
        if original_extension:
            ext = original_extension.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            # Use a lightweight wrapper path name for suffix-based checks only.
            check_path = path.with_name(path.stem + ext)
        self._assert_video_container_signature(check_path, path)

        try:
            import cv2  # noqa: F401
        except ImportError:
            notes.append(
                "OpenCV is not installed; video content cannot be validated. "
                "Validation status remains PENDING (not VALID)."
            )
            if not self.ffprobe_available():
                notes.append("FFmpeg/ffprobe not found on PATH.")
            result["metadata_notes"] = "; ".join(notes)
            return result

        data = self._opencv_inspect(path)
        if data.get("duration_note"):
            notes.append(str(data["duration_note"]))

        result.update(
            {
                "width": data["width"],
                "height": data["height"],
                "frame_count": data["frame_count"],
                "duration_seconds": data["duration_seconds"],
                "metadata_source": "opencv",
                "validation_status": ValidationStatus.VALID,
            }
        )
        if not self.ffprobe_available():
            notes.append(
                "FFmpeg/ffprobe not found on PATH; metadata limited to OpenCV"
            )
        result["metadata_notes"] = "; ".join(notes) if notes else None
        return result

    def validate_audio_content(
        self, path: Path, *, original_extension: str | None = None
    ) -> dict[str, object]:
        """Validate audio via wave (WAV) and/or ffprobe. Never invent success."""
        notes: list[str] = []
        result: dict[str, object] = {
            "duration_seconds": None,
            "sample_rate": None,
            "metadata_source": None,
            "metadata_notes": None,
            "validation_status": ValidationStatus.PENDING,
        }
        # Temp upload files use a .part suffix; prefer the client extension.
        ext = (original_extension or path.suffix).lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        if ext == ".wav":
            try:
                with wave.open(str(path), "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    # Force a read to ensure frames are accessible.
                    if n_frames > 0:
                        wav_file.readframes(min(n_frames, sample_rate or 1))
            except wave.Error as exc:
                raise MediaValidationError(
                    f"WAV file is corrupted or undecodable: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise MediaValidationError(
                    f"WAV file could not be validated: {exc}"
                ) from exc

            if channels < 1 or sample_width < 1 or sample_rate < 1 or n_frames < 1:
                raise MediaValidationError(
                    "WAV file has invalid channel, sample rate, or frame count"
                )

            duration = round(n_frames / float(sample_rate), 3)
            result.update(
                {
                    "duration_seconds": duration,
                    "sample_rate": int(sample_rate),
                    "metadata_source": "wave",
                    "validation_status": ValidationStatus.VALID,
                    "metadata_notes": None,
                }
            )
            return result

        if not self.ffprobe_available():
            notes.append(
                "FFmpeg/ffprobe not found on PATH. Non-WAV audio cannot be "
                "fully validated; validation status remains PENDING (not VALID). "
                "Install FFmpeg and ensure ffprobe is on PATH."
            )
            result["metadata_notes"] = "; ".join(notes)
            return result

        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"ffprobe invocation failed: {exc}")
            result["metadata_notes"] = "; ".join(notes)
            return result

        if proc.returncode != 0:
            raise MediaValidationError(
                "Audio file could not be probed by ffprobe (corrupted or unsupported)"
            )

        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MediaValidationError(
                "ffprobe returned non-JSON output for audio file"
            ) from exc

        duration: float | None = None
        sample_rate: int | None = None
        fmt = payload.get("format") or {}
        if fmt.get("duration") is not None:
            try:
                duration = round(float(fmt["duration"]), 3)
            except (TypeError, ValueError):
                notes.append("ffprobe duration was not a valid number")

        audio_stream_found = False
        for stream in payload.get("streams") or []:
            if stream.get("codec_type") != "audio":
                continue
            audio_stream_found = True
            if stream.get("sample_rate") is not None:
                try:
                    sample_rate = int(stream["sample_rate"])
                except (TypeError, ValueError):
                    notes.append("ffprobe sample_rate was not a valid integer")
            if duration is None and stream.get("duration") is not None:
                try:
                    duration = round(float(stream["duration"]), 3)
                except (TypeError, ValueError):
                    notes.append("ffprobe stream duration was not a valid number")
            break

        if not audio_stream_found:
            raise MediaValidationError(
                "No audio stream found in uploaded file (ffprobe)"
            )
        if sample_rate is None or sample_rate < 1:
            raise MediaValidationError(
                "Audio sample rate missing or invalid after ffprobe"
            )
        if duration is None or duration <= 0:
            raise MediaValidationError(
                "Audio duration missing or invalid after ffprobe"
            )

        result.update(
            {
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "metadata_source": "ffprobe",
                "validation_status": ValidationStatus.VALID,
                "metadata_notes": "; ".join(notes) if notes else None,
            }
        )
        return result

    async def save_video(self, avatar_id: str, upload: UploadFile) -> StoredMedia:
        if not upload.filename:
            raise MediaValidationError("Missing filename for video upload")

        self._validate_extension_and_mime(
            upload.filename,
            upload.content_type,
            allowed_extensions=self.settings.allowed_video_extensions,
            allowed_mimes=self.settings.allowed_video_mime_types,
            media_kind="video",
        )

        destination_dir = self.avatar_raw_dir(avatar_id) / "video"
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._unique_filename(upload.filename, destination_dir)
        final_path = destination_dir / safe_name

        tmp_path, size_bytes = await self._stream_to_temp(
            upload,
            self.settings.max_video_upload_bytes,
            "video",
            destination_dir,
        )

        try:
            meta = self.validate_video_content(
                tmp_path,
                original_extension=Path(upload.filename).suffix.lower(),
            )
            # Definitive decode failures raise; PENDING means tools unavailable.
            self._finalize_temp(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Stored video for avatar %s as %s (%s bytes, validation=%s)",
            avatar_id,
            safe_name,
            size_bytes,
            meta["validation_status"],
        )

        info = MediaFileInfo(
            filename=f"video/{safe_name}",
            original_filename=upload.filename,
            size_bytes=size_bytes,
            content_type=upload.content_type,
            uploaded_at=datetime.now(timezone.utc),
            duration_seconds=meta["duration_seconds"],  # type: ignore[arg-type]
            width=meta["width"],  # type: ignore[arg-type]
            height=meta["height"],  # type: ignore[arg-type]
            frame_count=meta["frame_count"],  # type: ignore[arg-type]
            metadata_source=meta["metadata_source"],  # type: ignore[arg-type]
            metadata_notes=meta["metadata_notes"],  # type: ignore[arg-type]
            validation_status=meta["validation_status"],  # type: ignore[arg-type]
        )
        return StoredMedia(info=info, absolute_path=final_path)

    async def save_audio(self, avatar_id: str, upload: UploadFile) -> StoredMedia:
        if not upload.filename:
            raise MediaValidationError("Missing filename for audio upload")

        self._validate_extension_and_mime(
            upload.filename,
            upload.content_type,
            allowed_extensions=self.settings.allowed_audio_extensions,
            allowed_mimes=self.settings.allowed_audio_mime_types,
            media_kind="audio",
        )

        destination_dir = self.avatar_raw_dir(avatar_id) / "audio"
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._unique_filename(upload.filename, destination_dir)
        final_path = destination_dir / safe_name

        tmp_path, size_bytes = await self._stream_to_temp(
            upload,
            self.settings.max_audio_upload_bytes,
            "audio",
            destination_dir,
        )

        try:
            meta = self.validate_audio_content(
                tmp_path,
                original_extension=Path(upload.filename).suffix.lower(),
            )
            self._finalize_temp(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Stored audio for avatar %s as %s (%s bytes, validation=%s)",
            avatar_id,
            safe_name,
            size_bytes,
            meta["validation_status"],
        )

        info = MediaFileInfo(
            filename=f"audio/{safe_name}",
            original_filename=upload.filename,
            size_bytes=size_bytes,
            content_type=upload.content_type,
            uploaded_at=datetime.now(timezone.utc),
            duration_seconds=meta["duration_seconds"],  # type: ignore[arg-type]
            sample_rate=meta["sample_rate"],  # type: ignore[arg-type]
            metadata_source=meta["metadata_source"],  # type: ignore[arg-type]
            metadata_notes=meta["metadata_notes"],  # type: ignore[arg-type]
            validation_status=meta["validation_status"],  # type: ignore[arg-type]
        )
        return StoredMedia(info=info, absolute_path=final_path)
