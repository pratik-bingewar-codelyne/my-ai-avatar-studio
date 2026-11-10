"""Helpers to build tiny valid/invalid media fixtures for tests."""

from __future__ import annotations

import io
import wave
from pathlib import Path


def build_valid_wav_bytes(
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.1,
) -> bytes:
    """Return a minimal valid mono 16-bit PCM WAV."""
    n_frames = max(int(sample_rate * duration_seconds), 1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
    return buffer.getvalue()


def build_valid_avi_file(path: Path) -> Path:
    """Write a tiny decodable AVI via OpenCV VideoWriter."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open for test fixture")
    for index in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 40) % 255
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Failed to create valid AVI test fixture")
    return path
