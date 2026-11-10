"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured, console-friendly format."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Helper to attach structured fields to log records."""
    return {"extra": kwargs}
