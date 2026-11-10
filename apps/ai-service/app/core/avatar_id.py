"""Avatar ID format validation.

Application-generated IDs are uuid4().hex (32 lowercase hex characters).
Incoming path parameters must match before any filesystem path is constructed.
"""

from __future__ import annotations

import re

AVATAR_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class InvalidAvatarIdError(ValueError):
    """Raised when an avatar_id path parameter fails format validation."""


def validate_avatar_id(avatar_id: str) -> str:
    """Return avatar_id if it matches uuid4().hex; otherwise raise."""
    if not AVATAR_ID_PATTERN.fullmatch(avatar_id):
        raise InvalidAvatarIdError(
            "Invalid avatar_id format. Expected 32 lowercase hexadecimal "
            "characters (uuid4().hex)."
        )
    return avatar_id
