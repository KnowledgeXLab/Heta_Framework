"""Data types for object stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal


S3AddressingStyle = Literal["auto", "path", "virtual"]


@dataclass(frozen=True)
class ObjectInfo:
    """Basic information about one stored object."""

    key: str
    size: int | None = None
    modified_at: datetime | None = None
    etag: str | None = None

    def __post_init__(self) -> None:
        validate_object_key(self.key)
        if self.size is not None and self.size < 0:
            raise ValueError("size must not be negative")


def validate_object_key(key: str) -> str:
    """Validate and return a normalized object key."""
    normalized = _normalize_key(key, allow_empty=False)
    return normalized


def validate_object_prefix(prefix: str = "") -> str:
    """Validate and return a normalized object prefix."""
    return _normalize_key(prefix, allow_empty=True)


def join_object_key(prefix: str, key: str) -> str:
    """Join a store prefix and object key."""
    normalized_key = validate_object_key(key)
    normalized_prefix = validate_object_prefix(prefix)
    if normalized_prefix == "":
        return normalized_key
    return f"{normalized_prefix.rstrip('/')}/{normalized_key}"


def strip_object_prefix(prefix: str, key: str) -> str:
    """Strip a store prefix from a full backend key."""
    normalized_key = validate_object_key(key)
    normalized_prefix = validate_object_prefix(prefix)
    if normalized_prefix == "":
        return normalized_key
    marker = f"{normalized_prefix.rstrip('/')}/"
    if normalized_key == normalized_prefix.rstrip("/"):
        return ""
    if not normalized_key.startswith(marker):
        return normalized_key
    return normalized_key[len(marker) :]


def _normalize_key(key: str, *, allow_empty: bool) -> str:
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    if "\\" in key:
        raise ValueError("key must use POSIX separators")
    if key.startswith("/"):
        raise ValueError("key must be relative")

    stripped = key.strip("/")
    if stripped == "":
        if allow_empty:
            return ""
        raise ValueError("key must not be empty")

    path = PurePosixPath(stripped)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("key must not contain empty, '.', or '..' segments")
    return path.as_posix()
