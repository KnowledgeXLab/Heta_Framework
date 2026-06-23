"""Local filesystem object store implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from heta_framework.common.stores.object.types import (
    ObjectInfo,
    validate_object_key,
    validate_object_prefix,
)


@dataclass(frozen=True)
class LocalObjectStoreConfig:
    """Configuration for local filesystem object stores."""

    root: Path | str

    def __post_init__(self) -> None:
        if str(self.root).strip() == "":
            raise ValueError("root must not be empty")


class LocalObjectStore:
    """Object store backed by a local directory."""

    def __init__(self, root: Path | str) -> None:
        self.config = LocalObjectStoreConfig(root=root)
        self._root = Path(root).expanduser().resolve()

    async def put(self, key: str, data: bytes) -> None:
        """Store bytes at a key."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get(self, key: str) -> bytes:
        """Read bytes from a key."""
        return self._path_for_key(key).read_bytes()

    async def exists(self, key: str) -> bool:
        """Return whether a key exists."""
        return self._path_for_key(key).is_file()

    async def list(self, prefix: str = "") -> list[ObjectInfo]:
        """List objects under a prefix."""
        normalized_prefix = validate_object_prefix(prefix)
        if not self._root.exists():
            return []

        objects: list[ObjectInfo] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self._root).as_posix()
            if normalized_prefix and not _matches_prefix(key, normalized_prefix):
                continue
            stat = path.stat()
            objects.append(
                ObjectInfo(
                    key=key,
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
        objects.sort(key=lambda item: item.key)
        return objects

    async def delete(self, key: str) -> None:
        """Delete a key if it exists."""
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()

    async def aclose(self) -> None:
        """Release resources held by the store."""

    def _path_for_key(self, key: str) -> Path:
        normalized = validate_object_key(key)
        path = (self._root / normalized).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("key escapes object store root")
        return path


def _matches_prefix(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix.rstrip('/')}/")
