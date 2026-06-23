"""Object store capability protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heta_framework.common.stores.object.types import ObjectInfo


@runtime_checkable
class ObjectStoreProtocol(Protocol):
    """Capability protocol for key-addressed object stores."""

    async def put(self, key: str, data: bytes) -> None:
        """Store bytes at a key."""
        ...

    async def get(self, key: str) -> bytes:
        """Read bytes from a key."""
        ...

    async def exists(self, key: str) -> bool:
        """Return whether a key exists."""
        ...

    async def list(self, prefix: str = "") -> list[ObjectInfo]:
        """List objects under a prefix."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a key if it exists."""
        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""
        ...
