"""Protocols for full-text index stores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from heta_framework.common.stores.text_index.types import (
    TextIndexConfig,
    TextIndexRecord,
    TextQuery,
    TextSearchResult,
)


@runtime_checkable
class TextIndexStoreProtocol(Protocol):
    """Store interface for full-text search indexes."""

    async def create_index(self, config: TextIndexConfig) -> None:
        """Create an index if it does not already exist."""
        ...

    async def drop_index(self, name: str) -> None:
        """Drop an index if it exists."""
        ...

    async def upsert(self, index: str, records: Sequence[TextIndexRecord]) -> None:
        """Insert or update text records."""
        ...

    async def search(self, index: str, query: TextQuery) -> list[TextSearchResult]:
        """Search one full-text index."""
        ...

    async def delete(self, index: str, ids: Sequence[str]) -> None:
        """Delete records by id."""
        ...

    async def count(self, index: str) -> int:
        """Return the number of records in an index."""
        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""
        ...
