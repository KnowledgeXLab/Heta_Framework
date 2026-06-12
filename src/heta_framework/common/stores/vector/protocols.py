"""Vector store capability protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from heta_framework.common.stores.vector.types import (
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorSearchResult,
)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Capability protocol for vector stores."""

    async def create_collection(self, config: VectorCollectionConfig) -> None:
        """Create a collection if it does not already exist."""
        ...

    async def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        ...

    async def has_collection(self, name: str) -> bool:
        """Return whether a collection exists."""
        ...

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or update vector records."""
        ...

    async def search(
        self,
        collection: str,
        query: VectorQuery,
    ) -> list[VectorSearchResult]:
        """Search a collection with one vector query."""
        ...

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Delete records by id."""
        ...

    async def count(self, collection: str) -> int:
        """Return the number of records in a collection."""
        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""
        ...
