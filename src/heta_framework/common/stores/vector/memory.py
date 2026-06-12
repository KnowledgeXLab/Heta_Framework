"""In-memory vector store implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from heta_framework.common.stores.vector.types import (
    DistanceMetric,
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorSearchResult,
)


@dataclass
class _Collection:
    config: VectorCollectionConfig
    records: dict[str, VectorRecord] = field(default_factory=dict)


class InMemoryVectorStore:
    """Simple in-memory vector store for tests, demos, and local pipelines."""

    def __init__(self) -> None:
        self._collections: dict[str, _Collection] = {}

    async def create_collection(self, config: VectorCollectionConfig) -> None:
        """Create a collection if it does not already exist."""
        existing = self._collections.get(config.name)
        if existing is not None:
            if existing.config.dimension != config.dimension:
                raise ValueError(
                    f"collection {config.name!r} already exists with dimension "
                    f"{existing.config.dimension}"
                )
            if existing.config.metric != config.metric:
                raise ValueError(
                    f"collection {config.name!r} already exists with metric "
                    f"{existing.config.metric!r}"
                )
            return
        self._collections[config.name] = _Collection(config=config)

    async def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        self._collections.pop(name, None)

    async def has_collection(self, name: str) -> bool:
        """Return whether a collection exists."""
        return name in self._collections

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or update vector records."""
        target = self._get_collection(collection)
        for record in records:
            _validate_dimension(record.vector, target.config)
            target.records[record.id] = record

    async def search(
        self,
        collection: str,
        query: VectorQuery,
    ) -> list[VectorSearchResult]:
        """Search a collection with one vector query."""
        target = self._get_collection(collection)
        _validate_dimension(query.vector, target.config)

        matches = [
            _score_record(record, query.vector, target.config.metric)
            for record in target.records.values()
            if _matches_filter(record.metadata, query.filter)
        ]
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: query.top_k]

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Delete records by id."""
        target = self._get_collection(collection)
        for record_id in ids:
            target.records.pop(record_id, None)

    async def count(self, collection: str) -> int:
        """Return the number of records in a collection."""
        return len(self._get_collection(collection).records)

    async def aclose(self) -> None:
        """Release resources held by the store."""

    def _get_collection(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError as exc:
            raise ValueError(f"collection does not exist: {name}") from exc


def _validate_dimension(vector: Sequence[float], config: VectorCollectionConfig) -> None:
    if len(vector) != config.dimension:
        raise ValueError(
            f"vector dimension mismatch for collection {config.name!r}: "
            f"expected {config.dimension}, got {len(vector)}"
        )


def _score_record(
    record: VectorRecord,
    query_vector: Sequence[float],
    metric: DistanceMetric,
) -> VectorSearchResult:
    score = _score(record.vector, query_vector, metric)
    return VectorSearchResult(
        id=record.id,
        score=score,
        text=record.text,
        metadata=record.metadata,
    )


def _score(left: Sequence[float], right: Sequence[float], metric: DistanceMetric) -> float:
    if metric == "cosine":
        return _cosine_similarity(left, right)
    if metric == "dot":
        return _dot(left, right)
    if metric == "l2":
        return -_l2_distance(left, right)
    raise ValueError(f"unsupported distance metric: {metric}")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(_dot(left, left))
    right_norm = math.sqrt(_dot(right, right))
    if left_norm == 0 or right_norm == 0:
        return 0
    return _dot(left, right) / (left_norm * right_norm)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _matches_filter(metadata: dict[str, Any] | None, filter: dict[str, Any] | None) -> bool:
    if not filter:
        return True
    if metadata is None:
        return False
    return all(metadata.get(key) == value for key, value in filter.items())
