"""Data types for vector stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


DistanceMetric = Literal["cosine", "dot", "l2"]


@dataclass(frozen=True)
class VectorRecord:
    """One vector record to be stored."""

    id: str
    vector: list[float]
    text: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if not self.vector:
            raise ValueError("vector must not be empty")


@dataclass(frozen=True)
class VectorQuery:
    """One vector search query."""

    vector: list[float]
    top_k: int = 10
    filter: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("vector must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class VectorSearchResult:
    """One vector search result."""

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class VectorCollectionConfig:
    """Configuration for a vector collection."""

    name: str
    dimension: int
    metric: DistanceMetric = "cosine"
    metadata_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
