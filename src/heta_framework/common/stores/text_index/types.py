"""Data types for full-text index stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextIndexConfig:
    """Configuration for one full-text index."""

    name: str

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")


@dataclass(frozen=True)
class TextIndexRecord:
    """One text document stored in a full-text index."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TextQuery:
    """One full-text query."""

    text: str
    top_k: int = 10
    filters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class TextSearchResult:
    """One full-text search result."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))
