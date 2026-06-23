"""Data types shared by rerank model clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankOptions:
    """Per-request rerank options."""

    top_n: int | None = None
    return_documents: bool | None = None
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be positive")


@dataclass(frozen=True)
class RerankRequest:
    """One rerank request."""

    query: str
    documents: list[str]
    options: RerankOptions | None = None
    trace_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class RerankItem:
    """One ranked document returned by a rerank model."""

    index: int
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RerankResult:
    """Final result for one rerank request."""

    rankings: list[RerankItem]
    model_name: str = ""
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
