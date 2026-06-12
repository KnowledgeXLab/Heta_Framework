"""Data types shared by embedding model clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingUsage:
    """Token usage reported by an embedding model service."""

    prompt_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingOptions:
    """Per-request embedding options."""

    dimensions: int | None = None
    encoding_format: str | None = None
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")


@dataclass(frozen=True)
class EmbeddingRequest:
    """One embedding request."""

    texts: list[str]
    options: EmbeddingOptions | None = None
    trace_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Final result for one embedding request."""

    vectors: list[list[float]]
    model_name: str = ""
    usage: EmbeddingUsage | None = None
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
