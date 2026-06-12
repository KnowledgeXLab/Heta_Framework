"""Embedding model error types."""

from __future__ import annotations

from typing import Any


class EmbeddingError(RuntimeError):
    """Base error for embedding model failures."""

    def __init__(
        self,
        message: str,
        *,
        trace_context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.trace_context = trace_context
        self.__cause__ = cause


class EmbeddingRequestError(EmbeddingError):
    """Raised when the embedding request fails."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when the embedding response is malformed or cannot be parsed."""
