"""Rerank model error types."""

from __future__ import annotations

from typing import Any


class RerankError(RuntimeError):
    """Base error for rerank model failures."""

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


class RerankRequestError(RerankError):
    """Raised when the rerank request fails."""


class RerankResponseError(RerankError):
    """Raised when the rerank response is malformed or cannot be parsed."""
