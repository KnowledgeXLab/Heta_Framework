"""Language model error types."""

from __future__ import annotations

from typing import Any


class ModelError(RuntimeError):
    """Base error for language model failures."""

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


class ModelRequestError(ModelError):
    """Raised when the provider request fails."""


class ModelResponseError(ModelError):
    """Raised when the provider response is malformed or cannot be parsed."""

