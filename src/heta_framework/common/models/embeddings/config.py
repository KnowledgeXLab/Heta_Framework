"""Configuration objects for embedding model clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingConfig:
    """Long-lived LiteLLM-backed embedding client configuration."""

    model_name: str
    api_key: str | None = None
    api_base: str | None = None
    request_timeout: float = 120
    max_retries: int = 3
    max_concurrent_requests: int = 10
    dimensions: int | None = None
    encoding_format: str | None = None
    drop_unsupported_params: bool = True
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
