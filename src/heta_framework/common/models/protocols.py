"""Model capability protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from heta_framework.common.models.embeddings.types import EmbeddingRequest, EmbeddingResult
from heta_framework.common.models.language.types import ModelChunk, ModelRequest, ModelResult
from heta_framework.common.models.rerankers.types import RerankRequest, RerankResult


@runtime_checkable
class LanguageModelProtocol(Protocol):
    """Capability protocol for language models."""

    @property
    def model_name(self) -> str:
        """Model name used by this client."""
        ...

    async def invoke(self, request: ModelRequest) -> ModelResult:
        """Run one language model request."""
        ...

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        """Run multiple language model requests while preserving order."""
        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        """Stream one language model request as text deltas."""
        ...


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Capability protocol for embedding models."""

    @property
    def model_name(self) -> str:
        """Model name used by this client."""
        ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed one batch of texts."""
        ...

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        """Run multiple embedding requests while preserving order."""
        ...


@runtime_checkable
class RerankModelProtocol(Protocol):
    """Capability protocol for rerank models."""

    @property
    def model_name(self) -> str:
        """Model name used by this client."""
        ...

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank one batch of documents for a query."""
        ...

    async def rerank_many(self, requests: Sequence[RerankRequest]) -> list[RerankResult]:
        """Run multiple rerank requests while preserving order."""
        ...
