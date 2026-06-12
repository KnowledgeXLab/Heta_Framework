"""LiteLLM-backed embedding model client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from heta_framework.common.models.embeddings.config import EmbeddingConfig
from heta_framework.common.models.embeddings.errors import (
    EmbeddingRequestError,
    EmbeddingResponseError,
)
from heta_framework.common.models.embeddings.types import (
    EmbeddingOptions,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingUsage,
)


class EmbeddingModel:
    """Async-first embedding model client backed by LiteLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        request_timeout: float = 120,
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        drop_unsupported_params: bool = True,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = EmbeddingConfig(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_concurrent_requests=max_concurrent_requests,
            dimensions=dimensions,
            encoding_format=encoding_format,
            drop_unsupported_params=drop_unsupported_params,
            provider_options=provider_options,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "EmbeddingModel":
        """Create an embedding model from ``EmbeddingConfig``."""
        return cls(**asdict(config))

    @property
    def model_name(self) -> str:
        """Model name passed to LiteLLM."""
        return self.config.model_name

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed one batch of texts."""
        _validate_request(request)
        litellm = _load_litellm()
        try:
            async with self._semaphore:
                response = await litellm.aembedding(**self._build_payload(request))
        except Exception as exc:
            raise EmbeddingRequestError(
                "embedding request failed",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        return self._parse_result(response, request)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        """Run multiple embedding requests concurrently while preserving order."""
        return list(await asyncio.gather(*(self.embed(request) for request in requests)))

    async def aclose(self) -> None:
        """Close resources held by the model.

        LiteLLM manages its own HTTP clients, so there is currently nothing to close.
        """

    async def __aenter__(self) -> "EmbeddingModel":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _build_payload(self, request: EmbeddingRequest) -> dict[str, Any]:
        options = request.options or EmbeddingOptions()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "input": request.texts,
            "timeout": self.config.request_timeout,
            "num_retries": self.config.max_retries,
            "drop_params": self.config.drop_unsupported_params,
        }
        if self.config.api_key:
            payload["api_key"] = self.config.api_key
        if self.config.api_base:
            payload["api_base"] = self.config.api_base

        dimensions = options.dimensions if options.dimensions is not None else self.config.dimensions
        if dimensions is not None:
            payload["dimensions"] = dimensions

        encoding_format = (
            options.encoding_format
            if options.encoding_format is not None
            else self.config.encoding_format
        )
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format

        if self.config.provider_options:
            payload.update(self.config.provider_options)
        if options.provider_options:
            payload.update(options.provider_options)
        return payload

    def _parse_result(self, response: Any, request: EmbeddingRequest) -> EmbeddingResult:
        data = _to_dict(response)
        try:
            embeddings = data["data"]
            vectors = [
                [float(value) for value in item["embedding"]]
                for item in sorted(embeddings, key=lambda item: item.get("index", 0))
            ]
        except Exception as exc:
            raise EmbeddingResponseError(
                "embedding response is missing expected fields",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        if len(vectors) != len(request.texts):
            raise EmbeddingResponseError(
                "embedding response vector count does not match input text count",
                trace_context=request.trace_context,
            )
        return EmbeddingResult(
            vectors=vectors,
            model_name=self.config.model_name,
            usage=_parse_usage(data.get("usage")),
            trace_context=request.trace_context,
            raw_response=data,
        )


def _validate_request(request: EmbeddingRequest) -> None:
    if not request.texts:
        raise EmbeddingRequestError(
            "embedding request must include at least one text",
            trace_context=request.trace_context,
        )
    if any(text.strip() == "" for text in request.texts):
        raise EmbeddingRequestError(
            "embedding request texts must not be empty",
            trace_context=request.trace_context,
        )


def _load_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise EmbeddingRequestError(
            "LiteLLM is not installed; install the `heta` package with runtime dependencies",
            cause=exc,
        ) from exc
    return litellm


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        return json.loads(value.json())
    raise TypeError(f"cannot convert {type(value)!r} to dict")


def _parse_usage(raw_usage: Any) -> EmbeddingUsage | None:
    if raw_usage is None:
        return None
    usage = raw_usage if isinstance(raw_usage, dict) else _to_dict(raw_usage)
    return EmbeddingUsage(
        prompt_tokens=usage.get("prompt_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
