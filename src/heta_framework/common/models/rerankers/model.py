"""LiteLLM-backed rerank model client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from heta_framework.common.models.rerankers.config import RerankConfig
from heta_framework.common.models.rerankers.errors import (
    RerankRequestError,
    RerankResponseError,
)
from heta_framework.common.models.rerankers.types import (
    RerankItem,
    RerankOptions,
    RerankRequest,
    RerankResult,
)


class RerankModel:
    """Async-first rerank model client backed by LiteLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        request_timeout: float = 120,
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
        top_n: int | None = None,
        drop_unsupported_params: bool = True,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = RerankConfig(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_concurrent_requests=max_concurrent_requests,
            top_n=top_n,
            drop_unsupported_params=drop_unsupported_params,
            provider_options=provider_options,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    @classmethod
    def from_config(cls, config: RerankConfig) -> "RerankModel":
        """Create a rerank model from ``RerankConfig``."""
        return cls(**asdict(config))

    @property
    def model_name(self) -> str:
        """Model name passed to LiteLLM."""
        return self.config.model_name

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank one batch of documents for a query."""
        _validate_request(request)
        litellm = _load_litellm()
        try:
            async with self._semaphore:
                response = await litellm.arerank(**self._build_payload(request))
        except Exception as exc:
            raise RerankRequestError(
                "rerank request failed",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc
        return self._parse_result(response, request)

    async def rerank_many(self, requests: Sequence[RerankRequest]) -> list[RerankResult]:
        """Run multiple rerank requests concurrently while preserving order."""
        return list(await asyncio.gather(*(self.rerank(request) for request in requests)))

    async def aclose(self) -> None:
        """Close resources held by the model.

        LiteLLM manages its own HTTP clients, so there is currently nothing to close.
        """

    async def __aenter__(self) -> "RerankModel":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _build_payload(self, request: RerankRequest) -> dict[str, Any]:
        options = request.options or RerankOptions()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "query": request.query,
            "documents": request.documents,
            "timeout": self.config.request_timeout,
            "num_retries": self.config.max_retries,
            "drop_params": self.config.drop_unsupported_params,
        }
        if self.config.api_key:
            payload["api_key"] = self.config.api_key
        if self.config.api_base:
            payload["api_base"] = self.config.api_base

        top_n = options.top_n if options.top_n is not None else self.config.top_n
        if top_n is not None:
            payload["top_n"] = top_n
        if options.return_documents is not None:
            payload["return_documents"] = options.return_documents

        if self.config.provider_options:
            payload.update(self.config.provider_options)
        if options.provider_options:
            payload.update(options.provider_options)
        return payload

    def _parse_result(self, response: Any, request: RerankRequest) -> RerankResult:
        data = _to_dict(response)
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            raise RerankResponseError(
                "rerank response is missing results",
                trace_context=request.trace_context,
            )

        rankings: list[RerankItem] = []
        for raw_item in raw_results:
            item = raw_item if isinstance(raw_item, dict) else _to_dict(raw_item)
            try:
                index = int(item["index"])
                score = _parse_score(item)
            except Exception as exc:
                raise RerankResponseError(
                    "rerank response item is missing index or score",
                    trace_context=request.trace_context,
                    cause=exc,
                ) from exc
            if index < 0 or index >= len(request.documents):
                raise RerankResponseError(
                    "rerank response item index is outside the document list",
                    trace_context=request.trace_context,
                )
            rankings.append(
                RerankItem(
                    index=index,
                    score=score,
                    text=_parse_document_text(item),
                    metadata=_parse_metadata(item),
                )
            )

        return RerankResult(
            rankings=rankings,
            model_name=self.config.model_name,
            trace_context=request.trace_context,
            raw_response=data,
        )


def _validate_request(request: RerankRequest) -> None:
    if request.query.strip() == "":
        raise RerankRequestError(
            "rerank query must not be empty",
            trace_context=request.trace_context,
        )
    if not request.documents:
        raise RerankRequestError(
            "rerank request must include at least one document",
            trace_context=request.trace_context,
        )
    if any(document.strip() == "" for document in request.documents):
        raise RerankRequestError(
            "rerank documents must not be empty",
            trace_context=request.trace_context,
        )


def _load_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise RerankRequestError(
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


def _parse_score(item: dict[str, Any]) -> float:
    if "relevance_score" in item:
        return float(item["relevance_score"])
    if "score" in item:
        return float(item["score"])
    raise KeyError("relevance_score")


def _parse_document_text(item: dict[str, Any]) -> str | None:
    document = item.get("document")
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        text = document.get("text")
        return text if isinstance(text, str) else None
    return None


def _parse_metadata(item: dict[str, Any]) -> dict[str, Any] | None:
    excluded_keys = {"index", "score", "relevance_score", "document"}
    metadata = {key: value for key, value in item.items() if key not in excluded_keys}
    document = item.get("document")
    if isinstance(document, dict):
        document_metadata = {key: value for key, value in document.items() if key != "text"}
        if document_metadata:
            metadata["document"] = document_metadata
    return metadata or None
