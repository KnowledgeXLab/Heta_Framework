"""Elasticsearch-backed full-text index store."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from heta_framework.common.stores.text_index.types import (
    TextIndexConfig,
    TextIndexRecord,
    TextQuery,
    TextSearchResult,
)

_BulkHelper = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ElasticsearchTextIndexStoreConfig:
    """Connection and index defaults for Elasticsearch full-text search."""

    hosts: str | Sequence[str] = "http://localhost:9200"
    api_key: str | tuple[str, str] | None = None
    basic_auth: tuple[str, str] | None = None
    request_timeout: float = 30.0
    refresh_after_write: bool = True
    content_field: str = "content_text"
    metadata_field: str = "metadata"

    def __post_init__(self) -> None:
        if isinstance(self.hosts, str):
            if self.hosts.strip() == "":
                raise ValueError("hosts must not be empty")
        elif len(self.hosts) == 0:
            raise ValueError("hosts must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.content_field.strip() == "":
            raise ValueError("content_field must not be empty")
        if self.metadata_field.strip() == "":
            raise ValueError("metadata_field must not be empty")


class ElasticsearchTextIndexStore:
    """TextIndexStore implementation backed by Elasticsearch BM25 search."""

    def __init__(
        self,
        config: ElasticsearchTextIndexStoreConfig | None = None,
        *,
        client: Any | None = None,
        bulk_helper: _BulkHelper | None = None,
    ) -> None:
        self.config = config or ElasticsearchTextIndexStoreConfig()
        self._client = client if client is not None else _create_client(self.config)
        self._bulk_helper = bulk_helper if bulk_helper is not None else _load_bulk_helper()
        self._owns_client = client is None

    async def create_index(self, config: TextIndexConfig) -> None:
        """Create an Elasticsearch index if it does not already exist."""
        exists = await self._client.indices.exists(index=config.name)
        if exists:
            return
        await self._client.indices.create(
            index=config.name,
            mappings={
                "properties": {
                    self.config.content_field: {"type": "text"},
                    self.config.metadata_field: {"type": "object", "dynamic": True},
                }
            },
        )

    async def drop_index(self, name: str) -> None:
        """Drop an Elasticsearch index if it exists."""
        await self._client.indices.delete(index=name, ignore_unavailable=True)

    async def upsert(self, index: str, records: Sequence[TextIndexRecord]) -> None:
        """Insert or update text records using the Elasticsearch bulk API."""
        if not records:
            return
        actions = [
            {
                "_op_type": "index",
                "_index": index,
                "_id": record.id,
                "_source": {
                    self.config.content_field: record.text,
                    self.config.metadata_field: record.metadata,
                },
            }
            for record in records
        ]
        await self._bulk_helper(
            self._client,
            actions,
            refresh=self.config.refresh_after_write,
        )

    async def search(self, index: str, query: TextQuery) -> list[TextSearchResult]:
        """Search an Elasticsearch full-text index."""
        response = await self._client.search(
            index=index,
            query={
                "bool": {
                    "must": [{"match": {self.config.content_field: query.text}}],
                    "filter": _filter_clauses(query.filters, self.config.metadata_field),
                }
            },
            size=query.top_k,
        )
        hits = response.get("hits", {}).get("hits", [])
        return [self._to_search_result(hit) for hit in hits]

    async def delete(self, index: str, ids: Sequence[str]) -> None:
        """Delete text records by id using the Elasticsearch bulk API."""
        if not ids:
            return
        actions = [
            {
                "_op_type": "delete",
                "_index": index,
                "_id": record_id,
            }
            for record_id in ids
        ]
        await self._bulk_helper(
            self._client,
            actions,
            refresh=self.config.refresh_after_write,
            ignore_status=(404,),
        )

    async def count(self, index: str) -> int:
        """Return the number of records in an Elasticsearch index."""
        response = await self._client.count(index=index)
        return int(response.get("count", 0))

    async def aclose(self) -> None:
        """Close the owned Elasticsearch client."""
        if self._owns_client:
            await self._client.close()

    def _to_search_result(self, hit: Mapping[str, Any]) -> TextSearchResult:
        source = _mapping(hit.get("_source"))
        metadata = _mapping(source.get(self.config.metadata_field))
        return TextSearchResult(
            id=str(hit.get("_id", "")),
            text=str(source.get(self.config.content_field, "")),
            score=float(hit.get("_score") or 0.0),
            metadata=dict(metadata),
        )


def _filter_clauses(filters: Mapping[str, Any] | None, metadata_field: str) -> list[dict[str, Any]]:
    if not filters:
        return []
    clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        field = f"{metadata_field}.{key}.keyword" if isinstance(value, str) else f"{metadata_field}.{key}"
        clauses.append({"term": {field: value}})
    return clauses


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _create_client(config: ElasticsearchTextIndexStoreConfig) -> Any:
    try:
        from elasticsearch import AsyncElasticsearch
    except ImportError as exc:
        raise ImportError(
            "ElasticsearchTextIndexStore requires the `elasticsearch` package. "
            "Install it with `pip install 'heta[elasticsearch]'`."
        ) from exc
    return AsyncElasticsearch(
        hosts=config.hosts,
        api_key=config.api_key,
        basic_auth=config.basic_auth,
        request_timeout=config.request_timeout,
    )


def _load_bulk_helper() -> _BulkHelper:
    try:
        from elasticsearch.helpers import async_bulk
    except ImportError as exc:
        raise ImportError(
            "ElasticsearchTextIndexStore requires the `elasticsearch` package. "
            "Install it with `pip install 'heta[elasticsearch]'`."
        ) from exc
    return async_bulk
