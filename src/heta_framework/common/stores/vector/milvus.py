"""Milvus vector store implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from heta_framework.common.stores.vector.types import (
    DistanceMetric,
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorSearchResult,
)


@dataclass(frozen=True)
class MilvusVectorStoreConfig:
    """Connection and schema settings for Milvus vector stores."""

    uri: str = "http://localhost:19530"
    token: str | None = None
    db_name: str | None = None
    timeout: float = 10
    id_field: str = "id"
    vector_field: str = "vector"
    text_field: str = "text"
    text_max_length: int = 65535
    id_max_length: int = 512

    def __post_init__(self) -> None:
        if self.uri.strip() == "":
            raise ValueError("uri must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.id_field.strip() == "":
            raise ValueError("id_field must not be empty")
        if self.vector_field.strip() == "":
            raise ValueError("vector_field must not be empty")
        if self.text_field.strip() == "":
            raise ValueError("text_field must not be empty")
        if self.text_max_length <= 0:
            raise ValueError("text_max_length must be positive")
        if self.id_max_length <= 0:
            raise ValueError("id_max_length must be positive")


class MilvusVectorStore:
    """Milvus-backed vector store adapter."""

    def __init__(
        self,
        *,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        db_name: str | None = None,
        timeout: float = 10,
        client: Any | None = None,
        id_field: str = "id",
        vector_field: str = "vector",
        text_field: str = "text",
        text_max_length: int = 65535,
        id_max_length: int = 512,
    ) -> None:
        self.config = MilvusVectorStoreConfig(
            uri=uri,
            token=token,
            db_name=db_name,
            timeout=timeout,
            id_field=id_field,
            vector_field=vector_field,
            text_field=text_field,
            text_max_length=text_max_length,
            id_max_length=id_max_length,
        )
        self._client = client if client is not None else _create_client(self.config)
        self._collection_configs: dict[str, VectorCollectionConfig] = {}

    async def create_collection(self, config: VectorCollectionConfig) -> None:
        """Create a collection if it does not already exist."""
        if await self.has_collection(config.name):
            self._collection_configs[config.name] = config
            return

        DataType = _load_data_type()
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name=self.config.id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.config.id_max_length,
        )
        schema.add_field(
            field_name=self.config.vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=config.dimension,
        )
        schema.add_field(
            field_name=self.config.text_field,
            datatype=DataType.VARCHAR,
            max_length=self.config.text_max_length,
            nullable=True,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self.config.vector_field,
            index_type="AUTOINDEX",
            metric_type=_milvus_metric(config.metric),
        )

        self._client.create_collection(
            collection_name=config.name,
            schema=schema,
            index_params=index_params,
        )
        self._collection_configs[config.name] = config

    async def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        if await self.has_collection(name):
            self._client.drop_collection(collection_name=name)
        self._collection_configs.pop(name, None)

    async def has_collection(self, name: str) -> bool:
        """Return whether a collection exists."""
        return bool(self._client.has_collection(collection_name=name))

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or update vector records."""
        if not records:
            return
        config = self._get_collection_config(collection)
        rows = [_record_to_row(record, self.config, config) for record in records]
        if hasattr(self._client, "upsert"):
            self._client.upsert(collection_name=collection, data=rows)
        else:
            await self.delete(collection, [record.id for record in records])
            self._client.insert(collection_name=collection, data=rows)
        self._flush(collection)

    async def search(
        self,
        collection: str,
        query: VectorQuery,
    ) -> list[VectorSearchResult]:
        """Search a collection with one vector query."""
        config = self._get_collection_config(collection)
        if len(query.vector) != config.dimension:
            raise ValueError(
                f"vector dimension mismatch for collection {collection!r}: "
                f"expected {config.dimension}, got {len(query.vector)}"
            )

        result_sets = self._client.search(
            collection_name=collection,
            data=[query.vector],
            anns_field=self.config.vector_field,
            limit=query.top_k,
            filter=_filter_to_expression(query.filter),
            output_fields=[self.config.text_field, "*"],
            search_params={"metric_type": _milvus_metric(config.metric)},
        )
        hits = result_sets[0] if result_sets else []
        return [_hit_to_result(hit, self.config, config.metric) for hit in hits]

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Delete records by id."""
        if not ids:
            return
        self._client.delete(
            collection_name=collection,
            ids=list(ids),
        )
        self._flush(collection)

    async def count(self, collection: str) -> int:
        """Return the number of records in a collection."""
        result = self._client.query(
            collection_name=collection,
            filter="",
            output_fields=["count(*)"],
        )
        if not result:
            return 0
        return int(result[0].get("count(*)", 0))

    async def aclose(self) -> None:
        """Release resources held by the store."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def _flush(self, collection: str) -> None:
        flush = getattr(self._client, "flush", None)
        if flush is not None:
            flush(collection_name=collection)

    def _get_collection_config(self, collection: str) -> VectorCollectionConfig:
        try:
            return self._collection_configs[collection]
        except KeyError as exc:
            raise ValueError(
                f"collection config is unknown: {collection!r}; call create_collection first"
            ) from exc


def _create_client(config: MilvusVectorStoreConfig) -> Any:
    MilvusClient = _load_milvus_client()
    kwargs: dict[str, Any] = {"uri": config.uri, "timeout": config.timeout}
    if config.token is not None:
        kwargs["token"] = config.token
    if config.db_name is not None:
        kwargs["db_name"] = config.db_name
    return MilvusClient(**kwargs)


def _load_milvus_client() -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise ImportError(
            "pymilvus is not installed; install the `heta[milvus]` extra to use "
            "MilvusVectorStore"
        ) from exc
    return MilvusClient


def _load_data_type() -> Any:
    try:
        from pymilvus import DataType
    except ImportError as exc:
        raise ImportError(
            "pymilvus is not installed; install the `heta[milvus]` extra to use "
            "MilvusVectorStore"
        ) from exc
    return DataType


def _record_to_row(
    record: VectorRecord,
    store_config: MilvusVectorStoreConfig,
    collection_config: VectorCollectionConfig,
) -> dict[str, Any]:
    if len(record.vector) != collection_config.dimension:
        raise ValueError(
            f"vector dimension mismatch for collection {collection_config.name!r}: "
            f"expected {collection_config.dimension}, got {len(record.vector)}"
        )
    row = {
        store_config.id_field: record.id,
        store_config.vector_field: record.vector,
        store_config.text_field: record.text,
    }
    if record.metadata:
        row.update(record.metadata)
    return row


def _hit_to_result(
    hit: Any,
    config: MilvusVectorStoreConfig,
    metric: DistanceMetric,
) -> VectorSearchResult:
    if not isinstance(hit, dict):
        hit = dict(hit)

    entity = hit.get("entity") or {}
    record_id = str(hit.get("id", entity.get(config.id_field, "")))
    score = float(hit.get("distance", hit.get("score", 0.0)))
    if metric == "l2":
        score = -score

    metadata = dict(entity)
    text = metadata.pop(config.text_field, None)
    metadata.pop(config.id_field, None)
    metadata.pop(config.vector_field, None)

    return VectorSearchResult(
        id=record_id,
        score=score,
        text=text or None,
        metadata=metadata or None,
    )


def _milvus_metric(metric: DistanceMetric) -> str:
    if metric == "cosine":
        return "COSINE"
    if metric == "dot":
        return "IP"
    if metric == "l2":
        return "L2"
    raise ValueError(f"unsupported distance metric: {metric}")


def _filter_to_expression(filter: dict[str, Any] | None) -> str:
    if not filter:
        return ""
    return " and ".join(f"{key} == {_format_filter_value(value)}" for key, value in filter.items())


def _format_filter_value(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    raise TypeError(f"unsupported metadata filter value: {value!r}")
