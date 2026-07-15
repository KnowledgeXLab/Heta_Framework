"""Build LightRAG SQL tables and vector indexes from graph artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import (
    VectorCollectionConfig,
    VectorRecord,
    VectorStoreProtocol,
)
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.graph_storage import batches, compact_json, validate_identifier
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class LightRAGTableNames:
    """SQL table names used by LightRAG storage."""

    entities: str = "light_rag_entities"
    relations: str = "light_rag_relations"
    chunks: str = "light_rag_chunks"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="table_names.entities")
        validate_identifier(self.relations, field_name="table_names.relations")
        validate_identifier(self.chunks, field_name="table_names.chunks")


@dataclass(frozen=True)
class LightRAGVectorCollections:
    """Vector collection names used by LightRAG search."""

    entities: str = "light_rag_entities"
    relationships: str = "light_rag_relationships"
    chunks: str = "light_rag_chunks"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="vector_collections.entities")
        validate_identifier(
            self.relationships,
            field_name="vector_collections.relationships",
        )
        validate_identifier(self.chunks, field_name="vector_collections.chunks")


@dataclass(frozen=True)
class BuildLightRAGGraphConfig:
    """Configuration for BuildLightRAGGraph."""

    table_names: LightRAGTableNames = field(default_factory=LightRAGTableNames)
    vector_collections: LightRAGVectorCollections = field(
        default_factory=LightRAGVectorCollections
    )
    graph_node_keys_artifact: str = "light_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "light_rag_graph_edge_keys"
    chunk_keys_artifact: str = "chunk_keys"
    result_artifact: str = "build_light_rag_graph_result"
    vector_metric: str = "cosine"
    batch_size: int = 128
    object_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "graph_node_keys_artifact",
            "graph_edge_keys_artifact",
            "chunk_keys_artifact",
            "result_artifact",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError(f"{field_name} must not be empty")
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


@dataclass(frozen=True)
class BuildLightRAGGraphResult:
    """Artifacts produced by BuildLightRAGGraph."""

    entity_count: int
    relation_count: int
    chunk_count: int
    entity_vector_count: int
    relationship_vector_count: int
    chunk_vector_count: int
    vector_dimension: int


class BuildLightRAGGraph:
    """Write LightRAG graph artifacts into SQL and vector stores."""

    name = "build_lightrag_graph"

    def __init__(self, config: BuildLightRAGGraphConfig | None = None) -> None:
        self.config = config or BuildLightRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("sql", self.config.sql_store),
                    store_ref("vector", self.config.vector_store),
                    model_ref("embedding", self.config.embedding_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                    self.config.chunk_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts, query modes, and search assets produced by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store)
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset({self.config.result_artifact}),
            queries=frozenset(
                {
                    "light_rag_local_query",
                    "light_rag_global_query",
                    "light_rag_hybrid_query",
                    "light_rag_mix_query",
                }
            ),
            search_assets=(
                SearchAsset(
                    kind="light_rag_tables",
                    name=self.config.table_names.entities,
                    store=sql_store_ref.key,
                    metadata={
                        "entities_table": self.config.table_names.entities,
                        "relations_table": self.config.table_names.relations,
                        "chunks_table": self.config.table_names.chunks,
                    },
                ),
                SearchAsset(
                    kind="light_rag_vector_index",
                    name=self.config.vector_collections.entities,
                    store=vector_store_ref.key,
                    metadata={
                        "entity_collection": self.config.vector_collections.entities,
                        "relationship_collection": (
                            self.config.vector_collections.relationships
                        ),
                        "chunk_collection": self.config.vector_collections.chunks,
                    },
                ),
            ),
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return SQL tables and vector collections produced by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store).key
        vector_store_ref = store_ref("vector", self.config.vector_store).key
        return StepCleanupPlan(
            (
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.entities,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.relations,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.chunks,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.vector_collections.entities,
                    component=vector_store_ref,
                ),
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.vector_collections.relationships,
                    component=vector_store_ref,
                ),
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.vector_collections.chunks,
                    component=vector_store_ref,
                ),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the LightRAG build step."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        sql_store = _require_sql_store(
            context.get_component(store_ref("sql", self.config.sql_store).key)
        )
        vector_store = _require_vector_store(
            context.get_component(store_ref("vector", self.config.vector_store).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )

        node_keys = tuple(context.get_artifact(self.config.graph_node_keys_artifact))
        edge_keys = tuple(context.get_artifact(self.config.graph_edge_keys_artifact))
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))

        nodes = [json.loads((await object_store.get(key)).decode("utf-8")) for key in node_keys]
        edges = [json.loads((await object_store.get(key)).decode("utf-8")) for key in edge_keys]
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        node_rows = [_entity_row(node) for node in nodes]
        edge_rows = [_relation_row(edge) for edge in edges]
        chunk_rows = [_chunk_row(chunk) for chunk in chunks]
        entity_vectors = await _embed_graph_nodes(
            embedding_model,
            nodes,
            batch_size=self.config.batch_size,
        )
        relationship_vectors = await _embed_graph_edges(
            embedding_model,
            edges,
            batch_size=self.config.batch_size,
        )
        chunk_vectors = await _embed_chunks(
            embedding_model,
            chunks,
            batch_size=self.config.batch_size,
        )
        vector_dimension = _vector_dimension(entity_vectors, relationship_vectors, chunk_vectors)

        async with sql_store.transaction() as tx:
            await _ensure_lightrag_tables(tx, self.config.table_names)
            for batch in batches(node_rows, self.config.batch_size):
                await _upsert_entity_rows(tx, self.config.table_names.entities, batch)
            for batch in batches(edge_rows, self.config.batch_size):
                await _upsert_relation_rows(tx, self.config.table_names.relations, batch)
            for batch in batches(chunk_rows, self.config.batch_size):
                await _upsert_chunk_rows(tx, self.config.table_names.chunks, batch)

        if entity_vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.entities,
                    dimension=len(entity_vectors[0].vector),
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(entity_vectors, self.config.batch_size):
                await vector_store.upsert(self.config.vector_collections.entities, batch)
        if relationship_vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.relationships,
                    dimension=len(relationship_vectors[0].vector),
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(relationship_vectors, self.config.batch_size):
                await vector_store.upsert(
                    self.config.vector_collections.relationships,
                    batch,
                )
        if chunk_vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.chunks,
                    dimension=len(chunk_vectors[0].vector),
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(chunk_vectors, self.config.batch_size):
                await vector_store.upsert(self.config.vector_collections.chunks, batch)

        context.set_artifact(
            self.config.result_artifact,
            BuildLightRAGGraphResult(
                entity_count=len(nodes),
                relation_count=len(edges),
                chunk_count=len(chunks),
                entity_vector_count=len(entity_vectors),
                relationship_vector_count=len(relationship_vectors),
                chunk_vector_count=len(chunk_vectors),
                vector_dimension=vector_dimension,
            ),
        )


async def _ensure_lightrag_tables(
    tx: SQLStoreProtocol,
    table_names: LightRAGTableNames,
) -> None:
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.entities} (
            entity_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            file_path TEXT,
            file_paths TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.relations} (
            relation_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            description TEXT NOT NULL,
            keywords TEXT NOT NULL,
            weight REAL NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            file_path TEXT,
            file_paths TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.chunks} (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source_key TEXT NOT NULL,
            source_name TEXT,
            source_file_type TEXT,
            page_index INTEGER,
            chunk_index INTEGER NOT NULL,
            token_start INTEGER,
            token_end INTEGER,
            metadata TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


async def _upsert_entity_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                entity_id, entity_name, entity_type, description, source_id,
                source_ids, file_path, file_paths, properties, updated_at
            )
            VALUES (
                :entity_id, :entity_name, :entity_type, :description, :source_id,
                :source_ids, :file_path, :file_paths, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_type = excluded.entity_type,
                description = excluded.description,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                file_path = excluded.file_path,
                file_paths = excluded.file_paths,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_relation_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                relation_id, source_entity_id, target_entity_id, description,
                keywords, weight, source_id, source_ids, file_path, file_paths,
                properties, updated_at
            )
            VALUES (
                :relation_id, :source_entity_id, :target_entity_id, :description,
                :keywords, :weight, :source_id, :source_ids, :file_path,
                :file_paths, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (relation_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                target_entity_id = excluded.target_entity_id,
                description = excluded.description,
                keywords = excluded.keywords,
                weight = excluded.weight,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                file_path = excluded.file_path,
                file_paths = excluded.file_paths,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_chunk_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                chunk_id, document_id, content, source_key, source_name,
                source_file_type, page_index, chunk_index, token_start, token_end,
                metadata, updated_at
            )
            VALUES (
                :chunk_id, :document_id, :content, :source_key, :source_name,
                :source_file_type, :page_index, :chunk_index, :token_start,
                :token_end, :metadata, CURRENT_TIMESTAMP
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                content = excluded.content,
                source_key = excluded.source_key,
                source_name = excluded.source_name,
                source_file_type = excluded.source_file_type,
                page_index = excluded.page_index,
                chunk_index = excluded.chunk_index,
                token_start = excluded.token_start,
                token_end = excluded.token_end,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _embed_graph_nodes(
    embedding_model: EmbeddingModelProtocol,
    nodes: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for batch in batches(nodes, batch_size):
        texts = [_node_vector_text(node) for node in batch]
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={
                    "step": BuildLightRAGGraph.name,
                    "purpose": "light_rag_entity_index",
                },
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("node embedding result count must match batch size")
        for node, text, vector in zip(batch, texts, result.vectors, strict=True):
            properties = dict(node.get("properties") or {})
            records.append(
                VectorRecord(
                    id=str(node["id"]),
                    vector=[float(value) for value in vector],
                    text=text,
                    metadata={
                        "fact_type": "light_rag_entity",
                        "entity_name": str(properties.get("entity_name") or node["id"]),
                        "entity_type": str(properties.get("entity_type") or "unknown"),
                        "description": str(properties.get("description") or ""),
                        "source_chunk_ids": list(_list_value(properties.get("source_ids"))),
                        "source_ids": list(_list_value(properties.get("source_ids"))),
                        "file_path": str(properties.get("file_path") or ""),
                        "file_paths": list(_list_value(properties.get("file_paths"))),
                        "extraction_format": str(
                            properties.get("extraction_format") or ""
                        ),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def _embed_graph_edges(
    embedding_model: EmbeddingModelProtocol,
    edges: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for batch in batches(edges, batch_size):
        texts = [_edge_vector_text(edge) for edge in batch]
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={
                    "step": BuildLightRAGGraph.name,
                    "purpose": "light_rag_relationship_index",
                },
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("edge embedding result count must match batch size")
        for edge, text, vector in zip(batch, texts, result.vectors, strict=True):
            properties = dict(edge.get("properties") or {})
            records.append(
                VectorRecord(
                    id=str(edge["id"]),
                    vector=[float(value) for value in vector],
                    text=text,
                    metadata={
                        "fact_type": "light_rag_relationship",
                        "src_id": str(properties.get("src_id") or edge["source_id"]),
                        "tgt_id": str(properties.get("tgt_id") or edge["target_id"]),
                        "keywords": str(properties.get("keywords") or ""),
                        "description": str(properties.get("description") or ""),
                        "weight": float(properties.get("weight") or 1.0),
                        "source_chunk_ids": list(_list_value(properties.get("source_ids"))),
                        "source_ids": list(_list_value(properties.get("source_ids"))),
                        "file_path": str(properties.get("file_path") or ""),
                        "file_paths": list(_list_value(properties.get("file_paths"))),
                        "extraction_format": str(
                            properties.get("extraction_format") or ""
                        ),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def _embed_chunks(
    embedding_model: EmbeddingModelProtocol,
    chunks: list[ParsedChunk],
    *,
    batch_size: int,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for batch in batches(chunks, batch_size):
        texts = [_chunk_vector_text(chunk) for chunk in batch]
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={
                    "step": BuildLightRAGGraph.name,
                    "purpose": "light_rag_chunk_index",
                },
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("chunk embedding result count must match batch size")
        for chunk, text, vector in zip(batch, texts, result.vectors, strict=True):
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=[float(value) for value in vector],
                    text=text,
                    metadata={
                        "fact_type": "light_rag_chunk",
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "source_key": chunk.source.key,
                        "source_name": chunk.source.name,
                        "source_file_type": chunk.source.file_type,
                        "page_index": chunk.page_index,
                        "chunk_index": chunk.chunk_index,
                        "token_start": chunk.token_start,
                        "token_end": chunk.token_end,
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


def _entity_row(node: dict[str, Any]) -> dict[str, object]:
    properties = dict(node.get("properties") or {})
    source_ids = list(_list_value(properties.get("source_ids")))
    file_paths = list(_list_value(properties.get("file_paths")))
    return {
        "entity_id": str(node["id"]),
        "entity_name": str(properties.get("entity_name") or properties.get("name") or node["id"]),
        "entity_type": str(properties.get("entity_type") or "unknown"),
        "description": str(properties.get("description") or ""),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "file_path": str(properties.get("file_path") or ""),
        "file_paths": compact_json(file_paths),
        "properties": compact_json(properties),
    }


def _relation_row(edge: dict[str, Any]) -> dict[str, object]:
    properties = dict(edge.get("properties") or {})
    source_ids = list(_list_value(properties.get("source_ids")))
    file_paths = list(_list_value(properties.get("file_paths")))
    return {
        "relation_id": str(edge["id"]),
        "source_entity_id": str(properties.get("src_id") or edge["source_id"]),
        "target_entity_id": str(properties.get("tgt_id") or edge["target_id"]),
        "description": str(properties.get("description") or ""),
        "keywords": str(properties.get("keywords") or ""),
        "weight": float(properties.get("weight") or 1.0),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "file_path": str(properties.get("file_path") or ""),
        "file_paths": compact_json(file_paths),
        "properties": compact_json(properties),
    }


def _chunk_row(chunk: ParsedChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.text,
        "source_key": chunk.source.key,
        "source_name": chunk.source.name,
        "source_file_type": chunk.source.file_type,
        "page_index": chunk.page_index,
        "chunk_index": chunk.chunk_index,
        "token_start": chunk.token_start,
        "token_end": chunk.token_end,
        "metadata": compact_json(
            {
                "parent_chunk_ids": list(chunk.parent_chunk_ids),
                "source_content_sha256": chunk.source.content_sha256,
            }
        ),
    }


def _node_vector_text(node: dict[str, Any]) -> str:
    properties = dict(node.get("properties") or {})
    return "\n".join(
        value
        for value in (
            str(properties.get("entity_name") or properties.get("name") or node["id"]),
            str(properties.get("entity_type") or "unknown"),
            str(properties.get("description") or ""),
        )
        if value
    )


def _edge_vector_text(edge: dict[str, Any]) -> str:
    properties = dict(edge.get("properties") or {})
    return "\n".join(
        value
        for value in (
            str(properties.get("src_id") or edge["source_id"]),
            str(properties.get("tgt_id") or edge["target_id"]),
            str(properties.get("keywords") or ""),
            str(properties.get("description") or ""),
        )
        if value
    )


def _chunk_vector_text(chunk: ParsedChunk) -> str:
    return chunk.text


def _vector_dimension(
    entity_vectors: list[VectorRecord],
    relationship_vectors: list[VectorRecord],
    chunk_vectors: list[VectorRecord],
) -> int:
    if entity_vectors:
        return len(entity_vectors[0].vector)
    if relationship_vectors:
        return len(relationship_vectors[0].vector)
    if chunk_vectors:
        return len(chunk_vectors[0].vector)
    return 0


def _list_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item for item in value.split("<SEP>") if item)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
