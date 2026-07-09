"""Build GraphRAG SQL tables and vector indexes from persisted graph artifacts."""

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
class RAGGraphTableNames:
    """SQL table names used by GraphRAG storage."""

    entities: str = "rag_entities"
    relations: str = "rag_relations"
    communities: str = "rag_communities"
    chunks: str = "rag_chunks"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="table_names.entities")
        validate_identifier(self.relations, field_name="table_names.relations")
        validate_identifier(self.communities, field_name="table_names.communities")
        validate_identifier(self.chunks, field_name="table_names.chunks")


@dataclass(frozen=True)
class RAGGraphVectorCollections:
    """Vector collection names used by GraphRAG storage."""

    entities: str = "rag_graph_entities"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="vector_collections.entities")


@dataclass(frozen=True)
class BuildRAGGraphConfig:
    """Configuration for BuildRAGGraph."""

    table_names: RAGGraphTableNames = field(default_factory=RAGGraphTableNames)
    vector_collections: RAGGraphVectorCollections = field(
        default_factory=RAGGraphVectorCollections
    )
    graph_node_keys_artifact: str = "graph_node_keys"
    graph_edge_keys_artifact: str = "graph_edge_keys"
    community_report_keys_artifact: str = "community_report_keys"
    chunk_keys_artifact: str = "chunk_keys"
    vector_metric: str = "cosine"
    batch_size: int = 128
    object_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.graph_node_keys_artifact.strip() == "":
            raise ValueError("graph_node_keys_artifact must not be empty")
        if self.graph_edge_keys_artifact.strip() == "":
            raise ValueError("graph_edge_keys_artifact must not be empty")
        if self.community_report_keys_artifact.strip() == "":
            raise ValueError("community_report_keys_artifact must not be empty")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


@dataclass(frozen=True)
class BuildRAGGraphResult:
    """Artifacts produced by BuildRAGGraph."""

    entity_count: int
    relation_count: int
    community_count: int
    chunk_count: int
    entity_vector_count: int
    vector_dimension: int


class BuildRAGGraph:
    """Write GraphRAG graph artifacts into SQL and vector stores."""

    name = "build_rag_graph"

    def __init__(self, config: BuildRAGGraphConfig | None = None) -> None:
        self.config = config or BuildRAGGraphConfig()

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
                    self.config.community_report_keys_artifact,
                    self.config.chunk_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store)
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset({"build_rag_graph_result"}),
            queries=frozenset({"graph_rag_local_query", "graph_rag_global_query"}),
            search_assets=(
                SearchAsset(
                    kind="rag_graph_tables",
                    name=self.config.table_names.entities,
                    store=sql_store_ref.key,
                    metadata={
                        "entities_table": self.config.table_names.entities,
                        "relations_table": self.config.table_names.relations,
                        "communities_table": self.config.table_names.communities,
                        "chunks_table": self.config.table_names.chunks,
                    },
                ),
                SearchAsset(
                    kind="rag_graph_vector_index",
                    name=self.config.vector_collections.entities,
                    store=vector_store_ref.key,
                    metadata={
                        "entity_collection": self.config.vector_collections.entities,
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
                    value=self.config.table_names.communities,
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
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the GraphRAG build step."""
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
        report_keys = tuple(context.get_artifact(self.config.community_report_keys_artifact))
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))

        nodes = [json.loads((await object_store.get(key)).decode("utf-8")) for key in node_keys]
        edges = [json.loads((await object_store.get(key)).decode("utf-8")) for key in edge_keys]
        reports = [
            json.loads((await object_store.get(key)).decode("utf-8")) for key in report_keys
        ]
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        node_rows = [_entity_row(node) for node in nodes]
        edge_rows = [_relation_row(edge) for edge in edges]
        community_rows = [_community_row(report) for report in reports]
        chunk_rows = [_chunk_row(chunk) for chunk in chunks]
        entity_vectors = await _embed_graph_nodes(
            embedding_model,
            nodes,
            batch_size=self.config.batch_size,
        )
        vector_dimension = len(entity_vectors[0].vector) if entity_vectors else 0

        async with sql_store.transaction() as tx:
            await _ensure_rag_graph_tables(tx, self.config.table_names)
            for batch in batches(node_rows, self.config.batch_size):
                await _upsert_entity_rows(tx, self.config.table_names.entities, batch)
            for batch in batches(edge_rows, self.config.batch_size):
                await _upsert_relation_rows(tx, self.config.table_names.relations, batch)
            for batch in batches(community_rows, self.config.batch_size):
                await _upsert_community_rows(
                    tx,
                    self.config.table_names.communities,
                    batch,
                )
            for batch in batches(chunk_rows, self.config.batch_size):
                await _upsert_chunk_rows(tx, self.config.table_names.chunks, batch)

        if entity_vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.entities,
                    dimension=vector_dimension,
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(entity_vectors, self.config.batch_size):
                await vector_store.upsert(self.config.vector_collections.entities, batch)

        result = BuildRAGGraphResult(
            entity_count=len(nodes),
            relation_count=len(edges),
            community_count=len(reports),
            chunk_count=len(chunks),
            entity_vector_count=len(entity_vectors),
            vector_dimension=vector_dimension,
        )
        context.set_artifact("build_rag_graph_result", result)


async def _ensure_rag_graph_tables(
    tx: SQLStoreProtocol,
    table_names: RAGGraphTableNames,
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
            relation_type TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.communities} (
            community_id TEXT PRIMARY KEY,
            level INTEGER NOT NULL,
            title TEXT NOT NULL,
            report TEXT NOT NULL,
            report_json TEXT NOT NULL,
            nodes TEXT NOT NULL,
            edges TEXT NOT NULL,
            chunk_ids TEXT NOT NULL,
            occurrence REAL NOT NULL,
            sub_communities TEXT NOT NULL,
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
                source_ids, properties, updated_at
            )
            VALUES (
                :entity_id, :entity_name, :entity_type, :description, :source_id,
                :source_ids, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_type = excluded.entity_type,
                description = excluded.description,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
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
                relation_id, source_entity_id, target_entity_id, relation_type,
                description, weight, source_id, source_ids, properties, updated_at
            )
            VALUES (
                :relation_id, :source_entity_id, :target_entity_id, :relation_type,
                :description, :weight, :source_id, :source_ids, :properties,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (relation_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                target_entity_id = excluded.target_entity_id,
                relation_type = excluded.relation_type,
                description = excluded.description,
                weight = excluded.weight,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_community_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                community_id, level, title, report, report_json, nodes, edges,
                chunk_ids, occurrence, sub_communities, updated_at
            )
            VALUES (
                :community_id, :level, :title, :report, :report_json, :nodes, :edges,
                :chunk_ids, :occurrence, :sub_communities, CURRENT_TIMESTAMP
            )
            ON CONFLICT (community_id) DO UPDATE SET
                level = excluded.level,
                title = excluded.title,
                report = excluded.report,
                report_json = excluded.report_json,
                nodes = excluded.nodes,
                edges = excluded.edges,
                chunk_ids = excluded.chunk_ids,
                occurrence = excluded.occurrence,
                sub_communities = excluded.sub_communities,
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
                trace_context={"step": BuildRAGGraph.name, "purpose": "rag_entity_index"},
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
                        "fact_type": "rag_entity",
                        "entity_name": str(node["id"]),
                        "entity_type": str(properties.get("entity_type") or "ENTITY"),
                        "source_ids": list(_list_value(properties.get("source_ids"))),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


def _entity_row(node: dict[str, Any]) -> dict[str, object]:
    properties = dict(node.get("properties") or {})
    source_ids = list(_list_value(properties.get("source_ids")))
    return {
        "entity_id": str(node["id"]),
        "entity_name": str(properties.get("name") or node["id"]),
        "entity_type": str(properties.get("entity_type") or "ENTITY"),
        "description": str(properties.get("description") or ""),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "properties": compact_json(properties),
    }


def _relation_row(edge: dict[str, Any]) -> dict[str, object]:
    properties = dict(edge.get("properties") or {})
    source_ids = list(_list_value(properties.get("source_ids")))
    return {
        "relation_id": str(edge["id"]),
        "source_entity_id": str(edge["source_id"]),
        "target_entity_id": str(edge["target_id"]),
        "relation_type": str(edge.get("type") or "RELATED"),
        "description": str(properties.get("description") or ""),
        "weight": float(properties.get("weight") or 0.0),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "properties": compact_json(properties),
    }


def _community_row(report: dict[str, Any]) -> dict[str, object]:
    return {
        "community_id": str(report["community_id"]),
        "level": int(report["level"]),
        "title": str(report["title"]),
        "report": str(report.get("report") or ""),
        "report_json": compact_json(report.get("report_json") or {}),
        "nodes": compact_json(report.get("nodes") or []),
        "edges": compact_json(report.get("edges") or []),
        "chunk_ids": compact_json(report.get("chunk_ids") or []),
        "occurrence": float(report.get("occurrence") or 0.0),
        "sub_communities": compact_json(report.get("sub_communities") or []),
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
            str(properties.get("name") or node["id"]),
            str(properties.get("entity_type") or "ENTITY"),
            str(properties.get("description") or ""),
            " ".join(str(item) for item in _list_value(properties.get("source_ids"))),
        )
        if value
    )


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
