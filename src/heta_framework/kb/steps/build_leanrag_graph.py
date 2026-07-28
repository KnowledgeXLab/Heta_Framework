"""Build LeanRAG SQL tables, vector index, and graph-store records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorCollectionConfig, VectorRecord, VectorStoreProtocol
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.graph_storage import batches, compact_json, validate_identifier
from heta_framework.kb.steps.extract_leanrag_graph import (
    export_all_entities_json_lines,
    export_communities_json_lines,
    export_generated_relations_json_lines,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class LeanRAGTableNames:
    """SQL table names used by LeanRAG storage."""

    entities: str = "leanrag_entities"
    relations: str = "leanrag_relations"
    communities: str = "leanrag_communities"
    chunks: str = "leanrag_chunks"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="table_names.entities")
        validate_identifier(self.relations, field_name="table_names.relations")
        validate_identifier(self.communities, field_name="table_names.communities")
        validate_identifier(self.chunks, field_name="table_names.chunks")


@dataclass(frozen=True)
class LeanRAGVectorCollections:
    """Vector collection names used by LeanRAG search."""

    entities: str = "leanrag_entities"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="vector_collections.entities")


@dataclass(frozen=True)
class BuildLeanRAGGraphConfig:
    """Configuration for BuildLeanRAGGraph."""

    table_names: LeanRAGTableNames = field(default_factory=LeanRAGTableNames)
    vector_collections: LeanRAGVectorCollections = field(default_factory=LeanRAGVectorCollections)
    chunks_artifact: str = "lean_rag_chunks"
    base_relations_artifact: str = "lean_rag_base_relations"
    all_entities_layers_artifact: str = "lean_rag_all_entities_layers"
    generated_relations_artifact: str = "lean_rag_generated_relations"
    communities_artifact: str = "lean_rag_communities"
    parent_edges_artifact: str = "lean_rag_parent_edges"
    result_artifact: str = "build_lean_rag_graph_result"
    all_entities_export_artifact: str = "lean_rag_all_entities_json"
    generated_relations_export_artifact: str = "lean_rag_generate_relations_json"
    communities_export_artifact: str = "lean_rag_community_json"
    vector_metric: str = "cosine"
    batch_size: int = 128
    graph_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        for name in (
            self.chunks_artifact,
            self.base_relations_artifact,
            self.all_entities_layers_artifact,
            self.generated_relations_artifact,
            self.communities_artifact,
            self.parent_edges_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class BuildLeanRAGGraphResult:
    """Artifacts produced by BuildLeanRAGGraph."""

    entity_count: int
    relation_count: int
    community_count: int
    chunk_count: int
    entity_vector_count: int
    vector_dimension: int


class BuildLeanRAGGraph:
    """Write LeanRAG artifacts into HetaFramework stores."""

    name = "build_leanrag_graph"

    def __init__(self, config: BuildLeanRAGGraphConfig | None = None) -> None:
        self.config = config or BuildLeanRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("graph", self.config.graph_store),
                    store_ref("sql", self.config.sql_store),
                    store_ref("vector", self.config.vector_store),
                    model_ref("embedding", self.config.embedding_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.chunks_artifact,
                    self.config.base_relations_artifact,
                    self.config.all_entities_layers_artifact,
                    self.config.generated_relations_artifact,
                    self.config.communities_artifact,
                    self.config.parent_edges_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        sql_store_ref = store_ref("sql", self.config.sql_store)
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.all_entities_export_artifact,
                    self.config.generated_relations_export_artifact,
                    self.config.communities_export_artifact,
                }
            ),
            queries=frozenset({"lean_rag_query"}),
            search_assets=(
                SearchAsset(
                    kind="leanrag_tables",
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
                    kind="leanrag_vector_index",
                    name=self.config.vector_collections.entities,
                    store=vector_store_ref.key,
                    metadata={"entity_collection": self.config.vector_collections.entities},
                ),
            ),
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        sql_store_ref = store_ref("sql", self.config.sql_store).key
        vector_store_ref = store_ref("vector", self.config.vector_store).key
        return StepCleanupPlan(
            (
                CleanupTarget("sql_table", self.config.table_names.entities, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.relations, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.communities, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.chunks, sql_store_ref),
                CleanupTarget("vector_collection", self.config.vector_collections.entities, vector_store_ref),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        graph_store = _require_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
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

        chunks = [dict(item) for item in context.get_artifact(self.config.chunks_artifact)]
        all_entities_layers = list(context.get_artifact(self.config.all_entities_layers_artifact))
        base_relations = [dict(item) for item in context.get_artifact(self.config.base_relations_artifact)]
        generated_relations = [dict(item) for item in context.get_artifact(self.config.generated_relations_artifact)]
        communities = [dict(item) for item in context.get_artifact(self.config.communities_artifact)]
        parent_edges = [dict(item) for item in context.get_artifact(self.config.parent_edges_artifact)]

        entities = flatten_all_entities_layers(all_entities_layers)
        relations = _dedupe_relations([*_normalize_base_relations(base_relations), *generated_relations])
        entity_rows = [_entity_row(entity) for entity in entities]
        relation_rows = [_relation_row(relation) for relation in relations]
        community_rows = [_community_row(community) for community in communities]
        chunk_rows = [_chunk_row(chunk) for chunk in chunks]
        vectors = await _embed_entities(embedding_model, entities, batch_size=self.config.batch_size)
        vector_dimension = len(vectors[0].vector) if vectors else 0

        await _upsert_graph_store(graph_store, entities, relations, parent_edges)
        async with sql_store.transaction() as tx:
            await _ensure_tables(tx, self.config.table_names)
            for batch in batches(entity_rows, self.config.batch_size):
                await _upsert_entity_rows(tx, self.config.table_names.entities, batch)
            for batch in batches(relation_rows, self.config.batch_size):
                await _upsert_relation_rows(tx, self.config.table_names.relations, batch)
            for batch in batches(community_rows, self.config.batch_size):
                await _upsert_community_rows(tx, self.config.table_names.communities, batch)
            for batch in batches(chunk_rows, self.config.batch_size):
                await _upsert_chunk_rows(tx, self.config.table_names.chunks, batch)

        if vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.entities,
                    dimension=vector_dimension,
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(vectors, self.config.batch_size):
                await vector_store.upsert(self.config.vector_collections.entities, batch)

        context.set_artifact(
            self.config.result_artifact,
            BuildLeanRAGGraphResult(
                entity_count=len(entities),
                relation_count=len(relations),
                community_count=len(communities),
                chunk_count=len(chunks),
                entity_vector_count=len(vectors),
                vector_dimension=vector_dimension,
            ),
        )
        context.set_artifact(
            self.config.all_entities_export_artifact,
            export_all_entities_json_lines(all_entities_layers),
        )
        context.set_artifact(
            self.config.generated_relations_export_artifact,
            export_generated_relations_json_lines(generated_relations),
        )
        context.set_artifact(
            self.config.communities_export_artifact,
            export_communities_json_lines(communities),
        )


def flatten_all_entities_layers(all_entities_layers: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(all_entities_layers):
        if isinstance(layer, list):
            for item in layer:
                row = dict(item)
                row.setdefault("level", layer_index)
                rows.append(row)
        elif isinstance(layer, dict):
            row = dict(layer)
            row.setdefault("level", layer_index)
            rows.append(row)
    return rows


async def _ensure_tables(tx: SQLStoreProtocol, tables: LeanRAGTableNames) -> None:
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.entities} (
            entity_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            degree INTEGER NOT NULL,
            parent TEXT NOT NULL,
            level INTEGER NOT NULL,
            is_aggregate INTEGER NOT NULL,
            children TEXT NOT NULL,
            findings TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.relations} (
            relation_id TEXT PRIMARY KEY,
            src_tgt TEXT NOT NULL,
            tgt_src TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL NOT NULL,
            level INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            is_generated INTEGER NOT NULL,
            evidence_relation_ids TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.communities} (
            community_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_description TEXT NOT NULL,
            findings TEXT NOT NULL,
            level INTEGER NOT NULL,
            children TEXT NOT NULL,
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
        CREATE TABLE IF NOT EXISTS {tables.chunks} (
            chunk_id TEXT PRIMARY KEY,
            hash_code TEXT NOT NULL,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            content TEXT NOT NULL,
            source_key TEXT,
            source_name TEXT,
            source_file_type TEXT,
            chunk_order_index INTEGER NOT NULL,
            token_start INTEGER,
            token_end INTEGER,
            token_count INTEGER,
            metadata TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


async def _upsert_entity_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                entity_id, entity_name, entity_type, description, source_id,
                source_ids, degree, parent, level, is_aggregate, children,
                findings, properties, updated_at
            )
            VALUES (
                :entity_id, :entity_name, :entity_type, :description, :source_id,
                :source_ids, :degree, :parent, :level, :is_aggregate, :children,
                :findings, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_type = excluded.entity_type,
                description = excluded.description,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                degree = excluded.degree,
                parent = excluded.parent,
                level = excluded.level,
                is_aggregate = excluded.is_aggregate,
                children = excluded.children,
                findings = excluded.findings,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_relation_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                relation_id, src_tgt, tgt_src, description, weight, level,
                source_id, source_ids, is_generated, evidence_relation_ids,
                properties, updated_at
            )
            VALUES (
                :relation_id, :src_tgt, :tgt_src, :description, :weight, :level,
                :source_id, :source_ids, :is_generated, :evidence_relation_ids,
                :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (relation_id) DO UPDATE SET
                src_tgt = excluded.src_tgt,
                tgt_src = excluded.tgt_src,
                description = excluded.description,
                weight = excluded.weight,
                level = excluded.level,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                is_generated = excluded.is_generated,
                evidence_relation_ids = excluded.evidence_relation_ids,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_community_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                community_id, entity_name, entity_description, findings, level,
                children, source_id, source_ids, properties, updated_at
            )
            VALUES (
                :community_id, :entity_name, :entity_description, :findings, :level,
                :children, :source_id, :source_ids, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (community_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_description = excluded.entity_description,
                findings = excluded.findings,
                level = excluded.level,
                children = excluded.children,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_chunk_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                chunk_id, hash_code, document_id, text, content, source_key,
                source_name, source_file_type, chunk_order_index, token_start,
                token_end, token_count, metadata, updated_at
            )
            VALUES (
                :chunk_id, :hash_code, :document_id, :text, :content, :source_key,
                :source_name, :source_file_type, :chunk_order_index, :token_start,
                :token_end, :token_count, :metadata, CURRENT_TIMESTAMP
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                hash_code = excluded.hash_code,
                document_id = excluded.document_id,
                text = excluded.text,
                content = excluded.content,
                source_key = excluded.source_key,
                source_name = excluded.source_name,
                source_file_type = excluded.source_file_type,
                chunk_order_index = excluded.chunk_order_index,
                token_start = excluded.token_start,
                token_end = excluded.token_end,
                token_count = excluded.token_count,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _embed_entities(
    embedding_model: EmbeddingModelProtocol,
    entities: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for batch in batches(entities, batch_size):
        texts = [_entity_vector_text(entity) for entity in batch]
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={"step": BuildLeanRAGGraph.name, "purpose": "leanrag_entity_index"},
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("entity embedding result count must match batch size")
        for entity, text, vector in zip(batch, texts, result.vectors, strict=True):
            records.append(
                VectorRecord(
                    id=_entity_id(entity),
                    vector=[float(value) for value in vector],
                    text=text,
                    metadata={
                        "fact_type": "leanrag_entity",
                        "entity_name": str(entity.get("entity_name") or ""),
                        "entity_type": str(entity.get("entity_type") or "UNKNOWN"),
                        "description": str(entity.get("description") or ""),
                        "source_id": str(entity.get("source_id") or ""),
                        "source_ids": _source_ids(entity),
                        "parent": str(entity.get("parent") or ""),
                        "level": int(entity.get("level") or 0),
                        "is_aggregate": bool(entity.get("is_aggregate")),
                        "children": _list_value(entity.get("children")),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def _upsert_graph_store(
    graph_store: GraphStoreProtocol,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    parent_edges: list[dict[str, Any]],
) -> None:
    nodes = [
        GraphNode(
            id=_entity_id(entity),
            labels=("Entity", str(entity.get("entity_type") or "UNKNOWN")),
            properties={**entity, "name": str(entity.get("entity_name") or "")},
        )
        for entity in entities
    ]
    if any(edge.get("tgt_src") == "root" for edge in parent_edges):
        nodes.append(GraphNode(id="root", labels=("Root",), properties={"name": "root"}))
    await graph_store.upsert_nodes(nodes)

    graph_edges: list[GraphEdge] = []
    for relation in relations:
        source = str(relation.get("src_tgt") or "")
        target = str(relation.get("tgt_src") or "")
        if source and target and source != target:
            graph_edges.append(
                GraphEdge(
                    id=_relation_id(relation),
                    source_id=source,
                    target_id=target,
                    type="RELATED",
                    properties=dict(relation),
                )
            )
    for edge in parent_edges:
        source = str(edge.get("src_tgt") or "")
        target = str(edge.get("tgt_src") or "")
        if source and target and source != target:
            graph_edges.append(
                GraphEdge(
                    id=_stable_id("parent", source, target, str(edge.get("level", 0))),
                    source_id=source,
                    target_id=target,
                    type="LEANRAG_PARENT",
                    properties=dict(edge),
                )
            )
    if graph_edges:
        await graph_store.upsert_edges(graph_edges)


def _entity_row(entity: Mapping[str, Any]) -> dict[str, object]:
    return {
        "entity_id": _entity_id(entity),
        "entity_name": str(entity.get("entity_name") or ""),
        "entity_type": str(entity.get("entity_type") or "UNKNOWN"),
        "description": str(entity.get("description") or ""),
        "source_id": str(entity.get("source_id") or ""),
        "source_ids": compact_json(_source_ids(entity)),
        "degree": int(entity.get("degree") or 0),
        "parent": str(entity.get("parent") or ""),
        "level": int(entity.get("level") or 0),
        "is_aggregate": 1 if bool(entity.get("is_aggregate")) else 0,
        "children": compact_json(_list_value(entity.get("children"))),
        "findings": compact_json(entity.get("findings") or []),
        "properties": compact_json(dict(entity.get("properties") or {})),
    }


def _relation_row(relation: Mapping[str, Any]) -> dict[str, object]:
    return {
        "relation_id": _relation_id(relation),
        "src_tgt": str(relation.get("src_tgt") or ""),
        "tgt_src": str(relation.get("tgt_src") or ""),
        "description": str(relation.get("description") or ""),
        "weight": float(relation.get("weight") or 1.0),
        "level": int(relation.get("level") or 0),
        "source_id": str(relation.get("source_id") or ""),
        "source_ids": compact_json(_source_ids(relation)),
        "is_generated": 1 if bool(relation.get("is_generated")) else 0,
        "evidence_relation_ids": compact_json(_list_value(relation.get("evidence_relation_ids"))),
        "properties": compact_json(dict(relation.get("properties") or {})),
    }


def _community_row(community: Mapping[str, Any]) -> dict[str, object]:
    return {
        "community_id": _stable_id("community", str(community.get("entity_name") or "")),
        "entity_name": str(community.get("entity_name") or ""),
        "entity_description": str(community.get("entity_description") or community.get("description") or ""),
        "findings": compact_json(community.get("findings") or []),
        "level": int(community.get("level") or 0),
        "children": compact_json(_list_value(community.get("children"))),
        "source_id": str(community.get("source_id") or ""),
        "source_ids": compact_json(_source_ids(community)),
        "properties": compact_json(dict(community.get("properties") or {})),
    }


def _chunk_row(chunk: Mapping[str, Any]) -> dict[str, object]:
    text = str(chunk.get("text") or chunk.get("content") or "")
    return {
        "chunk_id": str(chunk.get("chunk_id") or chunk.get("hash_code") or ""),
        "hash_code": str(chunk.get("hash_code") or hashlib.md5(text.encode()).hexdigest()),
        "document_id": str(chunk.get("document_id") or ""),
        "text": text,
        "content": str(chunk.get("content") or text),
        "source_key": str(chunk.get("source_key") or ""),
        "source_name": str(chunk.get("source_name") or ""),
        "source_file_type": str(chunk.get("source_file_type") or ""),
        "chunk_order_index": int(chunk.get("chunk_order_index") or chunk.get("chunk_index") or 0),
        "token_start": int(chunk.get("token_start") or 0),
        "token_end": int(chunk.get("token_end") or 0),
        "token_count": int(chunk.get("token_count") or 0),
        "metadata": compact_json(dict(chunk.get("metadata") or {})),
    }


def _entity_vector_text(entity: Mapping[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            str(entity.get("entity_name") or ""),
            str(entity.get("description") or ""),
        )
        if value
    )


def _normalize_base_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**relation, "is_generated": False, "level": int(relation.get("level") or 0)} for relation in relations]


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for relation in relations:
        by_id[_relation_id(relation)] = relation
    return list(by_id.values())


def _entity_id(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_name") or "")


def _relation_id(relation: Mapping[str, Any]) -> str:
    return _stable_id(
        "relation",
        str(relation.get("src_tgt") or ""),
        str(relation.get("tgt_src") or ""),
        str(relation.get("level") or 0),
        "generated" if relation.get("is_generated") else "base",
    )


def _stable_id(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _source_ids(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    source_ids = record.get("source_ids", ())
    if isinstance(source_ids, str):
        values.extend(source_ids.split("|"))
    elif isinstance(source_ids, list | tuple):
        values.extend(str(item) for item in source_ids)
    source_id = str(record.get("source_id") or "")
    if source_id:
        values.extend(source_id.split("|"))
    return list(dict.fromkeys(value for value in values if value))


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    return [value]


def _require_graph_store(component: object) -> GraphStoreProtocol:
    if not isinstance(component, GraphStoreProtocol):
        raise TypeError("stores.graph must satisfy GraphStoreProtocol")
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
