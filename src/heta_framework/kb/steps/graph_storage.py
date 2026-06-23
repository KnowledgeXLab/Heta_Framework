"""Storage helpers for Heta-style graph facts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypeVar

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import (
    VectorCollectionConfig,
    VectorRecord,
    VectorStoreProtocol,
)
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation
from heta_framework.kb.steps.types import IssueResolution, IssueSubject, StepIssue

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_T = TypeVar("_T")


@dataclass(frozen=True)
class GraphTableNames:
    """SQL table names used by Heta-style graph storage."""

    entities: str = "entities"
    relations: str = "relations"
    evidence: str = "graph_evidence"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="table_names.entities")
        validate_identifier(self.relations, field_name="table_names.relations")
        validate_identifier(self.evidence, field_name="table_names.evidence")


@dataclass(frozen=True)
class GraphVectorCollections:
    """Vector collection names used by Heta-style graph search."""

    entities: str = "graph_entities"
    relations: str = "graph_relations"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="vector_collections.entities")
        validate_identifier(self.relations, field_name="vector_collections.relations")


@dataclass(frozen=True)
class GraphStorageConfig:
    """Storage settings shared by graph build and merge steps."""

    table_names: GraphTableNames
    vector_collections: GraphVectorCollections
    vector_metric: str
    batch_size: int
    trace_step: str


def entity_row(entity: ExtractedEntity) -> dict[str, object]:
    """Return the SQL row for one entity fact."""
    return {
        "entity_id": entity.entity_id,
        "entity_name": entity.name,
        "entity_type": entity.type,
        "entity_subtype": entity.subtype,
        "description": entity.description,
        "attributes": compact_json(entity.attributes),
    }


def relation_row(relation: ExtractedRelation) -> dict[str, object]:
    """Return the SQL row for one relation fact."""
    return {
        "relation_id": relation.relation_id,
        "source_entity_id": relation.source_entity_id,
        "target_entity_id": relation.target_entity_id,
        "source_entity_name": relation.source_entity_name,
        "target_entity_name": relation.target_entity_name,
        "relation_type": relation.type,
        "relation_name": relation.name,
        "description": relation.description,
        "attributes": compact_json(relation.attributes),
    }


def entity_vector_text(entity: ExtractedEntity) -> str:
    """Return the canonical text embedded for entity graph search."""
    attributes = " ".join(f"{key}:{value}" for key, value in sorted(entity.attributes.items()))
    return "\n".join(
        value
        for value in (
            entity.name,
            entity.type,
            entity.subtype or "",
            entity.description,
            attributes,
        )
        if value
    )


def relation_vector_text(relation: ExtractedRelation) -> str:
    """Return the canonical text embedded for relation graph search."""
    attributes = " ".join(f"{key}:{value}" for key, value in sorted(relation.attributes.items()))
    return "\n".join(
        value
        for value in (
            f"{relation.source_entity_name} -> {relation.target_entity_name}",
            relation.type,
            relation.name,
            relation.description,
            attributes,
        )
        if value
    )


async def embed_entity_records(
    embedding_model: EmbeddingModelProtocol,
    entities: list[ExtractedEntity],
    config: GraphStorageConfig,
) -> list[VectorRecord]:
    """Embed entity facts into vector-store records."""
    records: list[VectorRecord] = []
    for batch in batches(entities, config.batch_size):
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=[entity_vector_text(entity) for entity in batch],
                trace_context={"step": config.trace_step, "purpose": "entity_graph_index"},
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("entity embedding result count must match batch size")
        for entity, vector in zip(batch, result.vectors, strict=True):
            records.append(
                VectorRecord(
                    id=entity.entity_id,
                    vector=[float(value) for value in vector],
                    text=entity_vector_text(entity),
                    metadata={
                        "fact_type": "entity",
                        "entity_name": entity.name,
                        "entity_type": entity.type,
                        "entity_subtype": entity.subtype,
                        "document_id": entity.document_id,
                        "source_chunk_ids": list(entity.source_chunk_ids),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def embed_relation_records(
    embedding_model: EmbeddingModelProtocol,
    relations: list[ExtractedRelation],
    config: GraphStorageConfig,
) -> list[VectorRecord]:
    """Embed relation facts into vector-store records."""
    records: list[VectorRecord] = []
    for batch in batches(relations, config.batch_size):
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=[relation_vector_text(relation) for relation in batch],
                trace_context={"step": config.trace_step, "purpose": "relation_graph_index"},
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("relation embedding result count must match batch size")
        for relation, vector in zip(batch, result.vectors, strict=True):
            records.append(
                VectorRecord(
                    id=relation.relation_id,
                    vector=[float(value) for value in vector],
                    text=relation_vector_text(relation),
                    metadata={
                        "fact_type": "relation",
                        "source_entity_id": relation.source_entity_id,
                        "target_entity_id": relation.target_entity_id,
                        "source_entity_name": relation.source_entity_name,
                        "target_entity_name": relation.target_entity_name,
                        "relation_type": relation.type,
                        "relation_name": relation.name,
                        "document_id": relation.document_id,
                        "source_chunk_ids": list(relation.source_chunk_ids),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def upsert_graph_vectors(
    vector_store: VectorStoreProtocol,
    entity_vectors: list[VectorRecord],
    relation_vectors: list[VectorRecord],
    config: GraphStorageConfig,
) -> None:
    """Upsert entity and relation graph vectors."""
    if entity_vectors:
        await vector_store.create_collection(
            VectorCollectionConfig(
                name=config.vector_collections.entities,
                dimension=len(entity_vectors[0].vector),
                metric=config.vector_metric,  # type: ignore[arg-type]
            )
        )
        for batch in batches(entity_vectors, config.batch_size):
            await vector_store.upsert(config.vector_collections.entities, batch)
    if relation_vectors:
        await vector_store.create_collection(
            VectorCollectionConfig(
                name=config.vector_collections.relations,
                dimension=len(relation_vectors[0].vector),
                metric=config.vector_metric,  # type: ignore[arg-type]
            )
        )
        for batch in batches(relation_vectors, config.batch_size):
            await vector_store.upsert(config.vector_collections.relations, batch)


def graph_vector_dimension(
    entity_vectors: list[VectorRecord],
    relation_vectors: list[VectorRecord],
) -> int:
    """Return the vector dimension used by graph vectors."""
    if entity_vectors:
        return len(entity_vectors[0].vector)
    if relation_vectors:
        return len(relation_vectors[0].vector)
    return 0


def evidence_rows_for_entity(
    entity: ExtractedEntity,
    chunk_sources: dict[str, ParsedChunk],
    issues: list[StepIssue],
    *,
    step_name: str,
) -> list[dict[str, object]]:
    """Return evidence rows for an entity, reporting missing chunks."""
    rows: list[dict[str, object]] = []
    for chunk_id in entity.source_chunk_ids:
        chunk = chunk_sources.get(chunk_id)
        if chunk is None:
            issues.append(missing_chunk_issue(step_name, fact_id=entity.entity_id, fact_type="entity", chunk_id=chunk_id))
            continue
        rows.append(evidence_row(entity.entity_id, "entity", chunk))
    return rows


def evidence_rows_for_relation(
    relation: ExtractedRelation,
    chunk_sources: dict[str, ParsedChunk],
    issues: list[StepIssue],
    *,
    step_name: str,
) -> list[dict[str, object]]:
    """Return evidence rows for a relation, reporting missing chunks."""
    rows: list[dict[str, object]] = []
    for chunk_id in relation.source_chunk_ids:
        chunk = chunk_sources.get(chunk_id)
        if chunk is None:
            issues.append(
                missing_chunk_issue(
                    step_name,
                    fact_id=relation.relation_id,
                    fact_type="relation",
                    chunk_id=chunk_id,
                )
            )
            continue
        rows.append(evidence_row(relation.relation_id, "relation", chunk))
    return rows


def available_evidence_rows_for_entity(
    entity: ExtractedEntity,
    chunk_sources: dict[str, ParsedChunk],
) -> list[dict[str, object]]:
    """Return entity evidence rows for chunks available in the current batch."""
    return [
        evidence_row(entity.entity_id, "entity", chunk)
        for chunk_id in entity.source_chunk_ids
        if (chunk := chunk_sources.get(chunk_id)) is not None
    ]


def available_evidence_rows_for_relation(
    relation: ExtractedRelation,
    chunk_sources: dict[str, ParsedChunk],
) -> list[dict[str, object]]:
    """Return relation evidence rows for chunks available in the current batch."""
    return [
        evidence_row(relation.relation_id, "relation", chunk)
        for chunk_id in relation.source_chunk_ids
        if (chunk := chunk_sources.get(chunk_id)) is not None
    ]


def evidence_row(fact_id: str, fact_type: str, chunk: ParsedChunk) -> dict[str, object]:
    """Return one graph evidence SQL row."""
    return {
        "fact_id": fact_id,
        "fact_type": fact_type,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_key": chunk.source.key,
        "source_name": chunk.source.name,
        "metadata": compact_json({"page_index": chunk.page_index}),
    }


async def ensure_graph_tables(tx: SQLStoreProtocol, config: GraphStorageConfig) -> None:
    """Create graph SQL tables and indexes if needed."""
    table_names = config.table_names
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.entities} (
            entity_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_subtype TEXT,
            description TEXT NOT NULL,
            attributes TEXT NOT NULL,
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
            source_entity_name TEXT NOT NULL,
            target_entity_name TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            relation_name TEXT NOT NULL,
            description TEXT NOT NULL,
            attributes TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_names.evidence} (
            fact_id TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source_key TEXT NOT NULL,
            source_name TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (fact_id, fact_type, chunk_id)
        )
        """
    )
    await tx.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_names.evidence}_fact ON "
        f"{table_names.evidence}(fact_id, fact_type)"
    )
    await tx.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_names.evidence}_chunk ON "
        f"{table_names.evidence}(chunk_id)"
    )


async def upsert_entity_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    """Upsert entity rows by entity_id."""
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                entity_id,
                entity_name,
                entity_type,
                entity_subtype,
                description,
                attributes,
                updated_at
            )
            VALUES (
                :entity_id,
                :entity_name,
                :entity_type,
                :entity_subtype,
                :description,
                :attributes,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_type = excluded.entity_type,
                entity_subtype = excluded.entity_subtype,
                description = excluded.description,
                attributes = excluded.attributes,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def upsert_relation_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    """Upsert relation rows by relation_id."""
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                relation_id,
                source_entity_id,
                target_entity_id,
                source_entity_name,
                target_entity_name,
                relation_type,
                relation_name,
                description,
                attributes,
                updated_at
            )
            VALUES (
                :relation_id,
                :source_entity_id,
                :target_entity_id,
                :source_entity_name,
                :target_entity_name,
                :relation_type,
                :relation_name,
                :description,
                :attributes,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (relation_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                target_entity_id = excluded.target_entity_id,
                source_entity_name = excluded.source_entity_name,
                target_entity_name = excluded.target_entity_name,
                relation_type = excluded.relation_type,
                relation_name = excluded.relation_name,
                description = excluded.description,
                attributes = excluded.attributes,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def upsert_evidence_rows(
    tx: SQLStoreProtocol,
    table: str,
    rows: list[dict[str, object]],
) -> None:
    """Upsert graph evidence rows."""
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                fact_id,
                fact_type,
                chunk_id,
                document_id,
                source_key,
                source_name,
                metadata,
                updated_at
            )
            VALUES (
                :fact_id,
                :fact_type,
                :chunk_id,
                :document_id,
                :source_key,
                :source_name,
                :metadata,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (fact_id, fact_type, chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                source_key = excluded.source_key,
                source_name = excluded.source_name,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


def missing_chunk_issue(step_name: str, *, fact_id: str, fact_type: str, chunk_id: str) -> StepIssue:
    """Return a standard non-fatal issue for a missing evidence chunk."""
    return StepIssue(
        step=step_name,
        subject=IssueSubject(type=fact_type, id=fact_id),
        code="missing_evidence_chunk",
        message="Evidence chunk was not found in the graph build input.",
        resolution=IssueResolution(
            action="skipped_evidence",
            outcome="The graph fact was written, but this evidence row was skipped.",
        ),
        details={"chunk_id": chunk_id},
    )


def compact_json(value: object) -> str:
    """Serialize a value to compact UTF-8 JSON text."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def batches(items: list[_T], batch_size: int) -> list[list[_T]]:
    """Split a list into fixed-size batches."""
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def validate_identifier(value: str, *, field_name: str) -> None:
    """Validate a SQL identifier managed by graph storage config."""
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{field_name} must be a valid SQL identifier")
