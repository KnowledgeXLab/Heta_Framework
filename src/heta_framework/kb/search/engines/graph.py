"""Heta-style graph search query engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorSearchResult, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.graph_storage import validate_identifier
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


@dataclass(frozen=True)
class HetaGraphSearchEngine:
    """Search Heta-style graph entities and relations."""

    mode: str = "heta_graph_search"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="graph_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="graph_vector_index")
    embedding_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by graph search."""
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Search graph vector collections and hydrate graph facts from SQL."""
        tables_asset = context.assets.require(self.graph_tables_ref)
        vectors_asset = context.assets.require(self.graph_vectors_ref)
        sql_store = _require_sql_store(
            context.recipe.get_component(_store_ref_from_asset(tables_asset.store, kind="sql"))
        )
        vector_store = _require_vector_store(
            context.recipe.get_component(_store_ref_from_asset(vectors_asset.store, kind="vector"))
        )
        embedding_model = _require_embedding_model(
            context.recipe.get_component(model_ref("embedding", self.embedding_model))
        )

        vector = (
            await embedding_model.embed(
                EmbeddingRequest(
                    texts=[request.text],
                    trace_context={"query_mode": self.mode},
                )
            )
        ).vectors[0]
        entity_collection = _metadata_string(
            vectors_asset.metadata,
            "entity_collection",
            default=vectors_asset.name,
        )
        relation_collection = _metadata_string(
            vectors_asset.metadata,
            "relation_collection",
            default="graph_relations",
        )
        entity_limit = request.top_k // 2
        relation_limit = request.top_k - entity_limit
        entity_hits = (
            await vector_store.search(
                entity_collection,
                VectorQuery(
                    vector=vector,
                    top_k=entity_limit,
                    filter=dict(request.filters) or None,
                ),
            )
            if entity_limit > 0
            else []
        )
        relation_hits = (
            await vector_store.search(
                relation_collection,
                VectorQuery(
                    vector=vector,
                    top_k=relation_limit,
                    filter=dict(request.filters) or None,
                ),
            )
            if relation_limit > 0
            else []
        )

        results = await _graph_results(
            sql_store,
            tables_asset,
            entity_hits=entity_hits,
            relation_hits=relation_hits,
            evidence_top_k=_evidence_top_k(request),
            relation_expand_limit=_relation_expand_limit(request),
        )
        final_results = tuple(
            sorted(results.values(), key=_result_score, reverse=True)[: request.top_k]
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage="heta_graph_search",
                    message="Searched graph entity and relation indexes.",
                    metadata={
                        "entity_collection": entity_collection,
                        "relation_collection": relation_collection,
                        "entity_hit_count": len(entity_hits),
                        "relation_hit_count": len(relation_hits),
                        "result_count": len(final_results),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=final_results,
            trace=trace,
            metadata={
                "entity_collection": entity_collection,
                "relation_collection": relation_collection,
                "graph_tables": tables_asset.metadata,
            },
        )


async def _graph_results(
    sql_store: SQLStoreProtocol,
    asset: SearchAsset,
    *,
    entity_hits: list[VectorSearchResult],
    relation_hits: list[VectorSearchResult],
    evidence_top_k: int,
    relation_expand_limit: int,
) -> dict[tuple[str, str], QueryResult]:
    entities_table = _metadata_string(asset.metadata, "entities_table", default="entities")
    relations_table = _metadata_string(asset.metadata, "relations_table", default="relations")
    evidence_table = _metadata_string(asset.metadata, "evidence_table", default="graph_evidence")
    validate_identifier(entities_table, field_name="entities_table")
    validate_identifier(relations_table, field_name="relations_table")
    validate_identifier(evidence_table, field_name="evidence_table")

    results: dict[tuple[str, str], QueryResult] = {}
    for hit in entity_hits:
        row = await _entity_row(sql_store, entities_table, entity_id=hit.id)
        if row is None:
            continue
        await _add_entity_result(
            results,
            sql_store,
            asset,
            evidence_table,
            row,
            score=hit.score,
            evidence_top_k=evidence_top_k,
            matched_by="entity_vector",
            vector_metadata=dict(hit.metadata or {}),
        )
        one_hop_relations = await _one_hop_relation_rows(
            sql_store,
            relations_table,
            entity_name=str(row["entity_name"]),
            limit=relation_expand_limit,
        )
        for relation in one_hop_relations:
            await _add_relation_result(
                results,
                sql_store,
                asset,
                evidence_table,
                relation,
                score=_expanded_score(hit.score),
                evidence_top_k=evidence_top_k,
                matched_by="entity_one_hop",
                vector_metadata=dict(hit.metadata or {}),
            )

    for hit in relation_hits:
        row = await _relation_row(sql_store, relations_table, relation_id=hit.id)
        if row is None:
            continue
        await _add_relation_result(
            results,
            sql_store,
            asset,
            evidence_table,
            row,
            score=hit.score,
            evidence_top_k=evidence_top_k,
            matched_by="relation_vector",
            vector_metadata=dict(hit.metadata or {}),
        )
        endpoint_entities = await _entities_by_names(
            sql_store,
            entities_table,
            names=(str(row["source_entity_name"]), str(row["target_entity_name"])),
        )
        for entity in endpoint_entities:
            await _add_entity_result(
                results,
                sql_store,
                asset,
                evidence_table,
                entity,
                score=_expanded_score(hit.score),
                evidence_top_k=evidence_top_k,
                matched_by="relation_endpoint",
                vector_metadata=dict(hit.metadata or {}),
            )
    return results


async def _entity_row(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    entity_id: str,
) -> dict[str, Any] | None:
    return await sql_store.fetch_one(
        f"""
        SELECT entity_id, entity_name, entity_type, entity_subtype, description, attributes
        FROM {table}
        WHERE entity_id = :entity_id
        """,
        {"entity_id": entity_id},
    )


async def _relation_row(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    relation_id: str,
) -> dict[str, Any] | None:
    return await sql_store.fetch_one(
        f"""
        SELECT
            relation_id,
            source_entity_id,
            target_entity_id,
            source_entity_name,
            target_entity_name,
            relation_type,
            relation_name,
            description,
            attributes
        FROM {table}
        WHERE relation_id = :relation_id
        """,
        {"relation_id": relation_id},
    )


async def _one_hop_relation_rows(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    entity_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    return await sql_store.fetch_all(
        f"""
        SELECT
            relation_id,
            source_entity_id,
            target_entity_id,
            source_entity_name,
            target_entity_name,
            relation_type,
            relation_name,
            description,
            attributes
        FROM {table}
        WHERE source_entity_name = :entity_name OR target_entity_name = :entity_name
        ORDER BY relation_id ASC
        LIMIT :limit
        """,
        {"entity_name": entity_name, "limit": limit},
    )


async def _entities_by_names(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    names: tuple[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in dict.fromkeys(names):
        row = await sql_store.fetch_one(
            f"""
            SELECT entity_id, entity_name, entity_type, entity_subtype, description, attributes
            FROM {table}
            WHERE entity_name = :entity_name
            """,
            {"entity_name": name},
        )
        if row is not None:
            rows.append(row)
    return rows


async def _add_entity_result(
    results: dict[tuple[str, str], QueryResult],
    sql_store: SQLStoreProtocol,
    asset: SearchAsset,
    evidence_table: str,
    row: dict[str, Any],
    *,
    score: float | None,
    evidence_top_k: int,
    matched_by: str,
    vector_metadata: dict[str, Any],
) -> None:
    evidence = await _evidence(
        sql_store,
        evidence_table,
        fact_id=str(row["entity_id"]),
        fact_type="entity",
        limit=evidence_top_k,
    )
    result = QueryResult(
        id=str(row["entity_id"]),
        text=_entity_text(row),
        score=score,
        kind="entity",
        source=_source_from_evidence(evidence),
        metadata={
            "fact_type": "entity",
            "matched_by": matched_by,
            "entity_name": row["entity_name"],
            "entity_type": row["entity_type"],
            "entity_subtype": row.get("entity_subtype"),
            "attributes": _json_dict(row.get("attributes")),
            "evidence": evidence,
            "search_asset": asset.key,
            "vector_metadata": vector_metadata,
        },
    )
    _keep_best_result(results, result)


async def _add_relation_result(
    results: dict[tuple[str, str], QueryResult],
    sql_store: SQLStoreProtocol,
    asset: SearchAsset,
    evidence_table: str,
    row: dict[str, Any],
    *,
    score: float | None,
    evidence_top_k: int,
    matched_by: str,
    vector_metadata: dict[str, Any],
) -> None:
    evidence = await _evidence(
        sql_store,
        evidence_table,
        fact_id=str(row["relation_id"]),
        fact_type="relation",
        limit=evidence_top_k,
    )
    result = QueryResult(
        id=str(row["relation_id"]),
        text=_relation_text(row),
        score=score,
        kind="relation",
        source=_source_from_evidence(evidence),
        metadata={
            "fact_type": "relation",
            "matched_by": matched_by,
            "source_entity_id": row["source_entity_id"],
            "target_entity_id": row["target_entity_id"],
            "source_entity_name": row["source_entity_name"],
            "target_entity_name": row["target_entity_name"],
            "relation_type": row["relation_type"],
            "relation_name": row["relation_name"],
            "attributes": _json_dict(row.get("attributes")),
            "evidence": evidence,
            "search_asset": asset.key,
            "vector_metadata": vector_metadata,
        },
    )
    _keep_best_result(results, result)


async def _evidence(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    fact_id: str,
    fact_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = await sql_store.fetch_all(
        f"""
        SELECT chunk_id, document_id, source_key, source_name, metadata
        FROM {table}
        WHERE fact_id = :fact_id AND fact_type = :fact_type
        ORDER BY chunk_id ASC
        LIMIT :limit
        """,
        {"fact_id": fact_id, "fact_type": fact_type, "limit": limit},
    )
    evidence: list[dict[str, Any]] = []
    for row in rows:
        metadata = _json_dict(row.get("metadata"))
        evidence.append(
            {
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "source_key": row.get("source_key"),
                "source_name": row.get("source_name"),
                "metadata": metadata,
            }
        )
    return evidence


def _entity_text(row: dict[str, Any]) -> str:
    subtype = row.get("entity_subtype")
    type_text = f"{row['entity_type']} / {subtype}" if subtype else str(row["entity_type"])
    return "\n".join(
        (
            f"Entity: {row['entity_name']}",
            f"Type: {type_text}",
            f"Description: {row['description']}",
        )
    )


def _relation_text(row: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"Relation: {row['source_entity_name']} -> {row['target_entity_name']}",
            f"Name: {row['relation_name']}",
            f"Type: {row['relation_type']}",
            f"Description: {row['description']}",
        )
    )


def _source_from_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {}
    first = evidence[0]
    return {
        "document_id": first.get("document_id"),
        "source_key": first.get("source_key"),
        "source_name": first.get("source_name"),
        "chunk_id": first.get("chunk_id"),
        "evidence_count": len(evidence),
    }


def _evidence_top_k(request: QueryRequest) -> int:
    value = request.options.get("evidence_top_k")
    if isinstance(value, int) and value > 0:
        return value
    return 3


def _relation_expand_limit(request: QueryRequest) -> int:
    value = request.options.get("relation_expand_limit")
    if isinstance(value, int) and value > 0:
        return value
    return 20


def _expanded_score(score: float | None) -> float | None:
    return None if score is None else score * 0.95


def _result_score(result: QueryResult) -> float:
    return result.score if result.score is not None else -1.0


def _keep_best_result(
    results: dict[tuple[str, str], QueryResult],
    result: QueryResult,
) -> None:
    key = (result.kind, result.id)
    existing = results.get(key)
    if existing is None or _result_score(result) > _result_score(existing):
        results[key] = result


def _json_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _store_ref_from_asset(store: str | None, *, kind: str) -> ComponentRef:
    if store is None:
        return store_ref(kind)
    parts = store.split(".")
    if len(parts) == 2 and parts == ["stores", kind]:
        return store_ref(kind)
    if len(parts) == 3 and parts[:2] == ["stores", kind]:
        return store_ref(kind, parts[2])
    if store == kind:
        return store_ref(kind)
    raise ValueError(f"graph asset must reference a {kind} store, got: {store}")


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component
