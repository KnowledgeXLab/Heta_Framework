"""GraphRAG local query engine."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorSearchResult, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results_with_prompt,
    optional_language_model_from_context,
    parse_json_object,
    should_generate_answer,
)
from heta_framework.kb.search.engines.answer_prompts import (
    graph_rag_global_map_prompt,
    graph_rag_global_reduce_prompt,
    graph_rag_local_answer_prompt,
)
from heta_framework.kb.search.engines._provenance import chunk_source, citations_from_results
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.graph_storage import validate_identifier
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


@dataclass(frozen=True)
class GraphRAGLocalQueryEngine:
    """Answer a local GraphRAG query from nearby entities, relations, and communities."""

    mode: str = "graph_rag_local_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="rag_graph_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="rag_graph_vector_index")
    chunk_vectors_ref: SearchAssetRef = SearchAssetRef(kind="chunk_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by local GraphRAG query."""
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run GraphRAG local query."""
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

        embedding = await embedding_model.embed(
            EmbeddingRequest(
                texts=[request.text],
                trace_context={"query_mode": self.mode},
            )
        )
        query_vector = embedding.vectors[0]
        entity_collection = _metadata_string(
            vectors_asset.metadata,
            "entity_collection",
            default=vectors_asset.name,
        )
        entity_hits = await vector_store.search(
            entity_collection,
            VectorQuery(
                vector=query_vector,
                top_k=request.top_k,
                filter=dict(request.filters) or None,
            ),
        )

        tables = _tables(tables_asset)
        entities = await _entities_from_hits(sql_store, tables.entities, entity_hits)
        relations = await _related_relations(
            sql_store,
            tables.relations,
            entities,
            limit=_relation_limit(request),
        )
        communities = await _related_communities(
            sql_store,
            tables.communities,
            entities,
            max_count=_community_limit(request),
            level=_community_level(request),
        )
        chunk_rows = await _related_chunk_rows(
            sql_store,
            tables.chunks,
            entities,
            relations,
            max_count=_source_limit(request),
        )
        vector_chunk_rows = await _optional_chunk_vector_hits(
            context,
            request,
            query_vector,
            max_count=_vector_source_limit(request),
        )
        context_text = _build_local_context(
            entities=entities,
            relations=relations,
            communities=communities,
            chunk_rows=(*chunk_rows, *vector_chunk_rows),
            request=request,
        )

        results = _query_results(
            context_text=context_text,
            entities=entities,
            relations=relations,
            communities=communities,
            chunk_rows=(*chunk_rows, *vector_chunk_rows),
            score=entity_hits[0].score if entity_hits else None,
        )
        answer, answer_metadata = await _generate_answer(
            context=context,
            request=request,
            local_context=context_text,
            results=results,
            mode=self.mode,
            language_model=self.language_model,
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage="graph_rag_local_query",
                    message="Built local GraphRAG context from entities, communities, chunks, and relations.",
                    metadata={
                        "entity_collection": entity_collection,
                        "entity_hit_count": len(entity_hits),
                        "entity_count": len(entities),
                        "relation_count": len(relations),
                        "community_count": len(communities),
                        "source_count": len(chunk_rows) + len(vector_chunk_rows),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=results,
            answer=answer,
            citations=citations_from_results(results),
            trace=trace,
            metadata={
                "entity_collection": entity_collection,
                "graph_tables": tables_asset.metadata,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "community_count": len(communities),
                "source_count": len(chunk_rows) + len(vector_chunk_rows),
                "embedding_model": embedding.model_name or embedding_model.model_name,
                **answer_metadata,
            },
        )


@dataclass(frozen=True)
class GraphRAGGlobalQueryEngine:
    """Answer a global GraphRAG query by map-reducing community reports."""

    mode: str = "graph_rag_global_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="rag_graph_tables")
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by global GraphRAG query."""
        return frozenset({self.graph_tables_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run global GraphRAG query."""
        tables_asset = context.assets.require(self.graph_tables_ref)
        sql_store = _require_sql_store(
            context.recipe.get_component(_store_ref_from_asset(tables_asset.store, kind="sql"))
        )
        tables = _tables(tables_asset)
        communities = await _global_communities(
            sql_store,
            tables.communities,
            level=_community_level(request),
            max_count=_global_max_consider_community(request),
            min_rating=_global_min_community_rating(request),
        )
        mapped_points = await _map_global_communities(
            context=context,
            request=request,
            communities=communities,
            mode=self.mode,
            language_model=self.language_model,
        )
        points = _global_support_points(
            mapped_points,
            max_chars=_global_points_budget(request),
        )
        points_context = _global_points_context(
            points,
            max_chars=_global_points_budget(request),
        )
        results = _global_query_results(
            points_context=points_context,
            communities=communities,
            points=points,
        )
        answer, answer_metadata = await _reduce_global_answer(
            context=context,
            request=request,
            points_context=points_context,
            results=results,
            mode=self.mode,
            language_model=self.language_model,
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage="graph_rag_global_query",
                    message="Mapped community reports to support points and reduced them into a global answer.",
                    metadata={
                        "community_count": len(communities),
                        "support_point_count": len(points),
                        "community_group_count": len(mapped_points),
                        "level": _community_level(request),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=results,
            answer=answer,
            citations=citations_from_results(results),
            trace=trace,
            metadata={
                "graph_tables": tables_asset.metadata,
                "community_count": len(communities),
                "support_point_count": len(points),
                "community_group_count": len(mapped_points),
                **answer_metadata,
            },
        )


@dataclass(frozen=True)
class _TableNames:
    entities: str
    relations: str
    communities: str
    chunks: str


async def _generate_answer(
    *,
    context: QueryContext,
    request: QueryRequest,
    local_context: str,
    results: tuple[QueryResult, ...],
    mode: str,
    language_model: str | None,
) -> tuple[str | None, dict[str, object]]:
    if not should_generate_answer(request, default=True):
        return None, {"answer_generation": "disabled"}
    model = optional_language_model_from_context(context, language_model)
    if model is None:
        return None, {
            "answer_generation": "missing_language_model",
            "answer_generation_requested": True,
        }
    answer = await answer_from_results_with_prompt(
        model,
        query=str(request.text),
        results=results,
        prompt=str(
            graph_rag_local_answer_prompt(
                local_context=str(local_context),
                response_type=str(_response_type(request)),
            )
        ),
        trace_context={"query_mode": mode, "stage": "answer_generation"},
    )
    return answer or None, {
        "answer_generation": "generated" if answer else "empty",
        "answer_model": model.model_name,
    }


async def _entities_from_hits(
    sql_store: SQLStoreProtocol,
    table: str,
    hits: list[VectorSearchResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        row = await sql_store.fetch_one(
            f"""
            SELECT
                entity_id,
                entity_name,
                entity_type,
                description,
                source_id,
                source_ids,
                properties
            FROM {table}
            WHERE entity_id = :entity_id
            """,
            {"entity_id": hit.id},
        )
        if row is None:
            continue
        rows.append(
            {
                **row,
                "score": hit.score,
                "order": index,
                "source_ids_list": _json_list(row.get("source_ids")),
                "properties_json": _json_dict(row.get("properties")),
            }
        )
    return rows


async def _related_relations(
    sql_store: SQLStoreProtocol,
    table: str,
    entities: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    relation_by_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        rows = await sql_store.fetch_all(
            f"""
            SELECT
                relation_id,
                source_entity_id,
                target_entity_id,
                relation_type,
                description,
                weight,
                source_id,
                source_ids,
                properties
            FROM {table}
            WHERE source_entity_id = :entity_id OR target_entity_id = :entity_id
            ORDER BY weight DESC, relation_id ASC
            LIMIT :limit
            """,
            {"entity_id": entity["entity_id"], "limit": limit},
        )
        for row in rows:
            relation_by_id[str(row["relation_id"])] = {
                **row,
                "source_ids_list": _json_list(row.get("source_ids")),
                "properties_json": _json_dict(row.get("properties")),
            }
    return sorted(
        relation_by_id.values(),
        key=lambda row: (float(row.get("weight") or 0.0), str(row.get("relation_id"))),
        reverse=True,
    )[:limit]


async def _related_communities(
    sql_store: SQLStoreProtocol,
    table: str,
    entities: list[dict[str, Any]],
    *,
    max_count: int,
    level: int,
) -> list[dict[str, Any]]:
    if not entities or max_count <= 0:
        return []
    entity_ids = {str(row["entity_id"]) for row in entities}
    entity_names = {str(row["entity_name"]) for row in entities}
    rows = await sql_store.fetch_all(
        f"""
        SELECT
            community_id,
            level,
            title,
            report,
            report_json,
            nodes,
            edges,
            chunk_ids,
            occurrence,
            sub_communities
        FROM {table}
        WHERE level <= :level
        ORDER BY level DESC, occurrence DESC, community_id ASC
        """,
        {"level": level},
    )
    ranked: list[dict[str, Any]] = []
    for row in rows:
        nodes = _json_list(row.get("nodes"))
        overlap = len((set(nodes) & entity_ids) | (set(nodes) & entity_names))
        if overlap <= 0:
            continue
        report_json = _json_dict(row.get("report_json"))
        ranked.append(
            {
                **row,
                "nodes_list": nodes,
                "edges_list": _json_list(row.get("edges")),
                "chunk_ids_list": _json_list(row.get("chunk_ids")),
                "sub_communities_list": _json_list(row.get("sub_communities")),
                "report_json_dict": report_json,
                "overlap": overlap,
                "rating": _number(report_json.get("rating"), default=-1.0),
            }
        )
    return sorted(
        ranked,
        key=lambda row: (
            int(row["overlap"]),
            float(row["rating"]),
            float(row.get("occurrence") or 0.0),
        ),
        reverse=True,
    )[:max_count]


async def _related_chunk_rows(
    sql_store: SQLStoreProtocol,
    table: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    max_count: int,
) -> tuple[dict[str, Any], ...]:
    counts: Counter[str] = Counter()
    first_order: dict[str, int] = {}
    for order, row in enumerate((*entities, *relations)):
        for chunk_id in row.get("source_ids_list", ()):
            counts[str(chunk_id)] += 1
            first_order.setdefault(str(chunk_id), order)
    chunk_ids = sorted(counts, key=lambda item: (counts[item], -first_order[item]), reverse=True)
    chunk_datas = await _chunk_rows_by_ids(sql_store, table, chunk_ids[:max_count])
    return tuple(
        {
            "id": chunk_id,
            "content": str(chunk_datas.get(chunk_id, {}).get("content") or f"Source chunk id: {chunk_id}"),
            "source": "graph_source_id",
            "relation_counts": counts[chunk_id],
            "metadata": chunk_datas.get(chunk_id, {}),
        }
        for chunk_id in chunk_ids[:max_count]
    )


async def _chunk_rows_by_ids(
    sql_store: SQLStoreProtocol,
    table: str,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for chunk_id in chunk_ids:
        row = await sql_store.fetch_one(
            f"""
            SELECT
                chunk_id,
                document_id,
                content,
                source_key,
                source_name,
                source_file_type,
                page_index,
                chunk_index,
                token_start,
                token_end,
                metadata
            FROM {table}
            WHERE chunk_id = :chunk_id
            """,
            {"chunk_id": chunk_id},
        )
        if row is not None:
            rows[str(row["chunk_id"])] = {
                **row,
                "metadata_json": _json_dict(row.get("metadata")),
            }
    return rows


async def _optional_chunk_vector_hits(
    context: QueryContext,
    request: QueryRequest,
    query_vector: list[float],
    *,
    max_count: int,
) -> tuple[dict[str, Any], ...]:
    if max_count <= 0 or request.options.get("include_chunk_vector_hits") is not True:
        return ()
    matches = context.assets.find(SearchAssetRef(kind="chunk_vector_index"))
    if len(matches) != 1:
        return ()
    asset = matches[0]
    vector_store = _require_vector_store(
        context.recipe.get_component(_store_ref_from_asset(asset.store, kind="vector"))
    )
    collection = _metadata_string(asset.metadata, "collection", default=asset.name)
    hits = await vector_store.search(
        collection,
        VectorQuery(
            vector=query_vector,
            top_k=max_count,
            filter=dict(request.filters) or None,
        ),
    )
    return tuple(
        {
            "id": hit.id,
            "content": hit.text or f"Source chunk id: {hit.id}",
            "source": "chunk_vector_hit",
            "score": hit.score,
            "metadata": dict(hit.metadata or {}),
        }
        for hit in hits
        if hit.text or hit.id
    )


def _build_local_context(
    *,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    chunk_rows: tuple[dict[str, Any], ...],
    request: QueryRequest,
) -> str:
    entities_rows = [["id", "entity", "type", "description", "rank"]]
    relation_counts = _entity_relation_counts(entities, relations)
    for index, row in enumerate(_truncate(entities, key="description", max_chars=_entity_budget(request))):
        entity_id = str(row["entity_id"])
        entities_rows.append(
            [
                index,
                row.get("entity_name") or entity_id,
                row.get("entity_type") or "UNKNOWN",
                row.get("description") or "",
                relation_counts.get(entity_id, 0),
            ]
        )

    relation_rows = [["id", "source", "target", "description", "weight", "rank"]]
    for index, row in enumerate(_truncate(relations, key="description", max_chars=_relation_budget(request))):
        relation_rows.append(
            [
                index,
                row.get("source_entity_id") or "",
                row.get("target_entity_id") or "",
                row.get("description") or "",
                row.get("weight") or 0.0,
                relation_counts.get(str(row.get("source_entity_id")), 0)
                + relation_counts.get(str(row.get("target_entity_id")), 0),
            ]
        )

    community_rows = [["id", "content"]]
    for index, row in enumerate(_truncate(communities, key="report", max_chars=_community_budget(request))):
        community_rows.append([index, row.get("report") or ""])

    source_rows = [["id", "content"]]
    for index, row in enumerate(_truncate(list(chunk_rows), key="content", max_chars=_source_budget(request))):
        source_rows.append([index, row.get("content") or ""])

    return f"""-----Reports-----
```csv
{_csv(community_rows)}
```
-----Entities-----
```csv
{_csv(entities_rows)}
```
-----Relationships-----
```csv
{_csv(relation_rows)}
```
-----Sources-----
```csv
{_csv(source_rows)}
```"""


def _query_results(
    *,
    context_text: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    chunk_rows: tuple[dict[str, Any], ...],
    score: float | None,
) -> tuple[QueryResult, ...]:
    chunk_ids = tuple(
        dict.fromkeys(
            str(chunk_id)
            for row in (*entities, *relations)
            for chunk_id in row.get("source_ids_list", ())
            if str(chunk_id).strip()
        )
    )
    return (
        QueryResult(
            id="graph_rag_local_context",
            text=context_text,
            score=score,
            kind="graph_rag_context",
            source=chunk_source(chunk_ids=chunk_ids, evidence_count=len(chunk_ids)),
            metadata={
                "entity_ids": [str(row["entity_id"]) for row in entities],
                "relation_ids": [str(row["relation_id"]) for row in relations],
                "community_ids": [str(row["community_id"]) for row in communities],
                "source_ids": [str(row["id"]) for row in chunk_rows],
            },
        ),
    )


async def _global_communities(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    level: int,
    max_count: int,
    min_rating: float,
) -> list[dict[str, Any]]:
    rating_expr = "COALESCE(CAST(json_extract(report_json, '$.rating') AS REAL), 0)"
    return [
        _community_data_from_row(row)
        for row in await sql_store.fetch_all(
            f"""
            SELECT
                community_id,
                level,
                title,
                report,
                report_json,
                nodes,
                edges,
                chunk_ids,
                occurrence,
                sub_communities
            FROM {table}
            WHERE level <= :level
              AND {rating_expr} >= :min_rating
            ORDER BY occurrence DESC, {rating_expr} DESC
            LIMIT :limit
            """,
            {"level": level, "min_rating": min_rating, "limit": max_count},
        )
    ]


def _community_data_from_row(row: dict[str, Any]) -> dict[str, Any]:
    report_json = _json_dict(row.get("report_json"))
    rating = _number(report_json.get("rating"), default=0.0)
    return {
        **row,
        "report_json_dict": report_json,
        "rating": rating,
        "nodes_list": _json_list(row.get("nodes")),
        "edges_list": _json_list(row.get("edges")),
        "chunk_ids_list": _json_list(row.get("chunk_ids")),
        "sub_communities_list": _json_list(row.get("sub_communities")),
    }


async def _map_global_communities(
    *,
    context: QueryContext,
    request: QueryRequest,
    communities: list[dict[str, Any]],
    mode: str,
    language_model: str | None,
) -> list[list[dict[str, Any]]]:
    model = optional_language_model_from_context(context, language_model)
    if model is None or not communities:
        return []
    groups = _community_groups(
        communities,
        max_chars=_global_community_group_budget(request),
    )

    async def _process_group(group_index: int, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        community_context = _global_community_context(group)
        result = await model.invoke(
            ModelRequest(
                prompt=str(request.text),
                system_prompt=str(graph_rag_global_map_prompt(str(community_context))),
                options=ModelOptions(
                    temperature=0,
                    max_output_tokens=_global_map_max_output_tokens(request),
                    response_format={"type": "json_object"},
                ),
                response_schema={"type": "object"},
                trace_context={
                    "query_mode": mode,
                    "stage": "global_community_map",
                    "group": group_index,
                },
            )
        )
        data = result.parsed if isinstance(result.parsed, dict) else parse_json_object(result.text)
        raw_points = data.get("points") if isinstance(data, dict) else None
        if not isinstance(raw_points, list):
            return []
        return [point for point in raw_points if isinstance(point, dict)]

    return list(
        await asyncio.gather(
            *[
                _process_group(group_index, group)
                for group_index, group in enumerate(groups)
            ]
        )
    )


async def _reduce_global_answer(
    *,
    context: QueryContext,
    request: QueryRequest,
    points_context: str,
    results: tuple[QueryResult, ...],
    mode: str,
    language_model: str | None,
) -> tuple[str | None, dict[str, object]]:
    if request.options.get("only_need_context") is True:
        return points_context or None, {"answer_generation": "context_only"}
    if not should_generate_answer(request, default=True):
        return None, {"answer_generation": "disabled"}
    model = optional_language_model_from_context(context, language_model)
    if model is None:
        return None, {
            "answer_generation": "missing_language_model",
            "answer_generation_requested": True,
        }
    if not points_context.strip():
        return None, {"answer_generation": "no_support_points"}
    answer = await answer_from_results_with_prompt(
        model,
        query=str(request.text),
        results=results,
        prompt=str(
            graph_rag_global_reduce_prompt(
                query=str(request.text),
                points_context=str(points_context),
                response_type=str(_response_type(request)),
            )
        ),
        trace_context={"query_mode": mode, "stage": "global_reduce"},
    )
    return answer or None, {
        "answer_generation": "generated" if answer else "empty",
        "answer_model": model.model_name,
    }


def _community_groups(
    communities: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    remaining = list(communities)
    while remaining:
        group = _truncate(remaining, key="report", max_chars=max_chars)
        if not group:
            group = remaining[:1]
        groups.append(group)
        remaining = remaining[len(group) :]
    return groups


def _global_community_context(communities: list[dict[str, Any]]) -> str:
    rows = [["id", "content", "rating", "importance"]]
    for index, community in enumerate(communities):
        rows.append(
            [
                index,
                community.get("report") or "",
                community.get("rating") or 0,
                community.get("occurrence") or 0,
            ]
        )
    return _csv(rows)


def _global_support_points(
    mapped_points: list[list[dict[str, Any]]],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    support_points: list[dict[str, Any]] = []
    for analyst, group_points in enumerate(mapped_points):
        for point in group_points:
            if "description" not in point:
                continue
            description = str(point.get("description") or "").strip()
            score = _number(point.get("score"), default=1.0)
            if description and score > 0:
                support_points.append(
                    {
                        "analyst": analyst,
                        "answer": description,
                        "score": score,
                    }
                )
    support_points = sorted(
        support_points,
        key=lambda point: float(point["score"]),
        reverse=True,
    )
    return _truncate(support_points, key="answer", max_chars=max_chars)


def _global_points_context(points: list[dict[str, Any]], *, max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for point in points:
        block = (
            f"----Analyst {point['analyst']}----\n"
            f"Importance Score: {point['score']}\n"
            f"{point['answer']}\n"
        )
        if lines and used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def _global_query_results(
    *,
    points_context: str,
    communities: list[dict[str, Any]],
    points: list[dict[str, Any]],
) -> tuple[QueryResult, ...]:
    if not points_context.strip():
        points_context = "No relevant community support points were retrieved."
    chunk_ids = tuple(
        dict.fromkeys(
            str(chunk_id)
            for community in communities
            for chunk_id in community.get("chunk_ids_list", ())
            if str(chunk_id).strip()
        )
    )
    return (
        QueryResult(
            id="graph_rag_global_context",
            text=points_context,
            score=float(points[0]["score"]) if points else None,
            kind="graph_rag_global_context",
            source=chunk_source(chunk_ids=chunk_ids, evidence_count=len(chunk_ids)),
            metadata={
                "community_ids": [str(row["community_id"]) for row in communities],
                "support_point_count": len(points),
            },
        ),
    )


def _tables(asset: SearchAsset) -> _TableNames:
    entities = _metadata_string(asset.metadata, "entities_table", default="rag_entities")
    relations = _metadata_string(asset.metadata, "relations_table", default="rag_relations")
    communities = _metadata_string(
        asset.metadata,
        "communities_table",
        default="rag_communities",
    )
    chunks = _metadata_string(asset.metadata, "chunks_table", default="rag_chunks")
    validate_identifier(entities, field_name="entities_table")
    validate_identifier(relations, field_name="relations_table")
    validate_identifier(communities, field_name="communities_table")
    validate_identifier(chunks, field_name="chunks_table")
    return _TableNames(
        entities=entities,
        relations=relations,
        communities=communities,
        chunks=chunks,
    )


def _entity_relation_counts(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {str(row["entity_id"]): 0 for row in entities}
    for relation in relations:
        for key in ("source_entity_id", "target_entity_id"):
            entity_id = str(relation.get(key) or "")
            if entity_id in counts:
                counts[entity_id] += 1
    return counts


def _csv(rows: list[list[object]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().strip()


def _truncate(rows: list[dict[str, Any]], *, key: str, max_chars: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        size = len(str(row.get(key) or ""))
        if kept and used + size > max_chars:
            break
        kept.append(row)
        used += size
    return kept


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return tuple(item for item in value.split("<SEP>") if item)
    else:
        parsed = value
    if isinstance(parsed, list | tuple | set):
        result: list[str] = []
        for item in parsed:
            if isinstance(item, list | tuple) and len(item) >= 2:
                result.append(f"{item[0]} -> {item[1]}")
            elif str(item).strip():
                result.append(str(item))
        return tuple(result)
    return ()


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def _number(value: object, *, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _option_int(request: QueryRequest, key: str, default: int, *, minimum: int = 0) -> int:
    value = request.options.get(key)
    if isinstance(value, int) and value >= minimum:
        return value
    return default


def _community_level(request: QueryRequest) -> int:
    return _option_int(request, "level", 2, minimum=0)


def _community_limit(request: QueryRequest) -> int:
    return _option_int(request, "local_community_top_k", 20, minimum=0)


def _relation_limit(request: QueryRequest) -> int:
    return _option_int(request, "local_relation_top_k", 20, minimum=0)


def _source_limit(request: QueryRequest) -> int:
    return _option_int(request, "local_source_top_k", 20, minimum=0)


def _vector_source_limit(request: QueryRequest) -> int:
    return _option_int(request, "local_vector_source_top_k", 0, minimum=0)


def _global_max_consider_community(request: QueryRequest) -> int:
    return _option_int(request, "global_max_consider_community", 512, minimum=1)


def _global_map_max_output_tokens(request: QueryRequest) -> int:
    return _option_int(request, "global_map_max_output_tokens", 1024, minimum=1)


def _community_budget(request: QueryRequest) -> int:
    return _option_int(request, "local_max_chars_for_community_report", 3200, minimum=1)


def _entity_budget(request: QueryRequest) -> int:
    return _option_int(request, "local_max_chars_for_entities", 4000, minimum=1)


def _relation_budget(request: QueryRequest) -> int:
    return _option_int(request, "local_max_chars_for_relations", 4800, minimum=1)


def _source_budget(request: QueryRequest) -> int:
    return _option_int(request, "local_max_chars_for_sources", 4000, minimum=1)


def _global_community_group_budget(request: QueryRequest) -> int:
    return _option_int(request, "global_max_chars_for_community_report", 16384, minimum=1)


def _global_points_budget(request: QueryRequest) -> int:
    return _option_int(request, "global_max_chars_for_support_points", 16384, minimum=1)


def _global_min_community_rating(request: QueryRequest) -> float:
    value = request.options.get("global_min_community_rating")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _response_type(request: QueryRequest) -> str:
    value = request.options.get("response_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Multiple Paragraphs"


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
    raise ValueError(f"GraphRAG asset must reference a {kind} store, got: {store}")


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
