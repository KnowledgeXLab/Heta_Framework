"""HiRAG hierarchical query engines."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Literal

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorSearchResult, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import optional_language_model_from_context, should_generate_answer
from heta_framework.kb.search.engines._provenance import citations_from_results, chunk_source
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.build_hirag_graph import HiRAGGraphIndexAdapter
from heta_framework.kb.steps.extract_hirag_graph import HIRAG_PROMPTS
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


Branch = Literal["hi", "nobridge", "local", "global", "bridge"]


@dataclass(frozen=True)
class HiRAGQueryEngine:
    """Generic HiRAG query engine parameterized by branch."""

    mode: str = "hi_rag_query"
    branch: Branch = "hi"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="hi_rag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="hi_rag_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None
    prompts: dict[str, Any] | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        return frozenset({model_ref("embedding", self.embedding_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        tables_asset = context.assets.require(self.graph_tables_ref)
        vectors_asset = context.assets.require(self.graph_vectors_ref)
        tables = _tables(tables_asset)
        sql_store = _require_sql_store(
            context.recipe.get_component(_store_ref_from_asset(tables_asset.store, kind="sql"))
        )
        vector_store = _require_vector_store(
            context.recipe.get_component(_store_ref_from_asset(vectors_asset.store, kind="vector"))
        )
        embedder = _require_embedding_model(
            context.recipe.get_component(model_ref("embedding", self.embedding_model))
        )

        query_vector = (
            await embedder.embed(
                EmbeddingRequest(
                    texts=[request.text],
                    trace_context={"query_mode": self.mode, "purpose": "hi_rag_query"},
                )
            )
        ).vectors[0]
        entity_collection = _metadata_string(
            vectors_asset.metadata,
            "entity_collection",
            default=vectors_asset.name,
        )
        hit_top_k = request.top_k * 10 if self.branch in {"hi", "bridge", "global"} else request.top_k
        entity_hits = await vector_store.search(
            entity_collection,
            VectorQuery(vector=query_vector, top_k=hit_top_k, filter=dict(request.filters) or None),
        )

        entities_by_id = await _entities_by_id(sql_store, tables.entities)
        relations = await _all_relations(sql_store, tables.relations)
        communities = await _all_communities(sql_store, tables.communities)
        chunks_by_id = await _chunks_by_id(sql_store, tables.chunks)
        adapter = HiRAGGraphIndexAdapter(list(entities_by_id.values()), relations)
        overall_entities = [entities_by_id[hit.id] for hit in entity_hits if hit.id in entities_by_id]
        selected_entities = overall_entities[: request.top_k]

        selected_communities = _related_communities(selected_entities, communities, request)
        selected_chunks = _related_chunks(selected_entities, chunks_by_id, request)
        selected_relations = _related_relations(selected_entities, relations, adapter, request)
        key_entities = _key_entities(selected_communities, overall_entities, request)
        shortest_path = _path_for_key_entities(adapter, key_entities)
        reasoning_edges = adapter.subgraph_edges(shortest_path) if shortest_path else []
        reasoning_edges = _truncate(reasoning_edges, key="description", max_chars=_bridge_budget(request))

        context_text = _build_context(
            branch=self.branch,
            entities=selected_entities,
            communities=selected_communities,
            relations=selected_relations,
            reasoning_edges=reasoning_edges,
            chunks=selected_chunks,
        )
        result = _query_result(
            mode=self.mode,
            context_text=context_text,
            score=entity_hits[0].score if entity_hits else None,
            entities=selected_entities,
            relations=selected_relations,
            reasoning_edges=reasoning_edges,
            communities=selected_communities,
            chunks=selected_chunks,
        )
        answer, answer_metadata = await _answer(
            context=context,
            request=request,
            context_text=context_text,
            result=result,
            mode=self.mode,
            language_model=self.language_model,
            prompts=self.prompts or HIRAG_PROMPTS,
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage=self.mode,
                    message="Built HiRAG context.",
                    metadata={
                        "entity_vector_hits": [_hit_metadata(hit) for hit in entity_hits],
                        "selected_community_ids": [row["community_id"] for row in selected_communities],
                        "selected_chunk_ids": [row["chunk_id"] for row in selected_chunks],
                        "key_entities": key_entities,
                        "shortest_path": shortest_path,
                        "reasoning_path_relation_ids": [row["relation_id"] for row in reasoning_edges],
                        "truncation_budgets": {
                            "text_unit": _text_budget(request),
                            "local_context": _local_budget(request),
                            "bridge_knowledge": _bridge_budget(request),
                            "community_report": _community_budget(request),
                        },
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=(result,),
            answer=answer,
            citations=citations_from_results((result,)),
            trace=trace,
            metadata={
                "entity_collection": entity_collection,
                "entity_count": len(selected_entities),
                "community_count": len(selected_communities),
                "source_count": len(selected_chunks),
                "reasoning_path_count": len(reasoning_edges),
                **answer_metadata,
            },
        )


@dataclass(frozen=True)
class HiRAGFullQueryEngine(HiRAGQueryEngine):
    mode: str = "hi_rag_query"
    branch: Branch = "hi"


@dataclass(frozen=True)
class HiRAGNobridgeQueryEngine(HiRAGQueryEngine):
    mode: str = "hi_rag_nobridge_query"
    branch: Branch = "nobridge"


@dataclass(frozen=True)
class HiRAGLocalQueryEngine(HiRAGQueryEngine):
    mode: str = "hi_rag_local_query"
    branch: Branch = "local"


@dataclass(frozen=True)
class HiRAGGlobalQueryEngine(HiRAGQueryEngine):
    mode: str = "hi_rag_global_query"
    branch: Branch = "global"


@dataclass(frozen=True)
class HiRAGBridgeQueryEngine(HiRAGQueryEngine):
    mode: str = "hi_rag_bridge_query"
    branch: Branch = "bridge"


@dataclass(frozen=True)
class _TableNames:
    entities: str
    relations: str
    communities: str
    chunks: str


async def _entities_by_id(sql_store: SQLStoreProtocol, table: str) -> dict[str, dict[str, Any]]:
    rows = await sql_store.fetch_all(f"SELECT * FROM {table}")
    return {
        str(row["entity_id"]): {
            **row,
            "source_ids_list": _json_list(row.get("source_ids")),
            "parent_entity_ids_list": _json_list(row.get("parent_entity_ids")),
        }
        for row in rows
    }


async def _all_relations(sql_store: SQLStoreProtocol, table: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "source_ids_list": _json_list(row.get("source_ids")),
        }
        for row in await sql_store.fetch_all(f'SELECT relation_id, source_entity_id, target_entity_id, description, weight, "order", source_ids FROM {table}')
    ]


async def _all_communities(sql_store: SQLStoreProtocol, table: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "nodes_list": _json_list(row.get("nodes")),
            "edges_list": _json_list(row.get("edges")),
            "chunk_ids_list": _json_list(row.get("chunk_ids")),
            "report_json_dict": _json_dict(row.get("report_json")),
        }
        for row in await sql_store.fetch_all(f"SELECT * FROM {table}")
    ]


async def _chunks_by_id(sql_store: SQLStoreProtocol, table: str) -> dict[str, dict[str, Any]]:
    return {str(row["chunk_id"]): row for row in await sql_store.fetch_all(f"SELECT * FROM {table}")}


def _related_communities(
    entities: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    request: QueryRequest,
) -> list[dict[str, Any]]:
    entity_ids = {row["entity_id"] for row in entities}
    scored: list[tuple[int, float, dict[str, Any]]] = []
    for community in communities:
        if int(community.get("level") or 0) > _level(request):
            continue
        count = len(entity_ids.intersection(community.get("nodes_list", ())))
        if count == 0:
            continue
        rating = _number(community.get("report_json_dict", {}).get("rating"), default=0.0)
        scored.append((count, rating, community))
    selected = [item[2] for item in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)]
    selected = _truncate(selected, key="report", max_chars=_community_budget(request))
    if request.options.get("community_single_one") is True:
        selected = selected[:1]
    return selected


def _related_chunks(
    entities: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    request: QueryRequest,
) -> list[dict[str, Any]]:
    chunk_ids = list(
        dict.fromkeys(
            chunk_id
            for entity in entities
            for chunk_id in entity.get("source_ids_list", ())
            if chunk_id in chunks_by_id
        )
    )
    return _truncate([chunks_by_id[chunk_id] for chunk_id in chunk_ids], key="content", max_chars=_text_budget(request))


def _related_relations(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    adapter: HiRAGGraphIndexAdapter,
    request: QueryRequest,
) -> list[dict[str, Any]]:
    entity_ids = {row["entity_id"] for row in entities}
    selected = [
        {
            **relation,
            "rank": adapter.node_degree(str(relation["source_entity_id"]))
            + adapter.node_degree(str(relation["target_entity_id"])),
        }
        for relation in relations
        if relation["source_entity_id"] in entity_ids or relation["target_entity_id"] in entity_ids
    ]
    selected = sorted(selected, key=lambda row: (row["rank"], row.get("weight") or 0.0), reverse=True)
    return _truncate(selected, key="description", max_chars=_local_budget(request))


def _key_entities(
    communities: list[dict[str, Any]],
    overall_entities: list[dict[str, Any]],
    request: QueryRequest,
) -> list[str]:
    top_m = _top_m(request)
    if not communities:
        return [row["entity_id"] for row in overall_entities[:top_m]]
    result: list[str] = []
    for community in communities:
        community_nodes = set(community.get("nodes_list", ()))
        result.extend(
            row["entity_id"]
            for row in overall_entities
            if row["entity_id"] in community_nodes
        )
    return list(dict.fromkeys(result))[: max(top_m, 1) * max(len(communities), 1)]


def _path_for_key_entities(adapter: HiRAGGraphIndexAdapter, key_entities: list[str]) -> list[str]:
    if not key_entities:
        return []
    if len(key_entities) == 1:
        return [key_entities[0]]
    path: list[str] = []
    current = key_entities[0]
    for target in key_entities[1:]:
        segment = adapter.shortest_path(current, target)
        if path:
            path.extend(segment[1:])
        else:
            path.extend(segment)
        current = target
    return path


def _build_context(
    *,
    branch: Branch,
    entities: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    reasoning_edges: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> str:
    entities_context = _csv(
        [["id", "entity", "type", "description", "rank"]]
        + [
            [index, row["entity_name"], row.get("entity_type") or "UNKNOWN", row.get("description") or "", 0]
            for index, row in enumerate(entities)
        ]
    )
    relations_context = _csv(
        [["id", "source", "target", "description", "weight", "rank"]]
        + [
            [
                index,
                row["source_entity_id"],
                row["target_entity_id"],
                row.get("description") or "",
                row.get("weight") or 0.0,
                row.get("rank") or 0,
            ]
            for index, row in enumerate(relations)
        ]
    )
    reasoning_context = _csv(
        [["id", "source", "target", "description", "weight", "rank"]]
        + [
            [
                index,
                row["source_entity_id"],
                row["target_entity_id"],
                row.get("description") or "",
                row.get("weight") or 0.0,
                row.get("rank") or 0,
            ]
            for index, row in enumerate(reasoning_edges)
        ]
    )
    communities_context = _csv(
        [["id", "content"]]
        + [[index, row.get("report") or ""] for index, row in enumerate(communities)]
    )
    sources_context = _csv(
        [["id", "content"]]
        + [[index, row.get("content") or ""] for index, row in enumerate(chunks)]
    )
    if branch == "hi":
        return f"""-----Backgrounds-----
```csv
{communities_context}
```
-----Reasoning Path-----
```csv
{reasoning_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{sources_context}
```"""
    if branch == "bridge":
        return f"""-----Reasoning Path-----
```csv
{reasoning_context}
```
-----Source Documents-----
```csv
{sources_context}
```"""
    if branch == "global":
        return f"""-----Backgrounds-----
```csv
{communities_context}
```
-----Source Documents-----
```csv
{sources_context}
```"""
    if branch == "local":
        return f"""-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relations_context}
```
-----Sources-----
```csv
{sources_context}
```"""
    return f"""-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{sources_context}
```"""


def _query_result(
    *,
    mode: str,
    context_text: str,
    score: float | None,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    reasoning_edges: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> QueryResult:
    chunk_ids = tuple(dict.fromkeys(row["chunk_id"] for row in chunks))
    return QueryResult(
        id=f"{mode}_context",
        text=context_text or "No relevant HiRAG context was retrieved.",
        score=score,
        kind="hi_rag_context",
        source=chunk_source(
            document_ids=_ordered_values(chunks, "document_id"),
            object_keys=_ordered_values(chunks, "source_key"),
            chunk_ids=chunk_ids,
            evidence_count=len(chunk_ids),
        ),
        metadata={
            "entity_ids": [row["entity_id"] for row in entities],
            "relation_ids": [row["relation_id"] for row in relations],
            "reasoning_path_relation_ids": [row["relation_id"] for row in reasoning_edges],
            "community_ids": [row["community_id"] for row in communities],
            "source_ids": list(chunk_ids),
        },
    )


async def _answer(
    *,
    context: QueryContext,
    request: QueryRequest,
    context_text: str,
    result: QueryResult,
    mode: str,
    language_model: str | None,
    prompts: dict[str, Any],
) -> tuple[str | None, dict[str, object]]:
    if request.options.get("only_need_context") is True:
        return context_text, {"answer_generation": "context_only"}
    if not should_generate_answer(request, default=True):
        return None, {"answer_generation": "disabled"}
    model = optional_language_model_from_context(context, language_model)
    if model is None:
        return None, {"answer_generation": "missing_language_model"}
    system_prompt = str(prompts["local_rag_response"]).format(
        context_data=context_text,
        response_type=str(request.options.get("response_type") or "Multiple Paragraphs"),
    )
    response = await model.invoke(
        ModelRequest(
            prompt=request.text,
            system_prompt=system_prompt,
            options=ModelOptions(temperature=0.1),
            trace_context={"query_mode": mode, "stage": "answer_generation"},
        )
    )
    return response.text or None, {"answer_generation": "generated", "answer_model": model.model_name}


def _tables(asset: SearchAsset) -> _TableNames:
    metadata = dict(asset.metadata)
    return _TableNames(
        entities=str(metadata.get("entities_table") or "hi_rag_entities"),
        relations=str(metadata.get("relations_table") or "hi_rag_relations"),
        communities=str(metadata.get("communities_table") or "hi_rag_communities"),
        chunks=str(metadata.get("chunks_table") or "hi_rag_chunks"),
    )


def _csv(rows: list[list[object]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue().strip()


def _truncate(rows: list[dict[str, Any]], *, key: str, max_chars: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        size = len(str(row.get(key) or ""))
        if selected and used + size > max_chars:
            break
        selected.append(row)
        used += size
    return selected


def _json_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item for item in value.split("<SEP>") if item]
        return parsed if isinstance(parsed, list) else []
    return []


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _number(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _option_int(request: QueryRequest, key: str, default: int, *, minimum: int = 0) -> int:
    value = request.options.get(key)
    if isinstance(value, int):
        return max(minimum, value)
    if isinstance(value, str) and value.strip().isdigit():
        return max(minimum, int(value))
    return default


def _top_m(request: QueryRequest) -> int:
    return _option_int(request, "top_m", 10, minimum=1)


def _level(request: QueryRequest) -> int:
    return _option_int(request, "level", 2, minimum=0)


def _text_budget(request: QueryRequest) -> int:
    return _option_int(request, "max_token_for_text_unit", 20000, minimum=1)


def _local_budget(request: QueryRequest) -> int:
    return _option_int(request, "max_token_for_local_context", 20000, minimum=1)


def _bridge_budget(request: QueryRequest) -> int:
    return _option_int(request, "max_token_for_bridge_knowledge", 12500, minimum=1)


def _community_budget(request: QueryRequest) -> int:
    return _option_int(request, "max_token_for_community_report", 12500, minimum=1)


def _ordered_values(rows: list[dict[str, Any]], key: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(row.get(key) or "") for row in rows if str(row.get(key) or "").strip()))


def _hit_metadata(hit: VectorSearchResult) -> dict[str, object]:
    return {"id": hit.id, "score": hit.score, "metadata": dict(hit.metadata or {})}


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _store_ref_from_asset(store: str | None, *, kind: str) -> ComponentRef:
    if store is None:
        return store_ref(kind)
    parts = store.split(".")
    if len(parts) == 2 and parts[0] == "stores":
        return store_ref(parts[1])
    if len(parts) == 3 and parts[0] == "stores":
        return store_ref(parts[1], parts[2])
    return store_ref(kind)


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
