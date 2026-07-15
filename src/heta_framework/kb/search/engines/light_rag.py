"""LightRAG local, global, and hybrid query engines."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorSearchResult, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import (
    optional_language_model_from_context,
    parse_json_object,
    should_generate_answer,
)
from heta_framework.kb.search.engines._provenance import citations_from_results, chunk_source
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.graph_storage import validate_identifier
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


@dataclass(frozen=True)
class LightRAGLocalQueryEngine:
    """Answer a LightRAG local query from entity-neighborhood context."""

    mode: str = "light_rag_local_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        return frozenset({model_ref("embedding", self.embedding_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await _query_lightrag(
            mode=self.mode,
            branch="local",
            request=request,
            context=context,
            graph_tables_ref=self.graph_tables_ref,
            graph_vectors_ref=self.graph_vectors_ref,
            embedding_model=self.embedding_model,
            language_model=self.language_model,
        )


@dataclass(frozen=True)
class LightRAGGlobalQueryEngine:
    """Answer a LightRAG global query from relationship vector context."""

    mode: str = "light_rag_global_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        return frozenset({model_ref("embedding", self.embedding_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await _query_lightrag(
            mode=self.mode,
            branch="global",
            request=request,
            context=context,
            graph_tables_ref=self.graph_tables_ref,
            graph_vectors_ref=self.graph_vectors_ref,
            embedding_model=self.embedding_model,
            language_model=self.language_model,
        )


@dataclass(frozen=True)
class LightRAGHybridQueryEngine:
    """Answer a LightRAG hybrid query by combining local and global branches."""

    mode: str = "light_rag_hybrid_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        return frozenset({model_ref("embedding", self.embedding_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await _query_lightrag(
            mode=self.mode,
            branch="hybrid",
            request=request,
            context=context,
            graph_tables_ref=self.graph_tables_ref,
            graph_vectors_ref=self.graph_vectors_ref,
            embedding_model=self.embedding_model,
            language_model=self.language_model,
        )


@dataclass(frozen=True)
class LightRAGMixQueryEngine:
    """Answer a LightRAG mix query by combining KG and direct chunk retrieval."""

    mode: str = "light_rag_mix_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="light_rag_vector_index")
    embedding_model: str | None = None
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        return frozenset({self.graph_tables_ref, self.graph_vectors_ref})

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        return frozenset({model_ref("embedding", self.embedding_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        return await _query_lightrag(
            mode=self.mode,
            branch="mix",
            request=request,
            context=context,
            graph_tables_ref=self.graph_tables_ref,
            graph_vectors_ref=self.graph_vectors_ref,
            embedding_model=self.embedding_model,
            language_model=self.language_model,
        )


@dataclass(frozen=True)
class _TableNames:
    entities: str
    relations: str
    chunks: str


async def _query_lightrag(
    *,
    mode: str,
    branch: str,
    request: QueryRequest,
    context: QueryContext,
    graph_tables_ref: SearchAssetRef,
    graph_vectors_ref: SearchAssetRef,
    embedding_model: str | None,
    language_model: str | None,
) -> QueryResponse:
    tables_asset = context.assets.require(graph_tables_ref)
    vectors_asset = context.assets.require(graph_vectors_ref)
    tables = _tables(tables_asset)
    sql_store = _require_sql_store(
        context.recipe.get_component(_store_ref_from_asset(tables_asset.store, kind="sql"))
    )
    vector_store = _require_vector_store(
        context.recipe.get_component(_store_ref_from_asset(vectors_asset.store, kind="vector"))
    )
    embedder = _require_embedding_model(
        context.recipe.get_component(model_ref("embedding", embedding_model))
    )
    model = optional_language_model_from_context(context, language_model)

    keywords = await _keywords_from_query(request, model, mode=mode)
    entity_collection = _metadata_string(
        vectors_asset.metadata,
        "entity_collection",
        default=vectors_asset.name,
    )
    relationship_collection = _metadata_string(
        vectors_asset.metadata,
        "relationship_collection",
        default="light_rag_relationships",
    )
    chunk_collection = _metadata_string(
        vectors_asset.metadata,
        "chunk_collection",
        default="light_rag_chunks",
    )

    local_entities: list[dict[str, Any]] = []
    local_relations: list[dict[str, Any]] = []
    global_entities: list[dict[str, Any]] = []
    global_relations: list[dict[str, Any]] = []
    vector_chunks: list[dict[str, Any]] = []

    if branch in {"local", "hybrid", "mix"}:
        local_query = ", ".join(keywords["low_level"]) or request.text
        local_vector = await _embed_query(embedder, local_query, mode=mode, purpose="local")
        entity_hits = await vector_store.search(
            entity_collection,
            VectorQuery(
                vector=local_vector,
                top_k=request.top_k,
                filter=dict(request.filters) or None,
            ),
        )
        local_entities = await _entities_from_hits(sql_store, tables.entities, entity_hits)
        local_relations = await _relations_for_entities(
            sql_store,
            tables.relations,
            local_entities,
            limit=_relation_limit(request),
        )

    if branch in {"global", "hybrid", "mix"}:
        global_query = ", ".join(keywords["high_level"]) or request.text
        global_vector = await _embed_query(embedder, global_query, mode=mode, purpose="global")
        relation_hits = await vector_store.search(
            relationship_collection,
            VectorQuery(
                vector=global_vector,
                top_k=request.top_k,
                filter=dict(request.filters) or None,
            ),
        )
        global_relations = await _relations_from_hits(
            sql_store,
            tables.relations,
            relation_hits,
        )
        global_entities = await _entities_for_relations(
            sql_store,
            tables.entities,
            global_relations,
        )

    if branch == "mix":
        chunk_vector = await _embed_query(embedder, request.text, mode=mode, purpose="chunk")
        chunk_hits = await vector_store.search(
            chunk_collection,
            VectorQuery(
                vector=chunk_vector,
                top_k=_chunk_limit(request),
                filter=dict(request.filters) or None,
            ),
        )
        vector_chunks = await _chunks_from_hits(sql_store, tables.chunks, chunk_hits)

    entities = _round_robin_entities(local_entities, global_entities)
    relations = _round_robin_relations(local_relations, global_relations)
    fact_chunks = await _chunks_for_facts(
        sql_store,
        tables.chunks,
        entities=entities,
        relations=relations,
        limit=_chunk_limit(request),
    )
    chunks = (
        _round_robin_chunks(vector_chunks, fact_chunks)
        if branch == "mix"
        else fact_chunks
    )[: _chunk_limit(request)]
    context_text, raw_data = _build_lightrag_context(
        mode=mode,
        entities=entities,
        relations=relations,
        chunks=chunks,
    )
    results = _query_results(
        context_text=context_text,
        entities=entities,
        relations=relations,
        chunks=chunks,
    )
    answer, answer_metadata = await _generate_answer(
        model=model,
        request=request,
        mode=mode,
        context_text=context_text,
    )
    trace = ()
    if request.trace:
        trace = (
            QueryTraceEvent(
                stage=mode,
                message="Built LightRAG context from entities, relationships, and chunks.",
                metadata={
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "chunk_count": len(chunks),
                    "vector_chunk_count": len(vector_chunks),
                    "fact_chunk_count": len(fact_chunks),
                    "branch": branch,
                },
            ),
        )
    return QueryResponse(
        mode=mode,
        results=results,
        answer=answer,
        citations=citations_from_results(results),
        trace=trace,
        metadata={
            "keywords": keywords,
            "context": context_text,
            "raw_data": raw_data,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "chunk_count": len(chunks),
            "vector_chunk_count": len(vector_chunks),
            "fact_chunk_count": len(fact_chunks),
            "entity_collection": entity_collection,
            "relationship_collection": relationship_collection,
            "chunk_collection": chunk_collection,
            "graph_tables": tables_asset.metadata,
            **answer_metadata,
        },
    )


async def _keywords_from_query(
    request: QueryRequest,
    model: Any,
    *,
    mode: str,
) -> dict[str, list[str]]:
    provided_high = request.options.get("high_level_keywords")
    provided_low = request.options.get("low_level_keywords")
    if isinstance(provided_high, list) or isinstance(provided_low, list):
        return {
            "high_level": [str(item) for item in provided_high or [] if str(item).strip()],
            "low_level": [str(item) for item in provided_low or [] if str(item).strip()],
        }
    if model is None:
        return {"high_level": [request.text], "low_level": [request.text]}
    prompts = _lightrag_prompts()
    examples = "\n".join(str(item).rstrip() for item in prompts["keywords_extraction_examples"])
    prompt = str(prompts["keywords_extraction"]).format(
        query=request.text,
        language=str(request.options.get("language") or "English"),
        examples=examples,
    )
    result = await model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(
                temperature=0,
                response_format={"type": "json_object"},
            ),
            trace_context={"query_mode": mode, "stage": "keyword_extraction"},
        )
    )
    parsed = result.parsed if isinstance(result.parsed, dict) else parse_json_object(result.text)
    if not isinstance(parsed, dict):
        parsed = {}
    high = parsed.get("high_level_keywords", [])
    low = parsed.get("low_level_keywords", [])
    return {
        "high_level": [str(item) for item in high if str(item).strip()] if isinstance(high, list) else [],
        "low_level": [str(item) for item in low if str(item).strip()] if isinstance(low, list) else [],
    }


async def _embed_query(
    embedder: EmbeddingModelProtocol,
    text: str,
    *,
    mode: str,
    purpose: str,
) -> list[float]:
    result = await embedder.embed(
        EmbeddingRequest(texts=[text], trace_context={"query_mode": mode, "purpose": purpose})
    )
    if not result.vectors:
        raise ValueError("query embedding result is empty")
    return [float(value) for value in result.vectors[0]]


async def _entities_from_hits(
    sql_store: SQLStoreProtocol,
    table: str,
    hits: list[VectorSearchResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        row = await sql_store.fetch_one(
            f"""
            SELECT * FROM {table}
            WHERE entity_id = :entity_id
            """,
            {"entity_id": hit.id},
        )
        if row is None:
            continue
        rows.append(_decode_row(row, score=hit.score, order=index))
    return rows


async def _relations_from_hits(
    sql_store: SQLStoreProtocol,
    table: str,
    hits: list[VectorSearchResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        row = await sql_store.fetch_one(
            f"""
            SELECT * FROM {table}
            WHERE relation_id = :relation_id
            """,
            {"relation_id": hit.id},
        )
        if row is None:
            continue
        rows.append(_decode_row(row, score=hit.score, order=index))
    return rows


async def _chunks_from_hits(
    sql_store: SQLStoreProtocol,
    table: str,
    hits: list[VectorSearchResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hit in enumerate(hits):
        row = await sql_store.fetch_one(
            f"""
            SELECT * FROM {table}
            WHERE chunk_id = :chunk_id
            """,
            {"chunk_id": hit.id},
        )
        if row is None:
            continue
        rows.append(_decode_row(row, score=hit.score, order=index))
    return rows


async def _relations_for_entities(
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
            SELECT * FROM {table}
            WHERE source_entity_id = :entity_id OR target_entity_id = :entity_id
            ORDER BY weight DESC, relation_id ASC
            LIMIT :limit
            """,
            {"entity_id": entity["entity_id"], "limit": limit},
        )
        for row in rows:
            relation_by_id[str(row["relation_id"])] = _decode_row(row)
    return list(relation_by_id.values())[:limit]


async def _entities_for_relations(
    sql_store: SQLStoreProtocol,
    table: str,
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_by_id: dict[str, dict[str, Any]] = {}
    for relation in relations:
        for entity_id in (relation["source_entity_id"], relation["target_entity_id"]):
            if entity_id in entity_by_id:
                continue
            row = await sql_store.fetch_one(
                f"""
                SELECT * FROM {table}
                WHERE entity_id = :entity_id
                """,
                {"entity_id": entity_id},
            )
            if row is not None:
                entity_by_id[str(entity_id)] = _decode_row(row)
    return list(entity_by_id.values())


async def _chunks_for_facts(
    sql_store: SQLStoreProtocol,
    table: str,
    *,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    chunk_ids: list[str] = []
    for fact in (*entities, *relations):
        for chunk_id in fact.get("source_ids_list", ()):
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    chunks: list[dict[str, Any]] = []
    for chunk_id in chunk_ids[:limit]:
        row = await sql_store.fetch_one(
            f"""
            SELECT * FROM {table}
            WHERE chunk_id = :chunk_id
            """,
            {"chunk_id": chunk_id},
        )
        if row is not None:
            chunks.append(_decode_row(row))
    return chunks


def _build_lightrag_context(
    *,
    mode: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    entity_context = [
        {
            "entity": row["entity_name"],
            "type": row["entity_type"],
            "description": row["description"],
            "file_path": row.get("file_path") or "unknown_source",
        }
        for row in entities
    ]
    relation_context = [
        {
            "entity1": row["source_entity_id"],
            "entity2": row["target_entity_id"],
            "description": row["description"],
            "keywords": row.get("keywords") or "",
            "file_path": row.get("file_path") or "unknown_source",
        }
        for row in relations
    ]
    chunk_context = [
        {"reference_id": index, "content": row["content"]}
        for index, row in enumerate(chunks, start=1)
    ]
    reference_list = [
        {"reference_id": index, "file_path": row.get("source_name") or row.get("source_key")}
        for index, row in enumerate(chunks, start=1)
    ]
    prompts = _lightrag_prompts()
    context_text = str(prompts["kg_query_context"]).format(
        entities_str="\n".join(json.dumps(item, ensure_ascii=False) for item in entity_context),
        relations_str="\n".join(json.dumps(item, ensure_ascii=False) for item in relation_context),
        text_chunks_str="\n".join(json.dumps(item, ensure_ascii=False) for item in chunk_context),
        reference_list_str="\n".join(
            f"[{item['reference_id']}] {item['file_path']}" for item in reference_list
        ),
    )
    raw_data = {
        "status": "success" if entity_context or relation_context or chunk_context else "failure",
        "metadata": {"query_mode": mode},
        "data": {
            "entities": entity_context,
            "relationships": relation_context,
            "chunks": chunk_context,
            "references": reference_list,
        },
    }
    return context_text, raw_data


def _query_results(
    *,
    context_text: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[QueryResult, ...]:
    results: list[QueryResult] = [
        QueryResult(
            id="light_rag_context",
            text=context_text,
            kind="light_rag_context",
            metadata={
                "entity_count": len(entities),
                "relation_count": len(relations),
                "chunk_count": len(chunks),
            },
        )
    ]
    for row in chunks:
        results.append(
            QueryResult(
                id=str(row["chunk_id"]),
                text=str(row["content"]),
                kind="chunk",
                source=chunk_source(
                    document_id=row.get("document_id"),
                    object_key=row.get("source_key"),
                    object_name=row.get("source_name"),
                    object_type=row.get("source_file_type"),
                    chunk_ids=(row.get("chunk_id"),),
                    page_index=row.get("page_index"),
                    chunk_index=row.get("chunk_index"),
                    token_start=row.get("token_start"),
                    token_end=row.get("token_end"),
                ),
                metadata={"source": "light_rag"},
            )
        )
    return tuple(results)


async def _generate_answer(
    *,
    model: Any,
    request: QueryRequest,
    mode: str,
    context_text: str,
) -> tuple[str | None, dict[str, object]]:
    if not should_generate_answer(request, default=True):
        return None, {"answer_generation": "disabled"}
    if model is None:
        return None, {"answer_generation": "missing_language_model"}
    prompts = _lightrag_prompts()
    user_prompt = f"\n\n{request.options.get('user_prompt')}" if request.options.get("user_prompt") else "n/a"
    response_type = str(request.options.get("response_type") or "Multiple Paragraphs")
    prompt = str(prompts["rag_response"]).format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_text,
    )
    result = await model.invoke(
        ModelRequest(
            prompt="\n\n".join([prompt, "---User Query---", request.text]),
            options=ModelOptions(temperature=0.1),
            trace_context={"query_mode": mode, "stage": "answer_generation"},
        )
    )
    return result.text or None, {
        "answer_generation": "generated" if result.text else "empty",
        "answer_model": getattr(model, "model_name", ""),
    }


def _round_robin_entities(
    local_entities: list[dict[str, Any]],
    global_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _round_robin_by_key(local_entities, global_entities, key="entity_id")


def _round_robin_relations(
    local_relations: list[dict[str, Any]],
    global_relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _round_robin_by_key(local_relations, global_relations, key="relation_id")


def _round_robin_chunks(
    vector_chunks: list[dict[str, Any]],
    fact_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _round_robin_by_key(vector_chunks, fact_chunks, key="chunk_id")


def _round_robin_by_key(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(max(len(first), len(second))):
        for rows in (first, second):
            if index >= len(rows):
                continue
            value = str(rows[index].get(key) or "")
            if value and value not in seen:
                results.append(rows[index])
                seen.add(value)
    return results


def _decode_row(row: Mapping[str, Any], *, score: float | None = None, order: int | None = None) -> dict[str, Any]:
    decoded = dict(row)
    for key in ("source_ids", "file_paths", "properties", "metadata"):
        if key in decoded:
            decoded[f"{key}_json"] = _json_value(decoded.get(key))
    decoded["source_ids_list"] = _json_list(decoded.get("source_ids"))
    if score is not None:
        decoded["score"] = score
    if order is not None:
        decoded["order"] = order
    return decoded


def _json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_list(value: Any) -> list[str]:
    parsed = _json_value(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if isinstance(value, str):
        return [item for item in value.split("<SEP>") if item]
    return []


def _relation_limit(request: QueryRequest) -> int:
    return int(request.options.get("relation_top_k") or request.top_k * 2)


def _chunk_limit(request: QueryRequest) -> int:
    return int(request.options.get("chunk_top_k") or request.top_k)


def _tables(asset: SearchAsset) -> _TableNames:
    entities = _metadata_string(asset.metadata, "entities_table", default="light_rag_entities")
    relations = _metadata_string(asset.metadata, "relations_table", default="light_rag_relations")
    chunks = _metadata_string(asset.metadata, "chunks_table", default="light_rag_chunks")
    for name in (entities, relations, chunks):
        validate_identifier(name, field_name="light_rag table")
    return _TableNames(entities=entities, relations=relations, chunks=chunks)


def _metadata_string(metadata: Mapping[str, Any], key: str, *, default: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _store_ref_from_asset(store_key: str, *, kind: str) -> ComponentRef:
    prefix = f"stores.{kind}"
    if store_key == prefix:
        return store_ref(kind)
    suffix = store_key.removeprefix(f"{prefix}:")
    return store_ref(kind, suffix) if suffix != store_key else store_ref(kind)


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


def _lightrag_prompts() -> Mapping[str, Any]:
    if not hasattr(_lightrag_prompts, "_cache"):
        setattr(_lightrag_prompts, "_cache", _load_lightrag_prompts())
    return getattr(_lightrag_prompts, "_cache")


def _load_lightrag_prompts() -> Mapping[str, Any]:
    try:
        from lightrag.prompt import PROMPTS  # type: ignore

        return PROMPTS
    except Exception:
        pass
    repo_root = Path(__file__).resolve().parents[6]
    prompt_path = repo_root / "LightRAG" / "lightrag" / "prompt.py"
    spec = importlib.util.spec_from_file_location("_heta_lightrag_prompt_query", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load LightRAG prompt module from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.YAMLError = ValueError
        yaml_stub.safe_load = lambda content: None
        sys.modules["yaml"] = yaml_stub
    spec.loader.exec_module(module)
    prompts = getattr(module, "PROMPTS", None)
    if not isinstance(prompts, dict):
        raise RuntimeError("LightRAG prompt module does not expose PROMPTS")
    return prompts
