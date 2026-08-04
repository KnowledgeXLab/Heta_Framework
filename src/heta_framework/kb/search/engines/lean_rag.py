"""LeanRAG bottom-up hierarchical query engine."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorSearchResult, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import optional_language_model_from_context
from heta_framework.kb.search.engines._provenance import citations_from_results
from heta_framework.kb.graphing.prompts import LEANRAG_PROMPTS
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


@dataclass(frozen=True)
class LeanRAGQueryEngine:
    """Answer using LeanRAG entity hits, parent chains, relations, communities, and text units."""

    mode: str = "lean_rag_query"
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="leanrag_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="leanrag_vector_index")
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
        sql_store = _require_sql_store(
            context.recipe.get_component(_store_ref_from_asset(tables_asset.store, kind="sql"))
        )
        vector_store = _require_vector_store(
            context.recipe.get_component(_store_ref_from_asset(vectors_asset.store, kind="vector"))
        )
        embedder = _require_embedding_model(
            context.recipe.get_component(model_ref("embedding", self.embedding_model))
        )
        tables = _tables(tables_asset)
        top_k = _top_k(request)
        level_mode = _level_mode(request)
        text_unit_k = _text_unit_k(request)
        vector_filter = _level_mode_filter(level_mode, request)
        entity_collection = _metadata_string(
            vectors_asset.metadata,
            "entity_collection",
            default=vectors_asset.name,
        )

        query_vector = (
            await embedder.embed(
                EmbeddingRequest(
                    texts=[request.text],
                    trace_context={"query_mode": self.mode, "purpose": "leanrag_query"},
                )
            )
        ).vectors[0]
        entity_hits = await vector_store.search(
            entity_collection,
            VectorQuery(vector=query_vector, top_k=top_k, filter=vector_filter),
        )

        entities_by_id = await _entities_by_id(sql_store, tables.entities)
        selected_entities = [
            {**entities_by_id[hit.id], "_score": hit.score}
            for hit in entity_hits
            if hit.id in entities_by_id
        ]
        relations = await _all_relations(sql_store, tables.relations)
        relation_by_pair = {
            tuple(sorted((row["src_tgt"], row["tgt_src"]))): row for row in relations
        }
        communities_by_name = await _communities_by_name(sql_store, tables.communities)
        chunks_by_hash = await _chunks_by_hash(sql_store, tables.chunks)

        parent_chains = {
            row["entity_name"]: _find_tree_root(row["entity_name"], entities_by_id)
            for row in selected_entities
        }
        reasoning_paths, reasoning_relations = _reasoning_paths(parent_chains, relation_by_pair)
        selected_communities = _aggregation_descriptions(reasoning_paths, communities_by_name)
        selected_chunks = _text_units(selected_entities, chunks_by_hash, text_unit_k)
        context_text = _build_context(
            entities=selected_entities,
            communities=selected_communities,
            reasoning_relations=reasoning_relations,
            chunks=selected_chunks,
        )

        result = QueryResult(
            id="leanrag_context",
            text=context_text,
            score=entity_hits[0].score if entity_hits else None,
            kind="leanrag_context",
            source={
                "document_ids": tuple(dict.fromkeys(row.get("document_id", "") for row in selected_chunks if row.get("document_id"))),
                "source_keys": tuple(dict.fromkeys(row.get("source_key", "") for row in selected_chunks if row.get("source_key"))),
                "chunk_ids": tuple(row["chunk_id"] for row in selected_chunks),
                "hash_codes": tuple(row["hash_code"] for row in selected_chunks),
                "entity_names": tuple(row["entity_name"] for row in selected_entities),
                "aggregate_entity_names": tuple(
                    row["entity_name"] for row in selected_entities if row.get("is_aggregate")
                ),
                "relation_ids": tuple(row["relation_id"] for row in reasoning_relations),
                "community_names": tuple(row["entity_name"] for row in selected_communities),
            },
            metadata={
                "entity_vector_hits": [_hit_metadata(hit) for hit in entity_hits],
                "res_entity": [row["entity_name"] for row in selected_entities],
                "parent_chains": parent_chains,
                "reasoning_paths": reasoning_paths,
                "reasoning_path_relation_ids": [row["relation_id"] for row in reasoning_relations],
                "selected_text_unit_hash_codes": [row["hash_code"] for row in selected_chunks],
            },
        )

        answer, answer_metadata = await _answer(
            request,
            context,
            context_text,
            language_model=self.language_model,
            prompts=self.prompts or LEANRAG_PROMPTS,
            mode=self.mode,
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage=self.mode,
                    message="Built LeanRAG bottom-up hierarchical context.",
                    metadata={
                        "entity_vector_hits": [_hit_metadata(hit) for hit in entity_hits],
                        "level_mode": level_mode,
                        "level_mode_filter": vector_filter,
                        "parent_chains": parent_chains,
                        "reasoning_paths": reasoning_paths,
                        "reasoning_path_relation_evidence": reasoning_relations,
                        "selected_communities": [row["entity_name"] for row in selected_communities],
                        "selected_text_units": [row["hash_code"] for row in selected_chunks],
                        "truncation_budgets": {"text_unit_k": text_unit_k, "top_k": top_k},
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
                "reasoning_path_count": len(reasoning_paths),
                "source_count": len(selected_chunks),
                "level_mode": level_mode,
                **answer_metadata,
            },
        )


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
            "children_list": _json_list(row.get("children")),
            "findings_list": _json_list(row.get("findings")),
            "is_aggregate": bool(row.get("is_aggregate")),
        }
        for row in rows
    }


async def _all_relations(sql_store: SQLStoreProtocol, table: str) -> list[dict[str, Any]]:
    return [
        {**row, "source_ids_list": _json_list(row.get("source_ids"))}
        for row in await sql_store.fetch_all(f"SELECT * FROM {table}")
    ]


async def _communities_by_name(sql_store: SQLStoreProtocol, table: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["entity_name"]): {
            **row,
            "findings_list": _json_list(row.get("findings")),
            "children_list": _json_list(row.get("children")),
            "source_ids_list": _json_list(row.get("source_ids")),
        }
        for row in await sql_store.fetch_all(f"SELECT * FROM {table}")
    }


async def _chunks_by_hash(sql_store: SQLStoreProtocol, table: str) -> dict[str, dict[str, Any]]:
    rows = await sql_store.fetch_all(f"SELECT * FROM {table}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row["hash_code"])] = row
        result[str(row["chunk_id"])] = row
    return result


def _find_tree_root(entity_name: str, entities_by_id: dict[str, dict[str, Any]]) -> list[str]:
    chain = [entity_name]
    current = entity_name
    max_depth = max((int(row.get("level") or 0) for row in entities_by_id.values()), default=0) + 3
    for _ in range(max_depth):
        row = entities_by_id.get(current)
        if not row:
            break
        parent = str(row.get("parent") or "")
        if not parent:
            break
        chain.append(parent)
        if parent == "root" or parent == current:
            break
        current = parent
    return chain


def _reasoning_paths(
    parent_chains: dict[str, list[str]],
    relation_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    paths: list[list[str]] = []
    evidence: list[dict[str, Any]] = []
    seen_relation_ids: set[str] = set()
    for left, right in combinations(parent_chains, 2):
        left_chain = parent_chains[left]
        right_chain = parent_chains[right]
        path = _join_parent_chains(left_chain, right_chain)
        paths.append(path)
        for source, target in combinations(dict.fromkeys(path), 2):
            relation = relation_by_pair.get(tuple(sorted((source, target))))
            if relation is not None and relation["relation_id"] not in seen_relation_ids:
                seen_relation_ids.add(relation["relation_id"])
                evidence.append(relation)
    return paths, evidence


def _join_parent_chains(left_chain: list[str], right_chain: list[str]) -> list[str]:
    right_positions = {name: index for index, name in enumerate(right_chain)}
    for left_index, name in enumerate(left_chain):
        if name in right_positions:
            return [*left_chain[: left_index + 1], *reversed(right_chain[: right_positions[name]])]
    return [*left_chain, *reversed(right_chain)]


def _aggregation_descriptions(
    reasoning_paths: list[list[str]],
    communities_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    names = dict.fromkeys(name for path in reasoning_paths for name in path)
    return [communities_by_name[name] for name in names if name in communities_by_name]


def _text_units(
    entities: list[dict[str, Any]],
    chunks_by_hash: dict[str, dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    chunk_ids = [
        source_id
        for entity in entities
        for source_id in entity.get("source_ids_list", ())
        if source_id in chunks_by_hash
    ]
    counts = Counter(chunk_ids)
    selected = [
        item for item, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True) if count > 1
    ][:k]
    used = set(selected)
    for item in chunk_ids:
        if item not in used:
            selected.append(item)
            used.add(item)
        if len(selected) == k:
            break
    return [chunks_by_hash[item] for item in selected]


def _build_context(
    *,
    entities: list[dict[str, Any]],
    communities: list[dict[str, Any]],
    reasoning_relations: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> str:
    entity_information = "entity_name\t\tparent\t\tdescription\n" + "\n".join(
        f"{row['entity_name']}\t\t{row.get('parent') or ''}\t\t{row.get('description') or ''}"
        for row in entities
    )
    aggregation_information = "entity_name\t\tentity_description\n" + "\n".join(
        f"{row['entity_name']}\t\t{row.get('entity_description') or ''}" for row in communities
    )
    reasoning_information = "\n".join(dict.fromkeys(str(row.get("description") or "") for row in reasoning_relations if row.get("description")))
    text_units = "\n".join(str(row.get("content") or row.get("text") or "") for row in chunks)
    return f"""
    entity_information:
    {entity_information}
    aggregation_entity_information:
    {aggregation_information}
    reasoning_path_information:
    {reasoning_information}
    text_units:
    {text_units}
    """


async def _answer(
    request: QueryRequest,
    context: QueryContext,
    context_text: str,
    *,
    language_model: str | None,
    prompts: dict[str, Any],
    mode: str,
) -> tuple[str | None, dict[str, Any]]:
    if request.options.get("only_need_context") is True:
        return None, {"answer_generation": "context_only"}
    generate_answer = request.options.get("generate_answer")
    if generate_answer is False:
        return None, {"answer_generation": "disabled"}
    model = optional_language_model_from_context(context, language_model)
    if model is None:
        return None, {"answer_generation": "missing_language_model"}
    prompt = str(prompts["rag_response"].format(context_data=context_text))
    result = await model.invoke(
        ModelRequest(
            prompt=f"{prompt}\n\nUser question:\n{request.text}",
            options=ModelOptions(temperature=0.1),
            trace_context={"query_mode": mode, "stage": "rag_response"},
        )
    )
    return result.text, {"answer_generation": "generated"}


def _level_mode_filter(level_mode: int, request: QueryRequest) -> dict[str, Any] | None:
    filters = dict(request.filters)
    if level_mode == 0:
        filters["level"] = 0
    elif level_mode == 1:
        filters["is_aggregate"] = True
    return filters or None


def _top_k(request: QueryRequest) -> int:
    return int(request.options.get("topk") or request.options.get("top_k") or request.top_k)


def _level_mode(request: QueryRequest) -> int:
    return int(request.options.get("level_mode", 2))


def _text_unit_k(request: QueryRequest) -> int:
    return int(request.options.get("text_unit_k", 5))


def _tables(asset: SearchAsset) -> _TableNames:
    return _TableNames(
        entities=_metadata_string(asset.metadata, "entities_table", default="leanrag_entities"),
        relations=_metadata_string(asset.metadata, "relations_table", default="leanrag_relations"),
        communities=_metadata_string(asset.metadata, "communities_table", default="leanrag_communities"),
        chunks=_metadata_string(asset.metadata, "chunks_table", default="leanrag_chunks"),
    )


def _metadata_string(metadata: dict[str, Any], key: str, *, default: str) -> str:
    value = metadata.get(key, default)
    text = str(value).strip()
    return text or default


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item for item in value.split("|") if item]
        return parsed if isinstance(parsed, list) else []
    return []


def _hit_metadata(hit: VectorSearchResult) -> dict[str, Any]:
    return {"id": hit.id, "score": hit.score, "metadata": dict(hit.metadata or {})}


def _store_ref_from_asset(store: str | None, *, kind: str) -> ComponentRef:
    if store is None:
        return store_ref(kind)
    parts = store.split(".")
    if len(parts) == 2 and parts[0] == "stores":
        return store_ref(parts[1])
    if len(parts) == 3 and parts[0] == "stores":
        return store_ref(parts[1], parts[2])
    return store_ref(kind)


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
