"""Keyword search query engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results_with_prompt,
    optional_language_model_from_context,
    should_generate_answer,
)
from heta_framework.kb.search.engines.answer_prompts import keyword_answer_prompt
from heta_framework.kb.search.engines._provenance import chunk_source, citations_from_results
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, store_ref


@dataclass(frozen=True)
class KeywordSearchEngine:
    """Search persisted chunk text produced by PersistChunks."""

    mode: str = "keyword_search"
    asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_text_index")
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by keyword search."""
        return frozenset({self.asset_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Search persisted chunk text with the SQL strategy declared by the asset."""
        asset = context.assets.require(self.asset_ref)
        sql_store = _require_sql_store(
            context.recipe.get_component(_store_ref_from_asset(asset.store))
        )
        dialect = _metadata_string(asset.metadata, "dialect", default="generic")
        table = _metadata_string(asset.metadata, "table", default=asset.name)
        if dialect == "postgresql":
            rows = await sql_store.fetch_all(
                _postgres_query(table),
                {"query": request.text, "limit": request.top_k},
            )
        else:
            rows = await sql_store.fetch_all(
                _generic_query(table),
                {
                    "query": request.text,
                    "pattern": f"%{request.text}%",
                    "limit": request.top_k,
                },
            )

        results = tuple(_row_to_result(row, asset=asset, dialect=dialect) for row in rows)
        answer, answer_metadata = await _generate_answer(
            context=context,
            request=request,
            results=results,
            mode=self.mode,
            language_model=self.language_model,
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage="keyword_search",
                    message="Searched persisted chunk text.",
                    metadata={
                        "table": table,
                        "dialect": dialect,
                        "top_k": request.top_k,
                        "result_count": len(results),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=results,
            answer=answer,
            citations=citations_from_results(results),
            trace=trace,
            metadata={"table": table, "dialect": dialect, **answer_metadata},
        )


async def _generate_answer(
    *,
    context: QueryContext,
    request: QueryRequest,
    results: tuple[QueryResult, ...],
    mode: str,
    language_model: str | None,
) -> tuple[str | None, dict[str, object]]:
    if not should_generate_answer(request):
        return None, {"answer_generation": "disabled"}
    model = optional_language_model_from_context(context, language_model)
    if model is None:
        return None, {
            "answer_generation": "missing_language_model",
            "answer_generation_requested": True,
        }
    answer = await answer_from_results_with_prompt(
        model,
        query=request.text,
        results=results,
        prompt=keyword_answer_prompt(request.text, results),
        trace_context={"query_mode": mode, "stage": "answer_generation"},
    )
    return answer or None, {
        "answer_generation": "generated" if answer else "empty",
        "answer_model": model.model_name,
    }
def _postgres_query(table: str) -> str:
    _validate_identifier(table, field_name="table")
    return f"""
    SELECT
        chunk_id,
        document_id,
        content_text,
        source_id,
        source_chunk,
        metadata_json,
        ts_rank(content_tsv, plainto_tsquery('simple', :query)) AS score
    FROM {table}
    WHERE content_tsv @@ plainto_tsquery('simple', :query)
    ORDER BY score DESC, chunk_id ASC
    LIMIT :limit
    """


def _generic_query(table: str) -> str:
    _validate_identifier(table, field_name="table")
    return f"""
    SELECT
        chunk_id,
        document_id,
        content_text,
        source_id,
        source_chunk,
        metadata_json,
        CASE
            WHEN LOWER(content_text) = LOWER(:query) THEN 1.0
            ELSE 0.5
        END AS score
    FROM {table}
    WHERE LOWER(content_text) LIKE LOWER(:pattern)
    ORDER BY score DESC, chunk_id ASC
    LIMIT :limit
    """


def _row_to_result(row: dict[str, Any], *, asset: SearchAsset, dialect: str) -> QueryResult:
    metadata = _metadata_from_json(row.get("metadata_json"))
    source_metadata = metadata.get("source")
    source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
    source = chunk_source(
        document_id=row.get("document_id"),
        object_key=row.get("source_id") or source_metadata.get("key"),
        object_name=source_metadata.get("name"),
        object_type=source_metadata.get("file_type"),
        chunk_ids=_source_chunk_ids(row.get("source_chunk")) or (row.get("chunk_id"),),
        page_index=metadata.get("page_index"),
        chunk_index=metadata.get("chunk_index"),
        token_start=metadata.get("token_start"),
        token_end=metadata.get("token_end"),
    )
    return QueryResult(
        id=str(row["chunk_id"]),
        text=str(row["content_text"]),
        score=_score(row.get("score")),
        kind="chunk",
        source=source,
        metadata={
            **metadata,
            "dialect": dialect,
            "search_asset": asset.key,
            "table": asset.name,
        },
    )


def _metadata_from_json(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_chunk_ids(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _store_ref_from_asset(store: str | None) -> ComponentRef:
    if store is None:
        return store_ref("sql")
    parts = store.split(".")
    if len(parts) == 2 and parts == ["stores", "sql"]:
        return store_ref("sql")
    if len(parts) == 3 and parts[:2] == ["stores", "sql"]:
        return store_ref("sql", parts[2])
    if store == "sql":
        return store_ref("sql")
    raise ValueError(f"chunk_text_index asset must reference a SQL store, got: {store}")


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def _validate_identifier(value: str, *, field_name: str) -> None:
    if value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    if not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(f"{field_name} must be a simple SQL identifier")


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component
