"""Full-text search query engine."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.common.stores.text_index import TextIndexStoreProtocol, TextQuery
from heta_framework.kb.search.assets import SearchAsset, SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results_with_prompt,
    optional_language_model_from_context,
    should_generate_answer,
)
from heta_framework.kb.search.engines._provenance import chunk_source, citations_from_results
from heta_framework.kb.search.engines.answer_prompts import keyword_answer_prompt
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, store_ref


@dataclass(frozen=True)
class FullTextSearchEngine:
    """Search chunks indexed in a full-text index store."""

    mode: str = "full_text_search"
    asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_full_text_index")
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by full-text search."""
        return frozenset({self.asset_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Search the chunk full-text index."""
        asset = context.assets.require(self.asset_ref)
        text_index_store = _require_text_index_store(
            context.recipe.get_component(_store_ref_from_asset(asset.store))
        )
        index = _metadata_string(asset.metadata, "index", default=asset.name)
        hits = await text_index_store.search(
            index,
            TextQuery(
                text=request.text,
                top_k=request.top_k,
                filters=dict(request.filters) or None,
            ),
        )
        results = tuple(_hit_to_result(hit, asset=asset) for hit in hits)
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
                    stage="full_text_search",
                    message="Searched chunk full-text index.",
                    metadata={
                        "index": index,
                        "ranking": _metadata_string(asset.metadata, "ranking", default="bm25"),
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
            metadata={
                "index": index,
                "retrieval_method": "full_text_search",
                "ranking": _metadata_string(asset.metadata, "ranking", default="bm25"),
                **answer_metadata,
            },
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


def _hit_to_result(hit: object, *, asset: SearchAsset) -> QueryResult:
    hit_id = getattr(hit, "id")
    hit_text = getattr(hit, "text")
    score = getattr(hit, "score")
    metadata = dict(getattr(hit, "metadata"))
    source = chunk_source(
        document_id=metadata.get("document_id"),
        object_key=metadata.get("source_key"),
        object_name=metadata.get("source_name"),
        object_type=metadata.get("source_file_type"),
        chunk_ids=tuple(metadata.get("parent_chunk_ids") or (hit_id,)),
        page_index=metadata.get("page_index"),
        chunk_index=metadata.get("chunk_index"),
        token_start=metadata.get("token_start"),
        token_end=metadata.get("token_end"),
    )
    return QueryResult(
        id=str(hit_id),
        text=str(hit_text),
        score=_score(score),
        kind="chunk",
        source=source,
        metadata={
            **metadata,
            "ranking": _metadata_string(asset.metadata, "ranking", default="bm25"),
            "retrieval_method": "full_text_search",
            "search_asset": asset.key,
            "index": asset.name,
        },
    )


def _store_ref_from_asset(store: str | None) -> ComponentRef:
    if store is None:
        return store_ref("text_index")
    parts = store.split(".")
    if len(parts) == 2 and parts == ["stores", "text_index"]:
        return store_ref("text_index")
    if len(parts) == 3 and parts[:2] == ["stores", "text_index"]:
        return store_ref("text_index", parts[2])
    if store == "text_index":
        return store_ref("text_index")
    raise ValueError(f"chunk_full_text_index asset must reference a text index store, got: {store}")


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def _score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _require_text_index_store(component: object) -> TextIndexStoreProtocol:
    if not isinstance(component, TextIndexStoreProtocol):
        raise TypeError("stores.text_index must satisfy TextIndexStoreProtocol")
    return component
