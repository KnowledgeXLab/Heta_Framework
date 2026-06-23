"""BM25/vector fusion with optional rerank model."""

from __future__ import annotations

from dataclasses import dataclass, replace

from heta_framework.common.models import RerankOptions, RerankRequest
from heta_framework.common.models.protocols import RerankModelProtocol
from heta_framework.kb.components import MissingComponentError
from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results_with_prompt,
    optional_language_model_from_context,
    should_generate_answer,
)
from heta_framework.kb.search.engines.answer_prompts import rerank_answer_prompt
from heta_framework.kb.search.engines._provenance import citations_from_results
from heta_framework.kb.search.engines._ranking import reciprocal_rank_fusion
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import model_ref


@dataclass(frozen=True)
class RerankSearchEngine:
    """Fuse Heta hybrid and keyword candidates, then rerank when available."""

    mode: str = "heta_rerank_search"
    candidate_modes: tuple[str, ...] = ("hybrid_search", "keyword_search")
    reranker_model: str | None = None
    vector_asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_vector_index")
    keyword_asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_text_index")
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="graph_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="graph_vector_index")
    language_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by rerank search."""
        return frozenset(
            {
                self.vector_asset_ref,
                self.keyword_asset_ref,
                self.graph_tables_ref,
                self.graph_vectors_ref,
            }
        )

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run candidate retrieval, RRF fusion, and optional reranking."""
        candidate_top_k = _candidate_top_k(request)
        candidate_request = _request_with_top_k(request, candidate_top_k)
        responses = [
            await context.query(mode, candidate_request)
            for mode in self.candidate_modes
        ]
        fused = reciprocal_rank_fusion(
            responses,
            k=_rrf_k(request),
            top_k=candidate_top_k,
        )
        reranker = _find_reranker(context, self.reranker_model)
        reranked = (
            await _rerank_results(reranker, request, fused)
            if reranker is not None and fused
            else fused[: request.top_k]
        )
        used_reranker = reranker is not None and bool(fused)
        answer, answer_metadata = await _generate_answer(
            context=context,
            request=request,
            results=tuple(reranked),
            mode=self.mode,
            language_model=self.language_model,
        )

        trace = ()
        if request.trace:
            trace = (
                *[event for response in responses for event in response.trace],
                QueryTraceEvent(
                    stage="heta_rerank_search",
                    message=(
                        "Fused Heta hybrid and keyword results, "
                        "then applied reranking when available."
                    ),
                    metadata={
                        "candidate_modes": self.candidate_modes,
                        "candidate_count": len(fused),
                        "result_count": len(reranked),
                        "used_reranker": used_reranker,
                        "reranker_model": getattr(reranker, "model_name", None),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=tuple(reranked),
            answer=answer,
            citations=citations_from_results(reranked),
            trace=trace,
            metadata={
                "candidate_modes": self.candidate_modes,
                "candidate_count": len(fused),
                "used_reranker": used_reranker,
                "reranker_model": getattr(reranker, "model_name", None),
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
        prompt=rerank_answer_prompt(request.text, results),
        trace_context={"query_mode": mode, "stage": "answer_generation"},
    )
    return answer or None, {
        "answer_generation": "generated" if answer else "empty",
        "answer_model": model.model_name,
    }
async def _rerank_results(
    reranker: RerankModelProtocol,
    request: QueryRequest,
    candidates: tuple[QueryResult, ...],
) -> tuple[QueryResult, ...]:
    rerank_result = await reranker.rerank(
        RerankRequest(
            query=request.text,
            documents=[candidate.text for candidate in candidates],
            options=RerankOptions(top_n=request.top_k),
            trace_context={"query_mode": "heta_rerank_search"},
        )
    )
    ordered: list[QueryResult] = []
    for item in rerank_result.rankings:
        candidate = candidates[item.index]
        ordered.append(
            replace(
                candidate,
                score=item.score,
                metadata={
                    **dict(candidate.metadata),
                    "rerank_score": item.score,
                    "reranker_model": rerank_result.model_name or reranker.model_name,
                    "pre_rerank_score": candidate.score,
                },
            )
        )
    return tuple(ordered[: request.top_k])


def _find_reranker(
    context: QueryContext,
    name: str | None,
) -> RerankModelProtocol | None:
    try:
        component = context.recipe.get_component(model_ref("reranker", name))
    except MissingComponentError:
        return None
    if not isinstance(component, RerankModelProtocol):
        raise TypeError("models.reranker must satisfy RerankModelProtocol")
    return component


def _candidate_top_k(request: QueryRequest) -> int:
    value = request.options.get("candidate_top_k")
    if isinstance(value, int) and value > 0:
        return value
    return min(max(request.top_k * 3, request.top_k), 50)


def _rrf_k(request: QueryRequest) -> int:
    value = request.options.get("rrf_k")
    if isinstance(value, int) and value > 0:
        return value
    return 60


def _request_with_top_k(request: QueryRequest, top_k: int) -> QueryRequest:
    return QueryRequest(
        text=request.text,
        mode=request.mode,
        top_k=top_k,
        filters=request.filters,
        options=request.options,
        trace=request.trace,
    )
