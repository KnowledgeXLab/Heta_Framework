"""Query rewriting search engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results_with_prompt,
    invoke_json,
    language_model_from_context,
    optional_language_model_from_context,
    should_generate_answer,
)
from heta_framework.kb.search.engines.answer_prompts import rewrite_answer_prompt
from heta_framework.kb.search.engines._provenance import citations_from_results
from heta_framework.kb.search.engines._ranking import reciprocal_rank_fusion
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, model_ref


@dataclass(frozen=True)
class RewriteSearchEngine:
    """Generate query variants and aggregate retrieval results."""

    mode: str = "heta_rewrite_search"
    base_mode: str = "heta_rerank_search"
    language_model: str | None = None
    required_asset_refs: frozenset[SearchAssetRef] = frozenset(
        {
            SearchAssetRef(kind="chunk_vector_index"),
            SearchAssetRef(kind="chunk_full_text_index"),
            SearchAssetRef(kind="graph_tables"),
            SearchAssetRef(kind="graph_vector_index"),
        }
    )

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by the configured base search mode."""
        return self.required_asset_refs

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        """Return components required by query rewriting."""
        return frozenset({model_ref("language", self.language_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Rewrite the query, run retrieval for each variant, and fuse results."""
        language_model = language_model_from_context(context, self.language_model)
        rewrite_result = await _rewrite_query(language_model, request, max_variations=3)
        if not rewrite_result.queries:
            base_response = await context.query(self.base_mode, request)
            return QueryResponse(
                mode=self.mode,
                results=base_response.results,
                answer=base_response.answer,
                citations=base_response.citations,
                trace=base_response.trace,
                metadata={
                    "base_mode": self.base_mode,
                    "variations": (),
                    "fallback": True,
                    "issues": (
                        {
                            "code": rewrite_result.issue_code,
                            "message": rewrite_result.issue_message,
                            "action": "used_base_search",
                        },
                    ),
                },
            )

        variant_requests = [
            QueryRequest(
                text=variation,
                mode=self.base_mode,
                top_k=request.top_k,
                filters=request.filters,
                options=request.options,
                trace=request.trace,
            )
            for variation in rewrite_result.queries
        ]
        responses = await asyncio.gather(
            *(
                context.query(self.base_mode, variant_request)
                for variant_request in variant_requests
            )
        )
        fused = reciprocal_rank_fusion(
            list(responses),
            k=_rrf_k(request),
            top_k=request.top_k,
        )
        answer, answer_metadata = await _generate_answer(
            context=context,
            request=request,
            results=fused,
            mode=self.mode,
            language_model=self.language_model,
        )

        trace = ()
        if request.trace:
            trace = (
                *[event for response in responses for event in response.trace],
                QueryTraceEvent(
                    stage="heta_rewrite_search",
                    message="Generated query variants and fused retrieval results.",
                    metadata={
                        "base_mode": self.base_mode,
                        "variations": tuple(rewrite_result.queries),
                        "result_count": len(fused),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=fused,
            answer=answer,
            citations=citations_from_results(fused),
            trace=trace,
            metadata={
                "base_mode": self.base_mode,
                "variations": tuple(rewrite_result.queries),
                "issues": (),
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
        prompt=rewrite_answer_prompt(request.text, results),
        trace_context={"query_mode": mode, "stage": "answer_generation"},
    )
    return answer or None, {
        "answer_generation": "generated" if answer else "empty",
        "answer_model": model.model_name,
    }
@dataclass(frozen=True)
class RewriteResult:
    """Structured result from query rewriting."""

    queries: tuple[str, ...]
    issue_code: str | None = None
    issue_message: str | None = None


async def _rewrite_query(
    language_model: object,
    request: QueryRequest,
    *,
    max_variations: int,
) -> RewriteResult:
    data = await invoke_json(
        language_model,
        prompt=(
            "Generate alternative search queries for the user question.\n"
            "Return JSON only with this shape: {\"queries\": [\"...\", \"...\", \"...\"]}.\n"
            "Keep acronyms and domain-specific terms unchanged. Do not explain.\n\n"
            f"User question: {request.text}"
        ),
        trace_context={"query_mode": "heta_rewrite_search"},
    )
    queries = data.get("queries")
    if not isinstance(queries, list):
        return RewriteResult(
            queries=(),
            issue_code="rewrite_invalid_output",
            issue_message="Language model did not return a queries list.",
        )
    clean: list[str] = []
    for item in queries:
        if not isinstance(item, str):
            continue
        query = item.strip()
        if query and query not in clean:
            clean.append(query)
    if not clean:
        return RewriteResult(
            queries=(),
            issue_code="rewrite_empty_output",
            issue_message="Language model returned no usable query variants.",
        )
    return RewriteResult(queries=tuple(clean[:max_variations]))


def _rrf_k(request: QueryRequest) -> int:
    value = request.options.get("rrf_k")
    if isinstance(value, int) and value > 0:
        return value
    return 60
