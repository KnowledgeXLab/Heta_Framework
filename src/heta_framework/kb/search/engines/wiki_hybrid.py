"""Hybrid semantic and lexical retrieval for generated wiki chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._provenance import citations_from_results
from heta_framework.kb.search.engines._ranking import reciprocal_rank_fusion
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryTraceEvent


@dataclass(frozen=True)
class WikiHybridSearchEngine:
    """Fuse wiki vector and full-text retrieval with reciprocal rank fusion."""

    mode: str = "wiki_hybrid_search"
    candidate_modes: tuple[str, ...] = (
        "wiki_vector_search",
        "wiki_full_text_search",
    )
    vector_asset_ref: SearchAssetRef = SearchAssetRef(kind="wiki_chunk_vector_index")
    full_text_asset_ref: SearchAssetRef = SearchAssetRef(
        kind="wiki_chunk_full_text_index"
    )
    discoverable: bool = False

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return wiki vector and full-text index requirements."""
        return frozenset({self.vector_asset_ref, self.full_text_asset_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run both wiki retrieval modes and fuse their ranked results."""
        candidate_top_k = _candidate_top_k(request)
        responses = await asyncio.gather(
            *(
                context.query(
                    mode,
                    QueryRequest(
                        text=request.text,
                        mode=mode,
                        top_k=candidate_top_k,
                        filters=request.filters,
                        options={**request.options, "generate_answer": False},
                        trace=request.trace,
                    ),
                )
                for mode in self.candidate_modes
            )
        )
        fused = reciprocal_rank_fusion(
            list(responses),
            k=_rrf_k(request),
            top_k=request.top_k,
        )
        trace = ()
        if request.trace:
            trace = (
                *(event for response in responses for event in response.trace),
                QueryTraceEvent(
                    stage=self.mode,
                    message="Fused wiki vector and full-text retrieval results.",
                    metadata={
                        "candidate_modes": self.candidate_modes,
                        "candidate_top_k": candidate_top_k,
                        "result_count": len(fused),
                        "fusion": "rrf",
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=fused,
            citations=citations_from_results(fused),
            trace=trace,
            metadata={
                "candidate_modes": self.candidate_modes,
                "candidate_top_k": candidate_top_k,
                "fusion": "rrf",
            },
        )


def _candidate_top_k(request: QueryRequest) -> int:
    value = request.options.get("candidate_top_k")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return min(value, 50)
    return min(max(request.top_k * 3, request.top_k), 50)


def _rrf_k(request: QueryRequest) -> int:
    value = request.options.get("rrf_k")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 60
