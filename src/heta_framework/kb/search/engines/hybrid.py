"""Vector and Heta graph hybrid search engine."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._ranking import weighted_reciprocal_rank_fusion
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryTraceEvent


@dataclass(frozen=True)
class HybridSearchEngine:
    """Fuse chunk vector search and Heta graph search with weighted RRF."""

    mode: str = "hybrid_search"
    candidate_modes: tuple[str, ...] = ("vector_search", "heta_graph_search")
    vector_asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_vector_index")
    graph_tables_ref: SearchAssetRef = SearchAssetRef(kind="graph_tables")
    graph_vectors_ref: SearchAssetRef = SearchAssetRef(kind="graph_vector_index")

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by hybrid search."""
        return frozenset(
            {
                self.vector_asset_ref,
                self.graph_tables_ref,
                self.graph_vectors_ref,
            }
        )

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run vector and graph search, then fuse the ranked lists."""
        candidate_top_k = _candidate_top_k(request)
        candidate_request = _request_with_top_k(request, candidate_top_k)
        responses = [
            await context.query(mode, candidate_request)
            for mode in self.candidate_modes
        ]
        weights = _weights(request)
        fused = weighted_reciprocal_rank_fusion(
            responses,
            weights=weights,
            k=_rrf_k(request),
            top_k=request.top_k,
        )

        trace = ()
        if request.trace:
            trace = (
                *[event for response in responses for event in response.trace],
                QueryTraceEvent(
                    stage="hybrid_search",
                    message="Fused vector and Heta graph search results.",
                    metadata={
                        "candidate_modes": self.candidate_modes,
                        "candidate_top_k": candidate_top_k,
                        "weights": weights,
                        "result_count": len(fused),
                        "fusion": "weighted_rrf",
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=fused,
            trace=trace,
            metadata={
                "candidate_modes": self.candidate_modes,
                "candidate_top_k": candidate_top_k,
                "weights": weights,
                "fusion": "weighted_rrf",
            },
        )


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


def _weights(request: QueryRequest) -> dict[str, float]:
    default = {"vector_search": 1.0, "heta_graph_search": 1.0}
    value = request.options.get("hybrid_weights")
    if not isinstance(value, dict):
        return default
    weights = dict(default)
    for mode, weight in value.items():
        if isinstance(mode, str) and isinstance(weight, (int, float)) and weight > 0:
            weights[mode] = float(weight)
    return weights


def _request_with_top_k(request: QueryRequest, top_k: int) -> QueryRequest:
    return QueryRequest(
        text=request.text,
        mode=request.mode,
        top_k=top_k,
        filters=request.filters,
        options=request.options,
        trace=request.trace,
    )
