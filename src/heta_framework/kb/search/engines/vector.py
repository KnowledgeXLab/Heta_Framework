"""Vector search query engine."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.vector import VectorQuery, VectorStoreProtocol
from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, model_ref, store_ref


@dataclass(frozen=True)
class VectorSearchEngine:
    """Search chunk vectors produced by IndexVectors."""

    mode: str = "vector_search"
    asset_ref: SearchAssetRef = SearchAssetRef(kind="chunk_vector_index")
    embedding_model: str | None = None

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by vector search."""
        return frozenset({self.asset_ref})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Embed the query text and search the chunk vector index."""
        asset = context.assets.require(self.asset_ref)
        embedding_model = _require_embedding_model(
            context.recipe.get_component(model_ref("embedding", self.embedding_model))
        )
        vector_store = _require_vector_store(
            context.recipe.get_component(_store_ref_from_asset(asset.store))
        )

        embedding = await embedding_model.embed(
            EmbeddingRequest(
                texts=[request.text],
                trace_context={"query_mode": self.mode},
            )
        )
        vector = embedding.vectors[0]
        collection = _metadata_string(asset.metadata, "collection", default=asset.name)
        hits = await vector_store.search(
            collection,
            VectorQuery(
                vector=vector,
                top_k=request.top_k,
                filter=dict(request.filters) or None,
            ),
        )

        results = tuple(
            QueryResult(
                id=hit.id,
                text=hit.text or "",
                score=hit.score,
                kind="chunk",
                source=_source_from_metadata(hit.metadata or {}),
                metadata={
                    **(hit.metadata or {}),
                    "collection": collection,
                    "search_asset": asset.key,
                },
            )
            for hit in hits
            if hit.text
        )
        trace = ()
        if request.trace:
            trace = (
                QueryTraceEvent(
                    stage="vector_search",
                    message="Searched chunk vector index.",
                    metadata={
                        "collection": collection,
                        "top_k": request.top_k,
                        "result_count": len(results),
                    },
                ),
            )
        return QueryResponse(
            mode=self.mode,
            results=results,
            trace=trace,
            metadata={
                "collection": collection,
                "embedding_model": embedding.model_name or embedding_model.model_name,
            },
        )


def _store_ref_from_asset(store: str | None) -> ComponentRef:
    if store is None:
        return store_ref("vector")
    parts = store.split(".")
    if len(parts) == 2 and parts == ["stores", "vector"]:
        return store_ref("vector")
    if len(parts) == 3 and parts[:2] == ["stores", "vector"]:
        return store_ref("vector", parts[2])
    if store == "vector":
        return store_ref("vector")
    raise ValueError(f"chunk_vector_index asset must reference a vector store, got: {store}")


def _metadata_string(metadata: object, key: str, *, default: str) -> str:
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def _source_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    source: dict[str, object] = {}
    for key in (
        "document_id",
        "source_key",
        "source_name",
        "source_file_type",
        "page_index",
        "chunk_index",
        "token_start",
        "token_end",
    ):
        if key in metadata:
            source[key] = metadata[key]
    return source


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component
