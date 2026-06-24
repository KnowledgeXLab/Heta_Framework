"""Index chunk embeddings into a vector store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.vector import (
    VectorCollectionConfig,
    VectorRecord,
    VectorStoreProtocol,
)
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, store_ref


@dataclass(frozen=True)
class ChunkVectorCollections:
    """Vector collection names used by chunk indexing."""

    chunks: str = "chunks"

    def __post_init__(self) -> None:
        if self.chunks.strip() == "":
            raise ValueError("collection_names.chunks must not be empty")


@dataclass(frozen=True)
class IndexVectorsConfig:
    """Configuration for IndexVectors."""

    collection_names: ChunkVectorCollections = field(default_factory=ChunkVectorCollections)
    metric: str = "cosine"
    batch_size: int = 128
    object_store: str | None = None
    vector_store: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    chunk_embedding_keys_artifact: str = "chunk_embedding_keys"

    def __post_init__(self) -> None:
        if self.metric not in {"cosine", "dot", "l2"}:
            raise ValueError("metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.chunk_embedding_keys_artifact.strip() == "":
            raise ValueError("chunk_embedding_keys_artifact must not be empty")


@dataclass(frozen=True)
class IndexVectorsResult:
    """Artifacts produced by IndexVectors."""

    collection: str
    indexed_count: int
    dimension: int


class IndexVectors:
    """Write chunk vectors into a VectorStore and enable vector search."""

    name = "index_vectors"

    def __init__(self, config: IndexVectorsConfig | None = None) -> None:
        self.config = config or IndexVectorsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("vector", self.config.vector_store),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.chunk_keys_artifact,
                    self.config.chunk_embedding_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts and query modes produced by this step."""
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset({"index_vectors_result"}),
            queries=frozenset({"vector_search"}),
            search_assets=(
                SearchAsset(
                    kind="chunk_vector_index",
                    name=self.config.collection_names.chunks,
                    store=vector_store_ref.key,
                    metadata={
                        "collection": self.config.collection_names.chunks,
                        "id_field": "id",
                        "text_field": "text",
                        "metadata_field": "metadata",
                    },
                ),
            ),
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return vector collections produced by this step."""
        return StepCleanupPlan(
            (
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.collection_names.chunks,
                    component=store_ref("vector", self.config.vector_store).key,
                ),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the indexing step and upsert records into the vector store."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        vector_store = _require_vector_store(
            context.get_component(store_ref("vector", self.config.vector_store).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        embedding_keys = tuple(context.get_artifact(self.config.chunk_embedding_keys_artifact))

        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        embeddings = [
            ChunkEmbedding.from_json(await object_store.get(key)) for key in embedding_keys
        ]
        embedding_by_chunk_id = {embedding.chunk_id: embedding for embedding in embeddings}
        if len(embedding_by_chunk_id) != len(embeddings):
            raise ValueError("chunk embedding keys must not contain duplicate chunk ids")

        records: list[VectorRecord] = []
        dimension = 0
        for chunk in chunks:
            try:
                embedding = embedding_by_chunk_id[chunk.chunk_id]
            except KeyError as exc:
                raise ValueError(f"missing embedding for chunk: {chunk.chunk_id}") from exc
            if embedding.document_id != chunk.document_id:
                raise ValueError(f"embedding document_id mismatch for chunk: {chunk.chunk_id}")
            dimension = embedding.dimension
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=embedding.vector,
                    text=chunk.text,
                    metadata={
                        "document_id": chunk.document_id,
                        "source_key": chunk.source.key,
                        "source_name": chunk.source.name,
                        "source_file_type": chunk.source.file_type,
                        "page_index": chunk.page_index,
                        "chunk_index": chunk.chunk_index,
                        "token_start": chunk.token_start,
                        "token_end": chunk.token_end,
                        "parent_chunk_ids": list(chunk.parent_chunk_ids),
                        "embedding_model": embedding.model_name,
                    },
                )
            )

        if records:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.collection_names.chunks,
                    dimension=dimension,
                    metric=self.config.metric,  # type: ignore[arg-type]
                )
            )
            for start in range(0, len(records), self.config.batch_size):
                await vector_store.upsert(
                    self.config.collection_names.chunks,
                    records[start : start + self.config.batch_size],
                )

        output = IndexVectorsResult(
            collection=self.config.collection_names.chunks,
            indexed_count=len(records),
            dimension=dimension,
        )
        context.set_artifact("index_vectors_result", output)


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component
