"""Index chunk embeddings into a vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

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


IndexVectorsPreset = Literal["default", "wiki"]


@dataclass(frozen=True)
class ChunkVectorCollections:
    """Vector collection names used by chunk indexing."""

    chunks: str = "chunks"

    def __post_init__(self) -> None:
        if self.chunks.strip() == "":
            raise ValueError("collection_names.chunks must not be empty")


@dataclass(frozen=True)
class _IndexVectorsPreset:
    collection: str
    chunk_keys_artifact: str
    chunk_embedding_keys_artifact: str
    result_artifact: str
    query_mode: str
    asset_kind: str


_INDEX_VECTORS_PRESETS: dict[IndexVectorsPreset, _IndexVectorsPreset] = {
    "default": _IndexVectorsPreset(
        collection="chunks",
        chunk_keys_artifact="chunk_keys",
        chunk_embedding_keys_artifact="chunk_embedding_keys",
        result_artifact="index_vectors_result",
        query_mode="vector_search",
        asset_kind="chunk_vector_index",
    ),
    "wiki": _IndexVectorsPreset(
        collection="wiki_chunks",
        chunk_keys_artifact="wiki_chunk_keys",
        chunk_embedding_keys_artifact="wiki_chunk_embedding_keys",
        result_artifact="index_wiki_vectors_result",
        query_mode="wiki_vector_search",
        asset_kind="wiki_chunk_vector_index",
    ),
}


@dataclass(frozen=True)
class IndexVectorsConfig:
    """Configuration for IndexVectors."""

    collection_names: ChunkVectorCollections | None = None
    metric: str = "cosine"
    batch_size: int = 128
    object_store: str | None = None
    vector_store: str | None = None
    chunk_keys_artifact: str | None = None
    chunk_embedding_keys_artifact: str | None = None
    preset: IndexVectorsPreset = "default"
    result_artifact: str | None = None
    query_mode: str | None = None
    asset_kind: str | None = None

    def __post_init__(self) -> None:
        preset = _index_vectors_preset(self.preset)
        if self.collection_names is None:
            object.__setattr__(
                self,
                "collection_names",
                ChunkVectorCollections(chunks=preset.collection),
            )
        _set_preset_value(
            self,
            "chunk_keys_artifact",
            self.chunk_keys_artifact,
            preset.chunk_keys_artifact,
            allow_custom=self.preset == "default",
        )
        _set_preset_value(
            self,
            "chunk_embedding_keys_artifact",
            self.chunk_embedding_keys_artifact,
            preset.chunk_embedding_keys_artifact,
            allow_custom=self.preset == "default",
        )
        _set_preset_value(
            self,
            "result_artifact",
            self.result_artifact,
            preset.result_artifact,
        )
        _set_preset_value(
            self,
            "query_mode",
            self.query_mode,
            preset.query_mode,
        )
        _set_preset_value(
            self,
            "asset_kind",
            self.asset_kind,
            preset.asset_kind,
        )
        if self.metric not in {"cosine", "dot", "l2"}:
            raise ValueError("metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


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
            artifacts=frozenset({self.config.result_artifact}),
            queries=frozenset({self.config.query_mode}),
            search_assets=(
                SearchAsset(
                    kind=self.config.asset_kind,
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
                        "heading_path": " > ".join(chunk.heading_path),
                        "embedding_model": embedding.model_name,
                        **_origin_source_metadata(chunk),
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
        context.set_artifact(self.config.result_artifact, output)


def _origin_source_metadata(chunk: ParsedChunk) -> dict[str, str]:
    if chunk.origin_source is None:
        return {}
    return {
        "origin_source_key": chunk.origin_source.key,
        "origin_source_name": chunk.origin_source.name,
        "origin_source_file_type": chunk.origin_source.file_type,
        "origin_source_content_sha256": chunk.origin_source.content_sha256,
    }


def _index_vectors_preset(preset: str) -> _IndexVectorsPreset:
    try:
        return _INDEX_VECTORS_PRESETS[preset]  # type: ignore[index]
    except KeyError as exc:
        allowed = ", ".join(sorted(_INDEX_VECTORS_PRESETS))
        raise ValueError(f"preset must be one of: {allowed}") from exc


def _set_preset_value(
    config: object,
    field_name: str,
    value: str | None,
    expected: str,
    *,
    allow_custom: bool = False,
) -> None:
    if value is None:
        object.__setattr__(config, field_name, expected)
        return
    if value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    if value != expected and not allow_custom:
        raise ValueError(
            f"{field_name} must be {expected!r} for preset {getattr(config, 'preset')!r}"
        )
    object.__setattr__(config, field_name, value)


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component
