"""Generate embeddings for ParsedChunk artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class EmbedChunksConfig:
    """Configuration for EmbedChunks."""

    embeddings_prefix: str = "embeddings"
    batch_size: int = 64
    object_store: str | None = None
    embedding_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.embeddings_prefix)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")


@dataclass(frozen=True)
class EmbedChunksResult:
    """Artifacts produced by EmbedChunks."""

    embedding_keys: tuple[str, ...]
    chunk_count: int
    model_name: str
    dimension: int


class EmbedChunks:
    """Generate embedding vectors for parsed chunks."""

    name = "embed_chunks"

    def __init__(self, config: EmbedChunksConfig | None = None) -> None:
        self.config = config or EmbedChunksConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    model_ref("embedding", self.config.embedding_model),
                }
            ),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"embed_chunks_result", "chunk_embedding_keys"})
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the embedding step and store ChunkEmbedding JSON objects."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))

        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        embedding_keys: list[str] = []
        dimension = 0

        for start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[start : start + self.config.batch_size]
            result = await embedding_model.embed(
                EmbeddingRequest(
                    texts=[chunk.text for chunk in batch],
                    trace_context={"step": self.name},
                )
            )
            if len(result.vectors) != len(batch):
                raise ValueError("embedding result count must match chunk batch size")
            for chunk, vector in zip(batch, result.vectors, strict=True):
                if not vector:
                    raise ValueError(f"embedding vector must not be empty for chunk: {chunk.chunk_id}")
                dimension = len(vector)
                embedding = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    model_name=result.model_name or embedding_model.model_name,
                    vector=[float(value) for value in vector],
                    dimension=dimension,
                )
                key = join_object_key(self.config.embeddings_prefix, f"{chunk.chunk_id}.json")
                await object_store.put(key, embedding.to_json_bytes())
                embedding_keys.append(key)

        model_name = embedding_model.model_name
        output = EmbedChunksResult(
            embedding_keys=tuple(embedding_keys),
            chunk_count=len(chunks),
            model_name=model_name,
            dimension=dimension,
        )
        context.set_artifact("embed_chunks_result", output)
        context.set_artifact("chunk_embedding_keys", output.embedding_keys)


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
