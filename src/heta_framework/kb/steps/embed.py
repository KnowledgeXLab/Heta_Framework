"""Generate embeddings for ParsedChunk artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from heta_framework.common.models import EmbeddingRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


EmbedChunksPreset = Literal["default", "wiki"]


@dataclass(frozen=True)
class _EmbedChunksPreset:
    embeddings_prefix: str
    chunk_keys_artifact: str
    chunk_embedding_keys_artifact: str
    result_artifact: str


_EMBED_CHUNKS_PRESETS: dict[EmbedChunksPreset, _EmbedChunksPreset] = {
    "default": _EmbedChunksPreset(
        embeddings_prefix="embeddings",
        chunk_keys_artifact="chunk_keys",
        chunk_embedding_keys_artifact="chunk_embedding_keys",
        result_artifact="embed_chunks_result",
    ),
    "wiki": _EmbedChunksPreset(
        embeddings_prefix="wiki_embeddings",
        chunk_keys_artifact="wiki_chunk_keys",
        chunk_embedding_keys_artifact="wiki_chunk_embedding_keys",
        result_artifact="embed_wiki_chunks_result",
    ),
}


@dataclass(frozen=True)
class EmbedChunksConfig:
    """Configuration for EmbedChunks."""

    embeddings_prefix: str | None = None
    batch_size: int = 10
    object_store: str | None = None
    embedding_model: str | None = None
    chunk_keys_artifact: str | None = None
    preset: EmbedChunksPreset = "default"
    chunk_embedding_keys_artifact: str | None = None
    result_artifact: str | None = None

    def __post_init__(self) -> None:
        preset = _embed_chunks_preset(self.preset)
        if self.embeddings_prefix is None:
            object.__setattr__(self, "embeddings_prefix", preset.embeddings_prefix)
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
        )
        _set_preset_value(
            self,
            "result_artifact",
            self.result_artifact,
            preset.result_artifact,
        )
        validate_object_prefix(self.embeddings_prefix)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


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
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.chunk_embedding_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return embedding objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.chunk_embedding_keys_artifact,
                component=store_ref("objects", self.config.object_store).key,
            )
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
        chunks_to_embed: list[ParsedChunk] = []

        for chunk in chunks:
            key = join_object_key(self.config.embeddings_prefix, f"{chunk.chunk_id}.json")
            if await object_store.exists(key):
                embedding = ChunkEmbedding.from_json(await object_store.get(key))
                if embedding.document_id != chunk.document_id:
                    raise ValueError(f"embedding document_id mismatch for chunk: {chunk.chunk_id}")
                if embedding.dimension <= 0:
                    raise ValueError(
                        f"embedding dimension must be positive for chunk: {chunk.chunk_id}"
                    )
                dimension = embedding.dimension
                embedding_keys.append(key)
                continue
            chunks_to_embed.append(chunk)

        for start in range(0, len(chunks_to_embed), self.config.batch_size):
            batch = chunks_to_embed[start : start + self.config.batch_size]
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
                    raise ValueError(
                        f"embedding vector must not be empty for chunk: {chunk.chunk_id}"
                    )
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
        context.set_artifact(self.config.result_artifact, output)
        context.set_artifact(self.config.chunk_embedding_keys_artifact, output.embedding_keys)


def _embed_chunks_preset(preset: str) -> _EmbedChunksPreset:
    try:
        return _EMBED_CHUNKS_PRESETS[preset]  # type: ignore[index]
    except KeyError as exc:
        allowed = ", ".join(sorted(_EMBED_CHUNKS_PRESETS))
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


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
