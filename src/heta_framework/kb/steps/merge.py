"""Merge semantically similar chunks before graph extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.common.stores.vector import (
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorSearchResult,
    VectorStoreProtocol,
)
from heta_framework.kb.chunking import ChunkEmbedding, ParsedChunk, make_chunk_id
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


DEFAULT_MERGE_PROMPT = """# Role
You are a professional text deduplication and chunk merging engine.
You are extremely strict about semantic overlap judgment.

# Task
You are given:
1. One main chunk
2. A list of candidate chunks

Determine whether each candidate chunk is semantically redundant with the main chunk.

A candidate chunk is REDUNDANT if and only if:
- More than 90% of its information content is already covered by the main chunk
- It introduces no new facts, events, numbers, time points, or conclusions
- Paraphrasing, reordering, or wording differences still count as overlap

If a candidate chunk is redundant, merge it into the main chunk, remove duplicate
expressions, preserve the most complete wording, and do not invent content.
If a candidate chunk is not redundant, do not merge it.

Return a strict JSON object.

If no merge occurred:
{{"text": null, "merge_id": null}}

If one or more candidate chunks were merged:
{{"text": "<merged main chunk text>", "merge_id": <id> | [<id>, ...]}}

merge_id must refer to the candidate chunk ids in the candidate list, not the main chunk id.

Main chunk:
{MAIN_CHUNK}

Candidate chunks:
{CANDIDATE_LIST}
"""


DEFAULT_REFINE_PROMPT = """# Role
You are a professional text deduplication, chunk merging, and content refinement engine.
You are extremely strict about semantic overlap judgment and conservative about merging.

# Task
You are given:
1. One main chunk
2. A list of candidate chunks

Phase 1: Determine whether each candidate chunk is semantically redundant with the main chunk.
A candidate is REDUNDANT if and only if more than 90% of its information content is
already covered and it introduces no new facts, events, dates, numbers, entities,
locations, or conclusions.

Phase 2: Refine the resulting main chunk. Extract pure narrative content and remove
web boilerplate, navigation, UI text, legal boilerplate, technical markers,
advertisements, tracking text, and other non-content noise. Preserve valid narrative
content, facts, names, events, time points, locations, data, and conclusions.

Return a strict JSON object.

If no candidate chunks were merged, return the refined main chunk:
{{"text": "<refined main chunk text>", "merge_id": null}}

If one or more candidate chunks were merged:
{{"text": "<refined merged text>", "merge_id": <id> | [<id>, ...]}}

merge_id must refer to the candidate chunk ids in the candidate list, not the main chunk id.

Main chunk:
{MAIN_CHUNK}

Candidate chunks:
{CANDIDATE_LIST}
"""


@dataclass(frozen=True)
class MergeChunksConfig:
    """Configuration for MergeChunks."""

    merged_chunks_prefix: str = "merged_chunks"
    merge_collection: str = "merge_chunks"
    metric: str = "cosine"
    top_k: int = 8
    num_topk_candidates: int = 5
    max_rounds: int = 10
    min_similarity: float = 0.85
    merge_threshold: float = 0.05
    recreate_collection: bool = True
    refine_prompt: str | None = None
    merge_prompt: str | None = None
    object_store: str | None = None
    vector_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    chunk_embedding_keys_artifact: str = "chunk_embedding_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.merged_chunks_prefix)
        if self.merge_collection.strip() == "":
            raise ValueError("merge_collection must not be empty")
        if self.metric not in {"cosine", "dot", "l2"}:
            raise ValueError("metric must be one of: cosine, dot, l2")
        if self.top_k <= 1:
            raise ValueError("top_k must be greater than one")
        if self.num_topk_candidates <= 0:
            raise ValueError("num_topk_candidates must be greater than zero")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be greater than zero")
        if self.merge_threshold < 0:
            raise ValueError("merge_threshold must not be negative")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.chunk_embedding_keys_artifact.strip() == "":
            raise ValueError("chunk_embedding_keys_artifact must not be empty")


@dataclass(frozen=True)
class MergeChunksResult:
    """Artifacts produced by MergeChunks."""

    chunk_keys: tuple[str, ...]
    collection: str
    input_chunk_count: int
    active_chunk_count: int
    merged_count: int
    round_count: int
    stopped_reason: str


class MergeChunks:
    """Merge similar chunks with vector search candidates and LLM decisions."""

    name = "merge_chunks"

    def __init__(self, config: MergeChunksConfig | None = None) -> None:
        self.config = config or MergeChunksConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("vector", self.config.vector_store),
                    model_ref("language", self.config.language_model),
                    model_ref("embedding", self.config.embedding_model),
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
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"merge_chunks_result", "merged_chunk_keys"})
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run chunk merge and expose the active chunk keys after merging."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        vector_store = _require_vector_store(
            context.get_component(store_ref("vector", self.config.vector_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )

        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        embedding_keys = tuple(context.get_artifact(self.config.chunk_embedding_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        embeddings = [
            ChunkEmbedding.from_json(await object_store.get(key)) for key in embedding_keys
        ]
        embedding_by_chunk_id = {embedding.chunk_id: embedding for embedding in embeddings}
        key_by_chunk_id = {chunk.chunk_id: key for chunk, key in zip(chunks, chunk_keys, strict=True)}
        active_chunks = {chunk.chunk_id: chunk for chunk in chunks}

        dimension = _validate_embeddings(chunks, embedding_by_chunk_id)
        if self.config.recreate_collection and await vector_store.has_collection(
            self.config.merge_collection
        ):
            await vector_store.drop_collection(self.config.merge_collection)
        await vector_store.create_collection(
            VectorCollectionConfig(
                name=self.config.merge_collection,
                dimension=dimension,
                metric=self.config.metric,  # type: ignore[arg-type]
            )
        )
        await vector_store.upsert(
            self.config.merge_collection,
            [_to_vector_record(chunk, embedding_by_chunk_id[chunk.chunk_id]) for chunk in chunks],
        )

        merged_count = 0
        round_count = 0
        stopped_reason = "max_rounds"
        for round_index in range(self.config.max_rounds):
            round_count = round_index + 1
            consumed_ids: set[str] = set()
            removed_this_round = 0
            phase = "refine" if round_index == 0 else "merge"
            for chunk_id in list(active_chunks):
                if chunk_id in consumed_ids or chunk_id not in active_chunks:
                    continue
                chunk = active_chunks[chunk_id]
                embedding = embedding_by_chunk_id[chunk_id]
                candidates = await _find_merge_candidates(
                    vector_store,
                    collection=self.config.merge_collection,
                    chunk=chunk,
                    vector=embedding.vector,
                    top_k=self.config.top_k,
                    min_similarity=self.config.min_similarity,
                    consumed_ids=consumed_ids,
                )
                if not candidates:
                    continue
                merge_decision = await _ask_llm_to_merge(
                    language_model,
                    chunk,
                    [
                        active_chunks[candidate.id]
                        for candidate in candidates[: self.config.num_topk_candidates]
                        if candidate.id in active_chunks
                    ],
                    phase=phase,
                    refine_prompt=self.config.refine_prompt or DEFAULT_REFINE_PROMPT,
                    merge_prompt=self.config.merge_prompt or DEFAULT_MERGE_PROMPT,
                )
                if merge_decision is None or merge_decision.text.strip() == "":
                    continue
                merged_ids = tuple(
                    candidate.chunk_id
                    for candidate in merge_decision.candidates
                    if candidate.chunk_id in active_chunks and candidate.chunk_id not in consumed_ids
                )
                if phase == "merge" and not merged_ids:
                    continue
                chunks_to_merge = (chunk,) + tuple(active_chunks[candidate_id] for candidate_id in merged_ids)

                merged_chunk = _make_merged_chunk(chunks_to_merge, merge_decision.text)
                merged_embedding = await _embed_merged_chunk(embedding_model, merged_chunk)
                merged_key = join_object_key(
                    self.config.merged_chunks_prefix,
                    f"{merged_chunk.chunk_id}.json",
                )
                await object_store.put(merged_key, merged_chunk.to_json_bytes())
                key_by_chunk_id[merged_chunk.chunk_id] = merged_key
                for merged_id in (chunk.chunk_id,) + merged_ids:
                    active_chunks.pop(merged_id, None)
                active_chunks[merged_chunk.chunk_id] = merged_chunk
                embedding_by_chunk_id[merged_chunk.chunk_id] = merged_embedding
                consumed_ids.update({chunk.chunk_id, *merged_ids, merged_chunk.chunk_id})
                await vector_store.delete(
                    self.config.merge_collection,
                    [chunk.chunk_id, *merged_ids],
                )
                await vector_store.upsert(
                    self.config.merge_collection,
                    [_to_vector_record(merged_chunk, merged_embedding)],
                )
                merged_count += 1
                removed_this_round += 1 + len(merged_ids)
            if removed_this_round == 0:
                stopped_reason = "no_merges"
                break
            merge_ratio = removed_this_round / len(chunks) if chunks else 0
            if merge_ratio < self.config.merge_threshold:
                stopped_reason = "merge_threshold"
                break

        active_keys = tuple(key_by_chunk_id[chunk_id] for chunk_id in active_chunks)
        result = MergeChunksResult(
            chunk_keys=active_keys,
            collection=self.config.merge_collection,
            input_chunk_count=len(chunks),
            active_chunk_count=len(active_keys),
            merged_count=merged_count,
            round_count=round_count,
            stopped_reason=stopped_reason,
        )
        context.set_artifact("merge_chunks_result", result)
        context.set_artifact("merged_chunk_keys", result.chunk_keys)


def _validate_embeddings(
    chunks: list[ParsedChunk],
    embedding_by_chunk_id: dict[str, ChunkEmbedding],
) -> int:
    dimension = 0
    for chunk in chunks:
        embedding = embedding_by_chunk_id.get(chunk.chunk_id)
        if embedding is None:
            raise ValueError(f"missing embedding for chunk: {chunk.chunk_id}")
        if embedding.document_id != chunk.document_id:
            raise ValueError(f"embedding document_id mismatch for chunk: {chunk.chunk_id}")
        dimension = embedding.dimension
    return dimension


def _to_vector_record(chunk: ParsedChunk, embedding: ChunkEmbedding) -> VectorRecord:
    return VectorRecord(
        id=chunk.chunk_id,
        vector=embedding.vector,
        text=chunk.text,
        metadata={
            "document_id": chunk.document_id,
            "source_key": chunk.source.key,
            "chunk_index": chunk.chunk_index,
            "parent_chunk_ids": list(chunk.parent_chunk_ids),
        },
    )


async def _find_merge_candidates(
    vector_store: VectorStoreProtocol,
    *,
    collection: str,
    chunk: ParsedChunk,
    vector: list[float],
    top_k: int,
    min_similarity: float,
    consumed_ids: set[str],
) -> list[VectorSearchResult]:
    results = await vector_store.search(
        collection,
        VectorQuery(
            vector=vector,
            top_k=top_k,
            filter={"document_id": chunk.document_id},
        ),
    )
    return [
        result
        for result in results
        if result.id != chunk.chunk_id
        and result.id not in consumed_ids
        and result.score >= min_similarity
    ]


async def _ask_llm_to_merge(
    language_model: LanguageModelProtocol,
    main_chunk: ParsedChunk,
    candidates: list[ParsedChunk],
    *,
    phase: str,
    refine_prompt: str,
    merge_prompt: str,
) -> "_MergeDecision | None":
    if not candidates:
        return None
    prompt_template = refine_prompt if phase == "refine" else merge_prompt
    id_to_candidate = {index + 1: candidate for index, candidate in enumerate(candidates)}
    main_payload = {"id": 0, "text": main_chunk.text}
    candidate_payload = [
        {"id": index, "text": candidate.text}
        for index, candidate in id_to_candidate.items()
    ]
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt_template.format(
                MAIN_CHUNK=json.dumps(main_payload, ensure_ascii=False),
                CANDIDATE_LIST=json.dumps(candidate_payload, ensure_ascii=False),
            ),
            options=ModelOptions(response_format={"type": "json_object"}, temperature=0),
            trace_context={"step": "merge_chunks"},
        )
    )
    payload = _load_json_payload(result.parsed if result.parsed is not None else result.text)
    text = str(payload.get("text", "")).strip()
    merge_ids = _normalize_merge_ids(payload.get("merge_id"))
    merged_candidates = tuple(
        id_to_candidate[merge_id] for merge_id in merge_ids if merge_id in id_to_candidate
    )
    if not text:
        return None
    return _MergeDecision(text=text, candidates=merged_candidates)


def _load_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _normalize_merge_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, list):
        ids: list[int] = []
        for item in value:
            if isinstance(item, int):
                ids.append(item)
        return tuple(ids)
    return ()


@dataclass(frozen=True)
class _MergeDecision:
    text: str
    candidates: tuple[ParsedChunk, ...]


def _make_merged_chunk(chunks: tuple[ParsedChunk, ...], text: str) -> ParsedChunk:
    first = chunks[0]
    parent_ids: tuple[str, ...] = ()
    for chunk in chunks:
        parent_ids += _root_parent_ids(chunk)
    chunk_index = min(chunk.chunk_index for chunk in chunks)
    page_index = min(chunk.page_index for chunk in chunks)
    return ParsedChunk(
        chunk_id=make_chunk_id(
            document_id=first.document_id,
            page_index=page_index,
            chunk_index=chunk_index,
            text=text,
        ),
        document_id=first.document_id,
        source=first.source,
        page_index=page_index,
        chunk_index=chunk_index,
        text=text,
        token_start=min(chunk.token_start for chunk in chunks),
        token_end=max(chunk.token_end for chunk in chunks),
        parent_chunk_ids=tuple(dict.fromkeys(parent_ids)),
    )


def _root_parent_ids(chunk: ParsedChunk) -> tuple[str, ...]:
    return chunk.parent_chunk_ids or (chunk.chunk_id,)


async def _embed_merged_chunk(
    embedding_model: EmbeddingModelProtocol,
    chunk: ParsedChunk,
) -> ChunkEmbedding:
    result = await embedding_model.embed(
        EmbeddingRequest(texts=[chunk.text], trace_context={"step": "merge_chunks"})
    )
    if len(result.vectors) != 1 or not result.vectors[0]:
        raise ValueError(f"embedding vector must not be empty for chunk: {chunk.chunk_id}")
    vector = [float(value) for value in result.vectors[0]]
    return ChunkEmbedding(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        model_name=result.model_name or embedding_model.model_name,
        vector=vector,
        dimension=len(vector),
    )


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
