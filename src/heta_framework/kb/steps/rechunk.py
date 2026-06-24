"""Rechunk merged chunks by source document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.chunking import ParsedChunk, get_text_encoding, make_chunk_id, split_text
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, store_ref


@dataclass(frozen=True)
class RechunkDocumentsConfig:
    """Configuration for RechunkDocuments."""

    rechunked_chunks_prefix: str = "rechunked_chunks"
    chunk_size: int = 1024
    overlap: int = 50
    encoding_name: str = "cl100k_base"
    split_punctuation: tuple[str, ...] = ("。", ".", ",", "，", "!", "?", "！", "？", "\n")
    object_store: str | None = None
    chunk_keys_artifact: str = "merged_chunk_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.rechunked_chunks_prefix)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap must not be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.encoding_name.strip() == "":
            raise ValueError("encoding_name must not be empty")
        if not self.split_punctuation:
            raise ValueError("split_punctuation must not be empty")
        if any(mark == "" for mark in self.split_punctuation):
            raise ValueError("split_punctuation must not contain empty values")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")


@dataclass(frozen=True)
class RechunkDocumentsResult:
    """Artifacts produced by RechunkDocuments."""

    chunk_keys: tuple[str, ...]
    input_chunk_count: int
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class _ChunkRange:
    token_start: int
    token_end: int
    parent_chunk_ids: tuple[str, ...]


class RechunkDocuments:
    """Group chunks by source document, concatenate them, and split them again."""

    name = "rechunk_documents"

    def __init__(self, config: RechunkDocumentsConfig | None = None) -> None:
        self.config = config or RechunkDocumentsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset({store_ref("objects", self.config.object_store)}),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"rechunk_documents_result", "rechunked_chunk_keys"})
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return rechunked chunk objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                "rechunked_chunk_keys",
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run rechunking and store rechunked ParsedChunk JSON objects."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        rechunked_keys: list[str] = []
        for group in _group_chunks(chunks):
            rechunks = _rechunk_group(group, config=self.config)
            for chunk in rechunks:
                key = join_object_key(
                    self.config.rechunked_chunks_prefix,
                    f"{chunk.chunk_id}.json",
                )
                await object_store.put(key, chunk.to_json_bytes())
                rechunked_keys.append(key)

        result = RechunkDocumentsResult(
            chunk_keys=tuple(rechunked_keys),
            input_chunk_count=len(chunks),
            document_count=len(_group_chunks(chunks)),
            chunk_count=len(rechunked_keys),
        )
        context.set_artifact("rechunk_documents_result", result)
        context.set_artifact("rechunked_chunk_keys", result.chunk_keys)


def _group_chunks(chunks: list[ParsedChunk]) -> list[list[ParsedChunk]]:
    groups: dict[tuple[str, str], list[ParsedChunk]] = {}
    for chunk in chunks:
        groups.setdefault((chunk.document_id, chunk.source.key), []).append(chunk)
    return [
        sorted(items, key=lambda chunk: (chunk.page_index, chunk.chunk_index, chunk.token_start))
        for _, items in sorted(groups.items())
    ]


def _rechunk_group(
    chunks: list[ParsedChunk],
    *,
    config: RechunkDocumentsConfig,
) -> list[ParsedChunk]:
    if not chunks:
        return []
    encoding = get_text_encoding(config.encoding_name)
    text_parts: list[str] = []
    ranges: list[_ChunkRange] = []
    cursor = 0
    for chunk in chunks:
        if text_parts:
            text_parts.append("\n\n")
            cursor += len(encoding.encode("\n\n"))
        token_count = len(encoding.encode(chunk.text))
        parent_ids = chunk.parent_chunk_ids or (chunk.chunk_id,)
        ranges.append(
            _ChunkRange(
                token_start=cursor,
                token_end=cursor + token_count,
                parent_chunk_ids=parent_ids,
            )
        )
        text_parts.append(chunk.text)
        cursor += token_count

    merged_text = "".join(text_parts)
    output: list[ParsedChunk] = []
    first = chunks[0]
    for chunk_index, split in enumerate(
        split_text(
            merged_text,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            encoding_name=config.encoding_name,
            split_punctuation=config.split_punctuation,
        )
    ):
        parent_ids = _collect_parent_ids(split.token_start, split.token_end, ranges)
        output.append(
            ParsedChunk(
                chunk_id=make_chunk_id(
                    document_id=first.document_id,
                    page_index=first.page_index,
                    chunk_index=chunk_index,
                    text=split.text,
                ),
                document_id=first.document_id,
                source=first.source,
                page_index=first.page_index,
                chunk_index=chunk_index,
                text=split.text,
                token_start=split.token_start,
                token_end=split.token_end,
                parent_chunk_ids=parent_ids,
            )
        )
    return output


def _collect_parent_ids(
    token_start: int,
    token_end: int,
    ranges: list[_ChunkRange],
) -> tuple[str, ...]:
    parent_ids: list[str] = []
    for item in ranges:
        if item.token_end > token_start and item.token_start < token_end:
            parent_ids.extend(item.parent_chunk_ids)
    return tuple(dict.fromkeys(parent_ids))


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component
