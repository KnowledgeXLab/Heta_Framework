"""Split ParsedDocument artifacts into ParsedChunk artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.chunking import ParsedChunk, make_chunk_id, split_text
from heta_framework.kb.parsing import ParsedDocument
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, store_ref


@dataclass(frozen=True)
class SplitDocumentsConfig:
    """Configuration for SplitDocuments."""

    chunks_prefix: str = "chunks"
    chunk_size: int = 1024
    overlap: int = 50
    encoding_name: str = "cl100k_base"
    split_punctuation: tuple[str, ...] = ("。", ".", ",", "，", "!", "?", "！", "？", "\n")
    object_store: str | None = None
    parsed_document_keys_artifact: str = "parsed_document_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.chunks_prefix)
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
        if self.parsed_document_keys_artifact.strip() == "":
            raise ValueError("parsed_document_keys_artifact must not be empty")


@dataclass(frozen=True)
class SplitDocumentsResult:
    """Artifacts produced by SplitDocuments."""

    chunk_keys: tuple[str, ...]
    document_count: int
    chunk_count: int


class SplitDocuments:
    """Split parsed documents into retrieval-ready chunks."""

    name = "split_documents"

    def __init__(self, config: SplitDocumentsConfig | None = None) -> None:
        self.config = config or SplitDocumentsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset({store_ref("objects", self.config.object_store)}),
            artifacts=frozenset({self.config.parsed_document_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(artifacts=frozenset({"split_documents_result", "chunk_keys"}))

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return chunk objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                "chunk_keys",
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the split step and store chunks as JSON bytes."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        parsed_document_keys = tuple(
            context.get_artifact(self.config.parsed_document_keys_artifact)
        )

        chunk_keys: list[str] = []
        for document_key in parsed_document_keys:
            document = ParsedDocument.from_json(await object_store.get(document_key))
            for chunk in _split_document(document, config=self.config):
                chunk_key = join_object_key(self.config.chunks_prefix, f"{chunk.chunk_id}.json")
                await object_store.put(chunk_key, chunk.to_json_bytes())
                chunk_keys.append(chunk_key)

        result = SplitDocumentsResult(
            chunk_keys=tuple(chunk_keys),
            document_count=len(parsed_document_keys),
            chunk_count=len(chunk_keys),
        )
        context.set_artifact("split_documents_result", result)
        context.set_artifact("chunk_keys", result.chunk_keys)


def _split_document(document: ParsedDocument, *, config: SplitDocumentsConfig) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    chunk_index = 0
    for page in sorted(document.pages, key=lambda item: item.page_index):
        splits = split_text(
            page.text,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            encoding_name=config.encoding_name,
            split_punctuation=config.split_punctuation,
        )
        for split in splits:
            chunks.append(
                ParsedChunk(
                    chunk_id=make_chunk_id(
                        document_id=document.document_id,
                        page_index=page.page_index,
                        chunk_index=chunk_index,
                        text=split.text,
                    ),
                    document_id=document.document_id,
                    source=document.source,
                    page_index=page.page_index,
                    chunk_index=chunk_index,
                    text=split.text,
                    token_start=split.token_start,
                    token_end=split.token_end,
                )
            )
            chunk_index += 1
    return chunks


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component
