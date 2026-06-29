"""Index chunk text into a full-text search store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.text_index import (
    TextIndexConfig,
    TextIndexRecord,
    TextIndexStoreProtocol,
)
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, store_ref


@dataclass(frozen=True)
class FullTextIndexNames:
    """Full-text index names used by chunk text indexing."""

    chunk_text: str = "chunk_full_text"

    def __post_init__(self) -> None:
        if self.chunk_text.strip() == "":
            raise ValueError("index_names.chunk_text must not be empty")


@dataclass(frozen=True)
class IndexFullTextConfig:
    """Configuration for IndexFullText."""

    index_names: FullTextIndexNames = field(default_factory=FullTextIndexNames)
    batch_size: int = 128
    object_store: str | None = None
    text_index_store: str | None = None
    chunk_keys_artifact: str = "chunk_keys"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")


@dataclass(frozen=True)
class IndexFullTextResult:
    """Artifacts produced by IndexFullText."""

    index_name: str
    indexed_count: int


class IndexFullText:
    """Write chunk text into a full-text index and enable full-text search."""

    name = "index_full_text"

    def __init__(self, config: IndexFullTextConfig | None = None) -> None:
        self.config = config or IndexFullTextConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("text_index", self.config.text_index_store),
                }
            ),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts and query modes produced by this step."""
        text_index_store_ref = store_ref("text_index", self.config.text_index_store)
        return StepCapabilities(
            artifacts=frozenset({"index_full_text_result"}),
            queries=frozenset({"full_text_search"}),
            search_assets=(
                SearchAsset(
                    kind="chunk_full_text_index",
                    name=self.config.index_names.chunk_text,
                    store=text_index_store_ref.key,
                    metadata={
                        "index": self.config.index_names.chunk_text,
                        "id_field": "chunk_id",
                        "text_field": "content_text",
                        "metadata_field": "metadata",
                        "ranking": "bm25",
                    },
                ),
            ),
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return full-text indexes produced by this step."""
        return StepCleanupPlan(
            (
                CleanupTarget(
                    kind="text_index",
                    value=self.config.index_names.chunk_text,
                    component=store_ref("text_index", self.config.text_index_store).key,
                ),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the full-text indexing step and upsert chunk text records."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        text_index_store = _require_text_index_store(
            context.get_component(store_ref("text_index", self.config.text_index_store).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        await text_index_store.create_index(TextIndexConfig(name=self.config.index_names.chunk_text))
        records = [_record_from_chunk(chunk) for chunk in chunks]
        for start in range(0, len(records), self.config.batch_size):
            await text_index_store.upsert(
                self.config.index_names.chunk_text,
                records[start : start + self.config.batch_size],
            )

        context.set_artifact(
            "index_full_text_result",
            IndexFullTextResult(
                index_name=self.config.index_names.chunk_text,
                indexed_count=len(records),
            ),
        )


def _record_from_chunk(chunk: ParsedChunk) -> TextIndexRecord:
    return TextIndexRecord(
        id=chunk.chunk_id,
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
        },
    )


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_text_index_store(component: object) -> TextIndexStoreProtocol:
    if not isinstance(component, TextIndexStoreProtocol):
        raise TypeError("stores.text_index must satisfy TextIndexStoreProtocol")
    return component
