"""Index chunk text into a full-text search store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

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


IndexFullTextPreset = Literal["default", "wiki"]


@dataclass(frozen=True)
class FullTextIndexNames:
    """Full-text index names used by chunk text indexing."""

    chunk_text: str = "chunk_full_text"

    def __post_init__(self) -> None:
        if self.chunk_text.strip() == "":
            raise ValueError("index_names.chunk_text must not be empty")


@dataclass(frozen=True)
class _IndexFullTextPreset:
    index_name: str
    chunk_keys_artifact: str
    result_artifact: str
    query_mode: str
    asset_kind: str


_INDEX_FULL_TEXT_PRESETS: dict[IndexFullTextPreset, _IndexFullTextPreset] = {
    "default": _IndexFullTextPreset(
        index_name="chunk_full_text",
        chunk_keys_artifact="chunk_keys",
        result_artifact="index_full_text_result",
        query_mode="full_text_search",
        asset_kind="chunk_full_text_index",
    ),
    "wiki": _IndexFullTextPreset(
        index_name="wiki_chunk_full_text",
        chunk_keys_artifact="wiki_chunk_keys",
        result_artifact="index_wiki_full_text_result",
        query_mode="wiki_full_text_search",
        asset_kind="wiki_chunk_full_text_index",
    ),
}


@dataclass(frozen=True)
class IndexFullTextConfig:
    """Configuration for IndexFullText."""

    index_names: FullTextIndexNames | None = None
    batch_size: int = 128
    object_store: str | None = None
    text_index_store: str | None = None
    chunk_keys_artifact: str | None = None
    preset: IndexFullTextPreset = "default"
    result_artifact: str | None = None
    query_mode: str | None = None
    asset_kind: str | None = None

    def __post_init__(self) -> None:
        preset = _index_full_text_preset(self.preset)
        if self.index_names is None:
            object.__setattr__(
                self,
                "index_names",
                FullTextIndexNames(chunk_text=preset.index_name),
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
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


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
            artifacts=frozenset({self.config.result_artifact}),
            queries=frozenset({self.config.query_mode}),
            search_assets=(
                SearchAsset(
                    kind=self.config.asset_kind,
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

        await text_index_store.create_index(
            TextIndexConfig(name=self.config.index_names.chunk_text)
        )
        records = [_record_from_chunk(chunk) for chunk in chunks]
        for start in range(0, len(records), self.config.batch_size):
            await text_index_store.upsert(
                self.config.index_names.chunk_text,
                records[start : start + self.config.batch_size],
            )

        context.set_artifact(
            self.config.result_artifact,
            IndexFullTextResult(
                index_name=self.config.index_names.chunk_text,
                indexed_count=len(records),
            ),
        )


def _index_full_text_preset(preset: str) -> _IndexFullTextPreset:
    try:
        return _INDEX_FULL_TEXT_PRESETS[preset]  # type: ignore[index]
    except KeyError as exc:
        allowed = ", ".join(sorted(_INDEX_FULL_TEXT_PRESETS))
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
            "heading_path": " > ".join(chunk.heading_path),
            **_origin_source_metadata(chunk),
        },
    )


def _origin_source_metadata(chunk: ParsedChunk) -> dict[str, str]:
    if chunk.origin_source is None:
        return {}
    return {
        "origin_source_key": chunk.origin_source.key,
        "origin_source_name": chunk.origin_source.name,
        "origin_source_file_type": chunk.origin_source.file_type,
        "origin_source_content_sha256": chunk.origin_source.content_sha256,
    }


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_text_index_store(component: object) -> TextIndexStoreProtocol:
    if not isinstance(component, TextIndexStoreProtocol):
        raise TypeError("stores.text_index must satisfy TextIndexStoreProtocol")
    return component
