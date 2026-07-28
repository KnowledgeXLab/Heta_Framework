"""Static Wiki knowledge base construction procedure."""

from __future__ import annotations

from dataclasses import dataclass, field

from heta_framework.kb.steps import (
    BuildWikiPages,
    BuildWikiPagesConfig,
    EmbedChunks,
    EmbedChunksConfig,
    IndexFullText,
    IndexFullTextConfig,
    IndexVectors,
    IndexVectorsConfig,
    KnowledgeStepProtocol,
    SplitWikiPages,
    SplitWikiPagesConfig,
)


def _wiki_embed_config() -> EmbedChunksConfig:
    return EmbedChunksConfig(preset="wiki")


def _wiki_vector_config() -> IndexVectorsConfig:
    return IndexVectorsConfig(preset="wiki")


def _wiki_full_text_config() -> IndexFullTextConfig:
    return IndexFullTextConfig(preset="wiki")


@dataclass(frozen=True)
class HetaWikiProcedure:
    """Compose Wiki page construction and retrieval indexing steps."""

    build_pages_config: BuildWikiPagesConfig = field(
        default_factory=BuildWikiPagesConfig
    )
    split_pages_config: SplitWikiPagesConfig = field(
        default_factory=SplitWikiPagesConfig
    )
    embed_chunks_config: EmbedChunksConfig = field(default_factory=_wiki_embed_config)
    index_vectors_config: IndexVectorsConfig = field(default_factory=_wiki_vector_config)
    index_full_text_config: IndexFullTextConfig = field(
        default_factory=_wiki_full_text_config
    )

    def __post_init__(self) -> None:
        _require_wiki_preset("embed_chunks_config", self.embed_chunks_config.preset)
        _require_wiki_preset("index_vectors_config", self.index_vectors_config.preset)
        _require_wiki_preset(
            "index_full_text_config",
            self.index_full_text_config.preset,
        )
        if self.split_pages_config.wiki_page_keys_artifact != "wiki_page_keys":
            raise ValueError(
                "split_pages_config.wiki_page_keys_artifact must be 'wiki_page_keys'"
            )
        _require_shared_object_store(
            self.build_pages_config.object_store,
            self.split_pages_config.object_store,
            self.embed_chunks_config.object_store,
            self.index_vectors_config.object_store,
            self.index_full_text_config.object_store,
        )

    @property
    def name(self) -> str:
        """Return the stable procedure name."""
        return "heta_wiki"

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Expand the procedure into executable Wiki construction steps."""
        return (
            BuildWikiPages(self.build_pages_config),
            SplitWikiPages(self.split_pages_config),
            EmbedChunks(self.embed_chunks_config),
            IndexVectors(self.index_vectors_config),
            IndexFullText(self.index_full_text_config),
        )


def _require_wiki_preset(field_name: str, preset: str) -> None:
    if preset != "wiki":
        raise ValueError(f"{field_name}.preset must be 'wiki'")


def _require_shared_object_store(*object_stores: str | None) -> None:
    if len(set(object_stores)) != 1:
        raise ValueError("all HetaWikiProcedure steps must use the same object store")
