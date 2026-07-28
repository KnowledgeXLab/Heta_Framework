import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    InMemoryTextIndexStore,
    LocalObjectStore,
    TextQuery,
)
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    FullTextIndexNames,
    IndexFullText,
    IndexFullTextConfig,
)


class FakeContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


def test_index_full_text_indexes_chunks_and_declares_search_asset(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        text_index = InMemoryTextIndexStore()
        chunk = ParsedChunk(
            chunk_id="chunk_heta",
            document_id="doc_heta",
            source=ParsedSource(
                key="raw/heta.txt",
                name="heta.txt",
                file_type="txt",
                content_sha256="a" * 64,
            ),
            page_index=0,
            chunk_index=0,
            text="Heta builds full text indexes for BM25 search.",
            token_start=0,
            token_end=9,
        )
        await object_store.put("chunks/chunk_heta.json", chunk.to_json_bytes())
        context = FakeContext(
            {
                "stores.objects": object_store,
                "stores.text_index": text_index,
            }
        )
        context.artifacts["chunk_keys"] = ("chunks/chunk_heta.json",)

        step = IndexFullText(
            IndexFullTextConfig(
                index_names=FullTextIndexNames(chunk_text="test_full_text"),
            )
        )
        await step.run(context)

        result = context.artifacts["index_full_text_result"]
        hits = await text_index.search("test_full_text", TextQuery(text="BM25 search", top_k=5))
        return step, result, hits, text_index

    step, result, hits, text_index = asyncio.run(run())

    assert result.index_name == "test_full_text"
    assert result.indexed_count == 1
    assert hits[0].id == "chunk_heta"
    assert hits[0].metadata["source_key"] == "raw/heta.txt"
    assert step.capabilities.queries == frozenset({"full_text_search"})
    assert step.capabilities.search_assets[0].kind == "chunk_full_text_index"
    cleanup_target = step.cleanup_plan({}).targets[0]
    assert cleanup_target.kind == "text_index"
    assert cleanup_target.value == "test_full_text"
    asyncio.run(text_index.aclose())


def test_index_full_text_wiki_preset_indexes_separate_artifact(tmp_path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        text_index = InMemoryTextIndexStore()
        chunk = ParsedChunk(
            chunk_id="wiki_chunk_heta",
            document_id="wiki_page_heta",
            source=ParsedSource(
                key="wiki/pages/1-heta.md",
                name="1-heta.md",
                file_type="wiki",
                content_sha256="b" * 64,
            ),
            page_index=0,
            chunk_index=0,
            text="Heta wiki pages support full text search.",
            token_start=0,
            token_end=8,
            heading_path=("Search", "Full Text"),
            origin_source=ParsedSource(
                key="raw/heta.md",
                name="heta.md",
                file_type="md",
                content_sha256="a" * 64,
            ),
        )
        await object_store.put("wiki_chunks/wiki_chunk_heta.json", chunk.to_json_bytes())
        context = FakeContext(
            {
                "stores.objects": object_store,
                "stores.text_index": text_index,
            }
        )
        context.artifacts["wiki_chunk_keys"] = ("wiki_chunks/wiki_chunk_heta.json",)

        step = IndexFullText(IndexFullTextConfig(preset="wiki"))
        await step.run(context)

        result = context.artifacts["index_wiki_full_text_result"]
        hits = await text_index.search(
            "wiki_chunk_full_text",
            TextQuery(text="wiki search", top_k=5),
        )
        return step, result, hits, context, text_index

    step, result, hits, context, text_index = asyncio.run(run())

    assert "index_full_text_result" not in context.artifacts
    assert result.index_name == "wiki_chunk_full_text"
    assert result.indexed_count == 1
    assert hits[0].metadata["source_file_type"] == "wiki"
    assert hits[0].metadata["origin_source_key"] == "raw/heta.md"
    assert hits[0].metadata["heading_path"] == "Search > Full Text"
    assert step.requirements.artifacts == frozenset({"wiki_chunk_keys"})
    assert step.capabilities.artifacts == frozenset({"index_wiki_full_text_result"})
    assert step.capabilities.queries == frozenset({"wiki_full_text_search"})
    assert step.capabilities.search_assets[0].kind == "wiki_chunk_full_text_index"
    asyncio.run(text_index.aclose())


def test_index_full_text_preset_rejects_mixed_artifacts():
    with pytest.raises(ValueError, match="chunk_keys_artifact"):
        IndexFullTextConfig(preset="wiki", chunk_keys_artifact="chunk_keys")


def test_index_full_text_config_preserves_v0_1_0_constructor_contract():
    config = IndexFullTextConfig(
        FullTextIndexNames(chunk_text="legacy_full_text"),
        64,
        "objects",
        "text",
        "legacy_chunk_keys",
    )

    assert config.index_names == FullTextIndexNames(chunk_text="legacy_full_text")
    assert config.batch_size == 64
    assert config.object_store == "objects"
    assert config.text_index_store == "text"
    assert config.chunk_keys_artifact == "legacy_chunk_keys"
    assert config.result_artifact == "index_full_text_result"
    assert config.query_mode == "full_text_search"
    assert config.asset_kind == "chunk_full_text_index"
