import asyncio
import sys
from pathlib import Path

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
