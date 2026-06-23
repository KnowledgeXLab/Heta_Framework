import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserRegistry,
    ParsedDocument,
    ParsedPage,
    TextParser,
    make_document_id,
    make_parsed_source,
)
from heta_framework.kb.steps import (  # noqa: E402
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
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


def test_split_documents_declares_requirements_and_capabilities():
    step = SplitDocuments()

    assert step.name == "split_documents"
    assert {ref.key for ref in step.requirements.components} == {"stores.objects"}
    assert step.requirements.artifacts == frozenset({"parsed_document_keys"})
    assert step.capabilities.artifacts == frozenset({"split_documents_result", "chunk_keys"})


def test_split_documents_splits_parsed_documents(tmp_path):
    store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects": store})
    data = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.".encode("utf-8")
    source = make_parsed_source(key="raw/doc.txt", name="doc.txt", file_type="txt", data=data)
    document = ParsedDocument(
        document_id=make_document_id(source.content_sha256),
        source=source,
        pages=[ParsedPage(page_index=0, text=data.decode("utf-8"))],
    )

    async def run():
        await store.put("parsed/doc.json", document.to_json_bytes())
        context.set_artifact("parsed_document_keys", ("parsed/doc.json",))
        await SplitDocuments(
            SplitDocumentsConfig(chunk_size=24, overlap=6, encoding_name="unicode")
        ).run(context)

    asyncio.run(run())

    result = context.artifacts["split_documents_result"]
    chunk_keys = context.artifacts["chunk_keys"]
    assert result.document_count == 1
    assert result.chunk_count == len(chunk_keys)
    assert len(chunk_keys) >= 2
    assert all(key.startswith("chunks/chunk_") and key.endswith(".json") for key in chunk_keys)

    async def read_chunks():
        return [ParsedChunk.from_json(await store.get(key)) for key in chunk_keys]

    chunks = asyncio.run(read_chunks())

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.source == source for chunk in chunks)
    assert all(chunk.page_index == 0 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert chunks[1].token_start < chunks[0].token_end


def test_split_documents_supports_named_object_store_and_artifact(tmp_path):
    store = LocalObjectStore(tmp_path)
    context = FakeContext({"stores.objects.local": store})
    data = b"hello chunk"
    source = make_parsed_source(key="raw/doc.txt", name="doc.txt", file_type="txt", data=data)
    document = ParsedDocument(
        document_id=make_document_id(source.content_sha256),
        source=source,
        pages=[ParsedPage(page_index=0, text="hello chunk")],
    )

    async def run():
        await store.put("parsed/doc.json", document.to_json_bytes())
        context.set_artifact("custom_parsed_keys", ("parsed/doc.json",))
        await SplitDocuments(
            SplitDocumentsConfig(
                chunks_prefix="outputs/chunks",
                object_store="local",
                parsed_document_keys_artifact="custom_parsed_keys",
                encoding_name="unicode",
            )
        ).run(context)

    asyncio.run(run())

    keys = context.artifacts["chunk_keys"]
    assert len(keys) == 1
    assert keys[0].startswith("outputs/chunks/chunk_")


def test_parse_then_split_documents_step(tmp_path):
    store = LocalObjectStore(tmp_path)
    registry = DocumentParserRegistry([TextParser()])
    context = FakeContext(
        {
            "stores.objects": store,
            "parsers.documents": registry,
        }
    )

    async def run():
        await store.put("raw/doc.md", b"# Title\n\nFirst sentence. Second sentence.")
        await ParseDocuments().run(context)
        await SplitDocuments(
            SplitDocumentsConfig(chunk_size=24, overlap=6, encoding_name="unicode")
        ).run(context)

    asyncio.run(run())

    assert len(context.artifacts["parsed_document_keys"]) == 1
    assert len(context.artifacts["chunk_keys"]) >= 1


def test_split_documents_config_validates_overlap():
    with pytest.raises(ValueError, match="overlap"):
        SplitDocumentsConfig(chunk_size=10, overlap=10)


def test_parsed_chunk_round_trips_json():
    data = b"hello"
    source = make_parsed_source(key="raw/doc.txt", name="doc.txt", file_type="txt", data=data)
    chunk = ParsedChunk(
        chunk_id="chunk_123",
        document_id="doc_123",
        source=source,
        page_index=0,
        chunk_index=0,
        text="hello",
        token_start=0,
        token_end=1,
    )

    assert ParsedChunk.from_json(chunk.to_json_bytes()) == chunk
