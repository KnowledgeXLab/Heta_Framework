import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.extractors import ExtractedBlock, ExtractedDocument  # noqa: E402
from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserRegistry,
    HtmlParser,
    ImageParser,
    OfficeParser,
    ParsedDocument,
    PdfParser,
    SheetParser,
    TextParser,
)
from heta_framework.kb.steps import ParseDocuments, ParseDocumentsConfig


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
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


class FakeVisionModel:
    @property
    def model_name(self):
        return "fake-vision"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        image = request.content[1]
        return ModelResult(text=f"described image with {image.mime_type}")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(text_delta=(await self.invoke(request)).text, model_name=self.model_name)


class FakeExtractor:
    async def extract(self, document, options=None):
        del options
        return ExtractedDocument(
            blocks=(
                ExtractedBlock(kind="text", text=f"extracted {document.filename}", page_index=0),
                ExtractedBlock(kind="table", text="A | B\n1 | 2", page_index=0),
            )
        )


def test_parse_documents_declares_requirements_and_capabilities():
    step = ParseDocuments()

    assert step.name == "parse_documents"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.objects",
        "parsers.documents",
    }
    assert step.capabilities.artifacts == frozenset({"parse_documents_result", "parsed_document_keys"})


def test_parse_documents_writes_parsed_documents(tmp_path):
    store = LocalObjectStore(tmp_path)
    registry = DocumentParserRegistry([TextParser()])
    context = FakeContext(
        {
            "stores.objects": store,
            "parsers.documents": registry,
        }
    )

    async def run():
        await store.put("raw/readme.md", b"# Heta\n\nParser step")
        await ParseDocuments().run(context)

    asyncio.run(run())

    result = context.artifacts["parse_documents_result"]
    keys = context.artifacts["parsed_document_keys"]
    assert result.document_keys == keys
    assert result.skipped_keys == ()
    assert len(keys) == 1
    assert keys[0].startswith("parsed/doc_")
    assert keys[0].endswith(".json")

    async def read_document():
        return ParsedDocument.from_json(await store.get(keys[0]))

    document = asyncio.run(read_document())

    assert document.source.key == "raw/readme.md"
    assert document.source.name == "readme.md"
    assert document.source.file_type == "md"
    assert document.pages[0].text == "# Heta\n\nParser step"


def test_parse_documents_routes_multiple_file_types(tmp_path):
    store = LocalObjectStore(tmp_path)
    extractor = FakeExtractor()
    registry = DocumentParserRegistry(
        [
            TextParser(),
            HtmlParser(),
            SheetParser(),
            ImageParser(FakeVisionModel()),
            PdfParser(extractor),
            OfficeParser(extractor),
        ]
    )
    context = FakeContext(
        {
            "stores.objects": store,
            "parsers.documents": registry,
        }
    )

    async def run():
        await store.put("raw/readme.md", b"# Heta\n\nParser smoke")
        await store.put(
            "raw/page.html",
            (
                b"<html><head><title>Demo</title></head><body><main>"
                b"<p>Hello HTML</p><table><tr><th>A</th><td>B</td></tr></table>"
                b"</main></body></html>"
            ),
        )
        await store.put("raw/table.csv", b"Model,Score\ngpt,98\nqwen,96\n")
        await store.put("raw/red.png", PNG_BYTES)
        await store.put("raw/paper.pdf", b"%PDF fake bytes")
        await store.put("raw/slides.docx", b"docx fake bytes")
        await store.put("raw/archive.zip", b"unsupported")
        await ParseDocuments().run(context)

    asyncio.run(run())

    result = context.artifacts["parse_documents_result"]
    assert len(result.document_keys) == 6
    assert result.skipped_keys == ("raw/archive.zip",)

    async def read_documents():
        documents = []
        for key in result.document_keys:
            documents.append(ParsedDocument.from_json(await store.get(key)))
        return documents

    documents = asyncio.run(read_documents())
    text_by_name = {document.source.name: document.pages[0].text for document in documents}

    assert text_by_name["readme.md"] == "# Heta\n\nParser smoke"
    assert "Title: Demo" in text_by_name["page.html"]
    assert "Hello HTML" in text_by_name["page.html"]
    assert "| Model | Score |" in text_by_name["table.csv"]
    assert "Image description: described image with image/png" in text_by_name["red.png"]
    assert "extracted paper.pdf" in text_by_name["paper.pdf"]
    assert "extracted slides.docx" in text_by_name["slides.docx"]


def test_parse_documents_skips_unsupported_files_by_default(tmp_path):
    store = LocalObjectStore(tmp_path)
    registry = DocumentParserRegistry([TextParser()])
    context = FakeContext(
        {
            "stores.objects": store,
            "parsers.documents": registry,
        }
    )

    async def run():
        await store.put("raw/archive.zip", b"zip")
        await ParseDocuments().run(context)

    asyncio.run(run())

    result = context.artifacts["parse_documents_result"]
    assert result.document_keys == ()
    assert result.skipped_keys == ("raw/archive.zip",)


def test_parse_documents_strict_mode_rejects_unsupported_files(tmp_path):
    store = LocalObjectStore(tmp_path)
    registry = DocumentParserRegistry([TextParser()])
    context = FakeContext(
        {
            "stores.objects": store,
            "parsers.documents": registry,
        }
    )

    async def run():
        await store.put("raw/archive.zip", b"zip")
        await ParseDocuments(ParseDocumentsConfig(skip_unsupported=False)).run(context)

    with pytest.raises(ValueError, match="no parser registered"):
        asyncio.run(run())


def test_parse_documents_supports_named_components(tmp_path):
    store = LocalObjectStore(tmp_path)
    registry = DocumentParserRegistry([TextParser()])
    context = FakeContext(
        {
            "stores.objects.local": store,
            "parsers.documents.strict": registry,
        }
    )
    step = ParseDocuments(
        ParseDocumentsConfig(
            raw_prefix="incoming",
            parsed_prefix="outputs/parsed",
            object_store="local",
            parser_registry="strict",
        )
    )

    async def run():
        await store.put("incoming/doc.txt", b"hello")
        await step.run(context)

    asyncio.run(run())

    keys = context.artifacts["parsed_document_keys"]
    assert len(keys) == 1
    assert keys[0].startswith("outputs/parsed/doc_")


def test_parse_documents_config_validates_prefixes():
    with pytest.raises(ValueError, match="relative"):
        ParseDocumentsConfig(raw_prefix="/raw")
