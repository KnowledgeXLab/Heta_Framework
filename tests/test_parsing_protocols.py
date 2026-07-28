import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserProtocol,
    ParsedDocument,
    ParsedPage,
    ParsedTextContent,
    compute_content_sha256,
    make_document_id,
    make_parsed_source,
)


def test_parsed_document_serializes_to_expected_shape():
    data = "hello heta".encode("utf-8")
    source = make_parsed_source(
        key="raw/rag_paper.pdf",
        name="rag_paper.pdf",
        file_type="pdf",
        data=data,
    )
    document = ParsedDocument(
        document_id=make_document_id(source.content_sha256),
        source=source,
        pages=[ParsedPage(page_index=0, text="这一页完整文本")],
    )

    payload = json.loads(document.to_json())

    assert payload == {
        "document_id": "doc_546b1adae4cd1140",
        "source": {
            "key": "raw/rag_paper.pdf",
            "name": "rag_paper.pdf",
            "file_type": "pdf",
            "content_sha256": (
                "546b1adae4cd1140e2693fc2d81794a8cbb01260540551b"
                "5513fb52dee41df8a"
            ),
        },
        "pages": [{"page_index": 0, "text": "这一页完整文本"}],
        "original_content": None,
    }
    assert ParsedDocument.from_json(document.to_json_bytes()) == document


def test_parsed_document_round_trips_optional_original_content():
    data = b"%PDF"
    source = make_parsed_source(
        key="raw/paper.pdf",
        name="paper.pdf",
        file_type="pdf",
        data=data,
    )
    markdown = "\n# Title\n\n| A | B |\n| - | - |\n| 1 | 2 |\n"
    document = ParsedDocument(
        document_id=make_document_id(source.content_sha256),
        source=source,
        pages=[ParsedPage(page_index=0, text="Title\n\nA B\n1 2")],
        original_content=ParsedTextContent(
            text=markdown,
            media_type="text/markdown",
        ),
    )

    payload = json.loads(document.to_json())
    restored = ParsedDocument.from_json(document.to_json_bytes())

    assert payload["original_content"] == {
        "text": markdown,
        "media_type": "text/markdown",
    }
    assert restored == document
    assert restored.original_content is not None
    assert restored.original_content.text == markdown


def test_parsed_document_reads_v0_1_0_json_without_original_content():
    source = make_parsed_source(
        key="raw/legacy.txt",
        name="legacy.txt",
        file_type="txt",
        data=b"legacy content",
    )
    legacy_payload = ParsedDocument(
        document_id=make_document_id(source.content_sha256),
        source=source,
        pages=[ParsedPage(page_index=0, text="legacy content")],
    ).to_dict()
    legacy_payload.pop("original_content")

    document = ParsedDocument.from_json(json.dumps(legacy_payload))

    assert document.pages == [ParsedPage(page_index=0, text="legacy content")]
    assert document.original_content is None


def test_document_parser_protocol_accepts_structural_parser():
    class TextParser:
        @property
        def supported_file_types(self) -> set[str]:
            return {"txt", "md"}

        async def parse(self, source, data):
            return ParsedDocument(
                document_id=make_document_id(source.content_sha256),
                source=source,
                pages=[ParsedPage(page_index=0, text=data.decode("utf-8"))],
            )

    parser = TextParser()
    source = make_parsed_source(key="raw/doc.txt", name="doc.txt", file_type="txt", data=b"hello")

    async def run():
        return await parser.parse(source, b"hello")

    assert isinstance(parser, DocumentParserProtocol)
    assert asyncio.run(run()).pages[0].text == "hello"


def test_parsed_source_requires_full_sha256():
    with pytest.raises(ValueError, match="full SHA-256"):
        make_document_id("abc")


def test_compute_content_sha256_requires_bytes():
    with pytest.raises(TypeError):
        compute_content_sha256("not bytes")  # type: ignore[arg-type]


def test_parsed_document_rejects_duplicate_page_indexes():
    data = b"hello"
    source = make_parsed_source(key="raw/doc.txt", name="doc.txt", file_type="txt", data=data)

    with pytest.raises(ValueError, match="unique"):
        ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[
                ParsedPage(page_index=0, text="a"),
                ParsedPage(page_index=0, text="b"),
            ],
        )
