import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserRegistry,
    ParsedDocument,
    ParsedPage,
    make_document_id,
    make_parsed_source,
)


class FakeParser:
    def __init__(self, file_types, label):
        self.supported_file_types = set(file_types)
        self.label = label
        self.calls = []

    async def parse(self, source, data):
        self.calls.append((source.file_type, data))
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[ParsedPage(page_index=0, text=f"{self.label}: {data.decode('utf-8')}")],
        )


def test_registry_routes_by_file_type():
    text_parser = FakeParser({"txt", "md"}, "text")
    html_parser = FakeParser({"html"}, "html")
    registry = DocumentParserRegistry([text_parser, html_parser])
    source = make_parsed_source(key="raw/doc.md", name="doc.md", file_type="md", data=b"hello")

    async def run():
        return await registry.parse(source, b"hello")

    document = asyncio.run(run())

    assert document.pages[0].text == "text: hello"
    assert text_parser.calls == [("md", b"hello")]
    assert html_parser.calls == []


def test_registry_can_only_include_selected_parsers():
    registry = DocumentParserRegistry([FakeParser({"txt"}, "text")])

    assert registry.find_parser("txt") is not None
    assert registry.find_parser("html") is None
    assert registry.supported_file_types == {"txt"}


def test_registry_normalizes_file_types():
    parser = FakeParser({".TXT"}, "text")
    registry = DocumentParserRegistry([parser])

    assert registry.get_parser(" txt ") is parser
    assert registry.get_parser(".txt") is parser


def test_registry_rejects_file_type_conflicts():
    registry = DocumentParserRegistry([FakeParser({"txt"}, "first")])

    with pytest.raises(ValueError, match="txt"):
        registry.register(FakeParser({"txt"}, "second"))


def test_registry_can_replace_registered_parser():
    first = FakeParser({"txt"}, "first")
    second = FakeParser({"txt"}, "second")
    registry = DocumentParserRegistry([first])

    registry.register(second, replace=True)

    assert registry.get_parser("txt") is second
    assert registry.parsers == (second,)


def test_registry_unregisters_one_file_type():
    parser = FakeParser({"txt", "md"}, "text")
    registry = DocumentParserRegistry([parser])

    removed = registry.unregister("txt")

    assert removed is parser
    assert registry.find_parser("txt") is None
    assert registry.find_parser("md") is parser
    assert registry.parsers == (parser,)


def test_registry_reports_unsupported_file_type():
    registry = DocumentParserRegistry([FakeParser({"txt"}, "text")])

    with pytest.raises(ValueError, match="supported: txt"):
        registry.get_parser("pdf")


def test_registry_rejects_invalid_parser():
    registry = DocumentParserRegistry()

    with pytest.raises(TypeError, match="DocumentParserProtocol"):
        registry.register(object())  # type: ignore[arg-type]
