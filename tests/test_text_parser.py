import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserProtocol,
    TextParser,
    TextParserConfig,
    make_document_id,
    make_parsed_source,
)


def test_text_parser_satisfies_protocol():
    assert isinstance(TextParser(), DocumentParserProtocol)


def test_text_parser_parses_utf8_text():
    data = "Heta parser\n第二行".encode("utf-8")
    source = make_parsed_source(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        data=data,
    )

    async def run():
        return await TextParser().parse(source, data)

    document = asyncio.run(run())

    assert document.document_id == make_document_id(source.content_sha256)
    assert document.source == source
    assert document.pages[0].page_index == 0
    assert document.pages[0].text == "Heta parser\n第二行"


def test_text_parser_parses_markdown_as_text():
    data = b"# Title\n\n- item"
    source = make_parsed_source(
        key="raw/doc.md",
        name="doc.md",
        file_type="md",
        data=data,
    )

    async def run():
        return await TextParser().parse(source, data)

    document = asyncio.run(run())

    assert document.pages[0].text == "# Title\n\n- item"


def test_text_parser_falls_back_to_configured_encoding():
    data = "中文".encode("gb18030")
    source = make_parsed_source(
        key="raw/doc.txt",
        name="doc.txt",
        file_type="txt",
        data=data,
    )

    async def run():
        return await TextParser().parse(source, data)

    document = asyncio.run(run())

    assert document.pages[0].text == "中文"


def test_text_parser_rejects_unsupported_file_type():
    data = b"hello"
    source = make_parsed_source(
        key="raw/doc.pdf",
        name="doc.pdf",
        file_type="pdf",
        data=data,
    )

    async def run():
        return await TextParser().parse(source, data)

    with pytest.raises(ValueError, match="unsupported file type"):
        asyncio.run(run())


def test_text_parser_config_requires_encodings():
    with pytest.raises(ValueError, match="encodings"):
        TextParserConfig(encodings=())
