import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import heta_framework.kb.parsing.sheet as sheet_module  # noqa: E402
from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.kb.parsing import (  # noqa: E402
    DocumentParserProtocol,
    SheetParser,
    SheetParserConfig,
    make_document_id,
    make_parsed_source,
)


class FakeLanguageModel:
    @property
    def model_name(self) -> str:
        return "fake-model"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        assert request.prompt is not None
        assert "Columns: Model, Score" in request.prompt
        return ModelResult(text="benchmark scores by model")

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(text_delta=(await self.invoke(request)).text)


class FakeCalamineSheet:
    def __init__(self, rows):
        self._rows = rows

    def to_python(self):
        return self._rows


class FakeCalamineWorkbook:
    sheet_names = ["Scores", "Metadata"]

    def __init__(self):
        self.closed = False

    def get_sheet_by_name(self, name):
        if name == "Scores":
            return FakeCalamineSheet(
                [
                    ["Model", "Score", ""],
                    ["gpt", 98.0, date(2026, 1, 2)],
                    ["qwen", 96, None],
                ]
            )
        return FakeCalamineSheet([["Key", "Value"], ["owner", "heta"]])

    def close(self):
        self.closed = True


def test_sheet_parser_satisfies_protocol():
    assert isinstance(SheetParser(), DocumentParserProtocol)


def test_sheet_parser_parses_csv_to_parsed_document():
    data = "Model,Score\ngpt,98\nqwen,96\n".encode("utf-8")
    source = make_parsed_source(
        key="raw/scores.csv",
        name="scores.csv",
        file_type="csv",
        data=data,
    )

    async def run():
        return await SheetParser().parse(source, data)

    document = asyncio.run(run())

    assert document.document_id == make_document_id(source.content_sha256)
    assert document.source == source
    assert document.pages[0].page_index == 0
    assert "Table: scores.csv / csv_sheet" in document.pages[0].text
    assert "| Model | Score |" in document.pages[0].text
    assert "| gpt | 98 |" in document.pages[0].text


def test_sheet_parser_chunks_rows():
    data = "Model,Score\ngpt,98\nqwen,96\n".encode("utf-8")
    source = make_parsed_source(
        key="raw/scores.csv",
        name="scores.csv",
        file_type="csv",
        data=data,
    )

    async def run():
        return await SheetParser(SheetParserConfig(chunk_rows=1)).parse(source, data)

    document = asyncio.run(run())

    assert [page.page_index for page in document.pages] == [0, 1]
    assert "| gpt | 98 |" in document.pages[0].text
    assert "continued" in document.pages[1].text
    assert "| qwen | 96 |" in document.pages[1].text


def test_sheet_parser_uses_calamine_for_excel_files(monkeypatch):
    workbook = FakeCalamineWorkbook()

    def open_workbook(data):
        assert data == b"xlsx-bytes"
        return workbook

    monkeypatch.setattr(sheet_module, "_open_calamine_workbook", open_workbook)
    source = make_parsed_source(
        key="raw/scores.xlsx",
        name="scores.xlsx",
        file_type="xlsx",
        data=b"xlsx-bytes",
    )

    async def run():
        return await SheetParser().parse(source, b"xlsx-bytes")

    document = asyncio.run(run())

    assert workbook.closed
    assert len(document.pages) == 2
    assert "Table: scores.xlsx / Scores" in document.pages[0].text
    assert "| Model | Score | column_3 |" in document.pages[0].text
    assert "| gpt | 98 | 2026-01-02 |" in document.pages[0].text
    assert "Table: scores.xlsx / Metadata" in document.pages[1].text
    assert "| owner | heta |" in document.pages[1].text


def test_sheet_parser_can_add_table_description():
    data = "Model,Score\ngpt,98\n".encode("utf-8")
    source = make_parsed_source(
        key="raw/scores.csv",
        name="scores.csv",
        file_type="csv",
        data=data,
    )

    async def run():
        return await SheetParser(
            SheetParserConfig(describe_tables=True),
            language_model=FakeLanguageModel(),
        ).parse(source, data)

    document = asyncio.run(run())

    assert "Description: benchmark scores by model" in document.pages[0].text


def test_sheet_parser_requires_model_when_describing_tables():
    with pytest.raises(ValueError, match="language_model"):
        SheetParser(SheetParserConfig(describe_tables=True))


def test_sheet_parser_rejects_unsupported_file_type():
    data = b"not a spreadsheet"
    source = make_parsed_source(
        key="raw/image.png",
        name="image.png",
        file_type="png",
        data=data,
    )

    async def run():
        return await SheetParser().parse(source, data)

    with pytest.raises(ValueError, match="unsupported file type"):
        asyncio.run(run())


def test_sheet_parser_config_validates_chunk_rows():
    with pytest.raises(ValueError, match="chunk_rows"):
        SheetParserConfig(chunk_rows=0)
