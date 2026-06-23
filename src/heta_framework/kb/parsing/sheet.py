"""Spreadsheet parser."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

from heta_framework.common.models import ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.kb.parsing.prompts import DEFAULT_TABLE_DESCRIPTION_PROMPT
from heta_framework.kb.parsing.types import ParsedDocument, ParsedPage, ParsedSource, make_document_id


@dataclass(frozen=True)
class SheetParserConfig:
    """Configuration for spreadsheet parsing."""

    encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
    preview_rows: int = 3
    chunk_rows: int = 50
    describe_tables: bool = False
    description_prompt: str = DEFAULT_TABLE_DESCRIPTION_PROMPT

    def __post_init__(self) -> None:
        if not self.encodings:
            raise ValueError("encodings must not be empty")
        if any(encoding.strip() == "" for encoding in self.encodings):
            raise ValueError("encodings must not contain empty values")
        if self.preview_rows < 0:
            raise ValueError("preview_rows must not be negative")
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be greater than zero")
        if self.description_prompt.strip() == "":
            raise ValueError("description_prompt must not be empty")


class SheetParser:
    """Parse CSV and spreadsheet files into text pages."""

    supported_file_types = {"csv", "xls", "xlsx", "xlsm", "xlsb", "ods", "odf", "odt"}

    def __init__(
        self,
        config: SheetParserConfig | None = None,
        *,
        language_model: LanguageModelProtocol | None = None,
    ) -> None:
        self.config = config or SheetParserConfig()
        self._language_model = language_model
        if self.config.describe_tables and self._language_model is None:
            raise ValueError("language_model is required when describe_tables is enabled")

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw spreadsheet bytes into a ParsedDocument."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        file_type = source.file_type.lower().lstrip(".")
        if file_type not in self.supported_file_types:
            raise ValueError(f"unsupported file type for SheetParser: {source.file_type}")

        tables = _reader_for(file_type, encodings=self.config.encodings).read(
            data,
            filename=source.name,
        )
        pages: list[ParsedPage] = []
        page_index = 0
        for table in tables:
            description = await self._describe_table(source.name, table) if self.config.describe_tables else ""
            for chunk in _table_chunks(
                table,
                source_name=source.name,
                description=description,
                chunk_rows=self.config.chunk_rows,
            ):
                pages.append(ParsedPage(page_index=page_index, text=chunk))
                page_index += 1

        if not pages:
            raise ValueError("sheet does not contain readable rows")
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=pages,
        )

    async def _describe_table(self, source_name: str, table: "_Table") -> str:
        if self._language_model is None:
            return ""
        prompt = (
            f"{self.config.description_prompt}\n\n"
            f"File: {source_name}\n"
            f"Sheet: {table.name}\n"
            f"Columns: {', '.join(table.headers)}\n"
            f"Preview:\n{_rows_to_markdown(table.headers, table.rows[: self.config.preview_rows])}"
        )
        result = await self._language_model.invoke(ModelRequest(prompt=prompt))
        return result.text.strip()


@dataclass(frozen=True)
class _Table:
    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


class _SheetReader(Protocol):
    def read(self, data: bytes, *, filename: str) -> list[_Table]:
        """Read raw sheet bytes into tables."""
        ...


@dataclass(frozen=True)
class _CsvSheetReader:
    encodings: tuple[str, ...]

    def read(self, data: bytes, *, filename: str) -> list[_Table]:
        del filename
        text = _decode_table_text(data, self.encodings)
        reader = csv.reader(io.StringIO(text))
        rows = [[cell.strip() for cell in row] for row in reader]
        return [_table_from_rows("csv_sheet", rows)]


class _CalamineSheetReader:
    def read(self, data: bytes, *, filename: str) -> list[_Table]:
        del filename
        workbook = _open_calamine_workbook(data)
        try:
            return [
                _table_from_rows(sheet_name, workbook.get_sheet_by_name(sheet_name).to_python())
                for sheet_name in workbook.sheet_names
            ]
        finally:
            close = getattr(workbook, "close", None)
            if close is not None:
                close()


def _reader_for(file_type: str, *, encodings: tuple[str, ...]) -> _SheetReader:
    if file_type == "csv":
        return _CsvSheetReader(encodings=encodings)
    return _CalamineSheetReader()


def _table_from_rows(name: str, rows: list[list[Any]]) -> _Table:
    rows = [[_cell_text(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return _Table(name=name, headers=(), rows=())

    width = max(len(row) for row in rows)
    headers = tuple(_header_text(cell, index) for index, cell in enumerate(_pad_row(rows[0], width)))
    body = tuple(tuple(_pad_row(row, width)) for row in rows[1:])
    return _Table(name=name, headers=headers, rows=body)


def _open_calamine_workbook(data: bytes) -> object:
    try:
        from python_calamine import CalamineWorkbook
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal runtime.
        raise ImportError(
            "python-calamine is required to parse xls, xlsx, xlsm, xlsb, ods, odf, and odt files"
        ) from exc

    return CalamineWorkbook.from_filelike(io.BytesIO(data))


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _header_text(value: str, index: int) -> str:
    return value if value else f"column_{index + 1}"


def _decode_table_text(data: bytes, encodings: tuple[str, ...]) -> str:
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode(encodings[0], errors="replace")


def _table_chunks(
    table: _Table,
    *,
    source_name: str,
    description: str,
    chunk_rows: int,
) -> list[str]:
    if not table.headers and not table.rows:
        return []
    chunks = []
    rows = table.rows or ((),)
    for start in range(0, len(rows), chunk_rows):
        row_chunk = rows[start : start + chunk_rows]
        title = f"Table: {source_name}"
        if table.name:
            title += f" / {table.name}"
        if start:
            title += " (continued)"
        parts = [title]
        if description and start == 0:
            parts.append(f"Description: {description}")
        parts.append(_rows_to_markdown(table.headers, row_chunk))
        chunks.append("\n\n".join(part for part in parts if part.strip()))
    return chunks


def _rows_to_markdown(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...] | list[tuple[str, ...]]) -> str:
    width = len(headers) or max((len(row) for row in rows), default=0)
    normalized_headers = _pad_row(list(headers), width) if headers else [f"column_{idx + 1}" for idx in range(width)]
    lines = [
        "| " + " | ".join(_markdown_cell(cell) for cell in normalized_headers) + " |",
        "| " + " | ".join("---" for _ in normalized_headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in _pad_row(list(row), width)) + " |")
    return "\n".join(lines)


def _pad_row(row: list[Any] | tuple[Any, ...], width: int) -> list[str]:
    values = [str(value) for value in row[:width]]
    if len(values) < width:
        values.extend("" for _ in range(width - len(values)))
    return values


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
