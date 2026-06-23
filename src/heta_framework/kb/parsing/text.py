"""Plain text and Markdown parser."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.kb.parsing.types import (
    ParsedDocument,
    ParsedPage,
    ParsedSource,
    make_document_id,
)


@dataclass(frozen=True)
class TextParserConfig:
    """Configuration for text parsers."""

    encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030", "latin-1")

    def __post_init__(self) -> None:
        if not self.encodings:
            raise ValueError("encodings must not be empty")
        if any(encoding.strip() == "" for encoding in self.encodings):
            raise ValueError("encodings must not contain empty values")


class TextParser:
    """Parse plain text and Markdown into one ParsedDocument page."""

    supported_file_types = {"txt", "text", "md", "markdown"}

    def __init__(self, config: TextParserConfig | None = None) -> None:
        self.config = config or TextParserConfig()

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw text bytes into a ParsedDocument."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        file_type = source.file_type.lower().lstrip(".")
        if file_type not in self.supported_file_types:
            raise ValueError(f"unsupported file type for TextParser: {source.file_type}")

        text = _decode_text(data, self.config.encodings)
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[ParsedPage(page_index=0, text=text)],
        )


def _decode_text(data: bytes, encodings: tuple[str, ...]) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return data.decode("utf-8")
