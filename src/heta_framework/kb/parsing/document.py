"""Document parser helpers backed by common extractors."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass

from heta_framework.common.extractors import (
    DocumentExtractorProtocol,
    DocumentInput,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionOptions,
)
from heta_framework.kb.parsing.types import ParsedDocument, ParsedPage, ParsedSource, make_document_id


@dataclass(frozen=True)
class ExtractorParserConfig:
    """Configuration for parsers backed by a document extractor."""

    supported_file_types: tuple[str, ...]
    extraction_options: ExtractionOptions = ExtractionOptions()

    def __post_init__(self) -> None:
        if not self.supported_file_types:
            raise ValueError("supported_file_types must not be empty")
        if any(file_type.strip() == "" for file_type in self.supported_file_types):
            raise ValueError("supported_file_types must not contain empty values")


class ExtractorDocumentParser:
    """Shared implementation for parsers that use a document extractor."""

    def __init__(
        self,
        extractor: DocumentExtractorProtocol,
        config: ExtractorParserConfig,
        *,
        parser_name: str,
    ) -> None:
        self._extractor = extractor
        self.config = config
        self._parser_name = parser_name
        self.supported_file_types = {
            file_type.lower().lstrip(".") for file_type in self.config.supported_file_types
        }

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw document bytes into a ParsedDocument."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        file_type = source.file_type.lower().lstrip(".")
        if file_type not in self.supported_file_types:
            raise ValueError(f"unsupported file type for {self._parser_name}: {source.file_type}")

        extracted = await self._extractor.extract(
            DocumentInput(
                data=data,
                filename=source.name,
                media_type=mimetypes.guess_type(source.name)[0],
            ),
            options=self.config.extraction_options,
        )
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=pages_from_extracted_document(extracted),
        )


def pages_from_extracted_document(document: ExtractedDocument) -> list[ParsedPage]:
    """Convert ordered extracted blocks into page-like ParsedPage objects."""
    page_blocks: dict[int, list[ExtractedBlock]] = {}
    current_page_index = 0
    for block in document.blocks:
        if block.page_index is not None:
            current_page_index = block.page_index
        page_blocks.setdefault(current_page_index, []).append(block)

    if page_blocks:
        pages = []
        for page_index, blocks in sorted(page_blocks.items()):
            text = _render_page(blocks).strip()
            if text:
                pages.append(ParsedPage(page_index=page_index, text=text))
        if pages:
            return pages

    text = document.to_text().strip()
    if not text:
        raise ValueError("extracted document does not contain text")
    return [ParsedPage(page_index=0, text=text)]


def _render_page(blocks: list[ExtractedBlock]) -> str:
    from heta_framework.common.extractors import render_extracted_blocks

    return render_extracted_blocks(blocks)
