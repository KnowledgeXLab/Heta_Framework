"""PDF parser."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.common.extractors import DocumentExtractorProtocol, ExtractionOptions
from heta_framework.kb.parsing.document import ExtractorDocumentParser, ExtractorParserConfig


@dataclass(frozen=True)
class PdfParserConfig:
    """Configuration for PDF parsing."""

    extraction_options: ExtractionOptions = ExtractionOptions()


class PdfParser(ExtractorDocumentParser):
    """Parse PDF documents through a configured document extractor."""

    def __init__(
        self,
        extractor: DocumentExtractorProtocol,
        config: PdfParserConfig | None = None,
    ) -> None:
        parser_config = config or PdfParserConfig()
        super().__init__(
            extractor,
            ExtractorParserConfig(
                supported_file_types=("pdf",),
                extraction_options=parser_config.extraction_options,
            ),
            parser_name="PdfParser",
        )
        self.config = parser_config
