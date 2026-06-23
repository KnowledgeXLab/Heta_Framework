"""Office document parser."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.common.extractors import DocumentExtractorProtocol, ExtractionOptions
from heta_framework.kb.parsing.document import ExtractorDocumentParser, ExtractorParserConfig


@dataclass(frozen=True)
class OfficeParserConfig:
    """Configuration for Office document parsing."""

    supported_file_types: tuple[str, ...] = ("doc", "docx", "ppt", "pptx")
    extraction_options: ExtractionOptions = ExtractionOptions()

    def __post_init__(self) -> None:
        if not self.supported_file_types:
            raise ValueError("supported_file_types must not be empty")
        if any(file_type.strip() == "" for file_type in self.supported_file_types):
            raise ValueError("supported_file_types must not contain empty values")


class OfficeParser(ExtractorDocumentParser):
    """Parse Office documents through a configured document extractor."""

    def __init__(
        self,
        extractor: DocumentExtractorProtocol,
        config: OfficeParserConfig | None = None,
    ) -> None:
        parser_config = config or OfficeParserConfig()
        super().__init__(
            extractor,
            ExtractorParserConfig(
                supported_file_types=parser_config.supported_file_types,
                extraction_options=parser_config.extraction_options,
            ),
            parser_name="OfficeParser",
        )
        self.config = parser_config
