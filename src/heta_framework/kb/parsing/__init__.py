"""Parser protocols and data types for Heta knowledge bases."""

from heta_framework.kb.parsing.protocols import DocumentParserProtocol
from heta_framework.kb.parsing.html import (
    BasicHtmlExtractor,
    HtmlParser,
    HtmlParserConfig,
)
from heta_framework.kb.parsing.image import ImageParser, ImageParserConfig
from heta_framework.kb.parsing.office import OfficeParser, OfficeParserConfig
from heta_framework.kb.parsing.pdf import PdfParser, PdfParserConfig
from heta_framework.kb.parsing.registry import DocumentParserRegistry
from heta_framework.kb.parsing.sheet import SheetParser, SheetParserConfig
from heta_framework.kb.parsing.text import TextParser, TextParserConfig
from heta_framework.kb.parsing.types import (
    ParsedDocument,
    ParsedPage,
    ParsedSource,
    compute_content_sha256,
    make_document_id,
    make_parsed_source,
)

__all__ = [
    "BasicHtmlExtractor",
    "DocumentParserProtocol",
    "DocumentParserRegistry",
    "HtmlParser",
    "HtmlParserConfig",
    "ImageParser",
    "ImageParserConfig",
    "OfficeParser",
    "OfficeParserConfig",
    "ParsedDocument",
    "ParsedPage",
    "ParsedSource",
    "PdfParser",
    "PdfParserConfig",
    "SheetParser",
    "SheetParserConfig",
    "TextParser",
    "TextParserConfig",
    "compute_content_sha256",
    "make_document_id",
    "make_parsed_source",
]
