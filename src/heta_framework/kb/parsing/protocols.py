"""Parser capability protocols for knowledge base recipes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heta_framework.kb.parsing.types import ParsedDocument, ParsedSource


@runtime_checkable
class DocumentParserProtocol(Protocol):
    """Capability protocol for parsers that produce ParsedDocument objects."""

    @property
    def supported_file_types(self) -> set[str]:
        """Lowercase file types supported by this parser, without leading dots."""
        ...

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw bytes from one source object."""
        ...
