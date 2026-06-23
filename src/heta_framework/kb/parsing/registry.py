"""Registry for routing files to document parsers."""

from __future__ import annotations

from collections.abc import Iterable

from heta_framework.kb.parsing.protocols import DocumentParserProtocol
from heta_framework.kb.parsing.types import ParsedDocument, ParsedSource


class DocumentParserRegistry:
    """Route parsed sources to registered document parsers."""

    def __init__(self, parsers: Iterable[DocumentParserProtocol] = ()) -> None:
        self._parsers: list[DocumentParserProtocol] = []
        self._parser_by_file_type: dict[str, DocumentParserProtocol] = {}
        for parser in parsers:
            self.register(parser)

    @property
    def supported_file_types(self) -> set[str]:
        """Return all file types supported by registered parsers."""
        return set(self._parser_by_file_type)

    @property
    def parsers(self) -> tuple[DocumentParserProtocol, ...]:
        """Return registered parsers that are still active for at least one file type."""
        active = set(self._parser_by_file_type.values())
        return tuple(parser for parser in self._parsers if parser in active)

    def register(
        self,
        parser: DocumentParserProtocol,
        *,
        replace: bool = False,
    ) -> "DocumentParserRegistry":
        """Register a parser for its supported file types."""
        if not isinstance(parser, DocumentParserProtocol):
            raise TypeError("parser must satisfy DocumentParserProtocol")

        file_types = _normalize_file_types(parser.supported_file_types)
        conflicts = {
            file_type: registered
            for file_type, registered in self._parser_by_file_type.items()
            if file_type in file_types and registered is not parser
        }
        if conflicts and not replace:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"parser already registered for file type(s): {names}")

        if parser not in self._parsers:
            self._parsers.append(parser)
        for file_type in file_types:
            self._parser_by_file_type[file_type] = parser
        return self

    def unregister(self, file_type: str) -> DocumentParserProtocol:
        """Remove and return the parser registered for one file type."""
        normalized = _normalize_file_type(file_type)
        try:
            return self._parser_by_file_type.pop(normalized)
        except KeyError as exc:
            raise ValueError(f"no parser registered for file type: {file_type}") from exc

    def find_parser(self, file_type: str) -> DocumentParserProtocol | None:
        """Return the parser for a file type, or None when unsupported."""
        return self._parser_by_file_type.get(_normalize_file_type(file_type))

    def get_parser(self, file_type: str) -> DocumentParserProtocol:
        """Return the parser for a file type, raising when unsupported."""
        parser = self.find_parser(file_type)
        if parser is None:
            supported = ", ".join(sorted(self.supported_file_types)) or "none"
            raise ValueError(f"no parser registered for file type: {file_type}; supported: {supported}")
        return parser

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse a source by routing it to the registered parser for its file type."""
        return await self.get_parser(source.file_type).parse(source, data)


def _normalize_file_types(file_types: Iterable[str]) -> set[str]:
    normalized = {_normalize_file_type(file_type) for file_type in file_types}
    if not normalized:
        raise ValueError("parser.supported_file_types must not be empty")
    return normalized


def _normalize_file_type(file_type: str) -> str:
    normalized = file_type.strip().lower().lstrip(".")
    if normalized == "":
        raise ValueError("file_type must not be empty")
    return normalized
