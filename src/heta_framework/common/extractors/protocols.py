"""Extractor capability protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heta_framework.common.extractors.types import (
    DocumentInput,
    ExtractedDocument,
    ExtractionOptions,
)


@runtime_checkable
class DocumentExtractorProtocol(Protocol):
    """Capability protocol for document content extraction providers."""

    async def extract(
        self,
        document: DocumentInput,
        options: ExtractionOptions | None = None,
    ) -> ExtractedDocument:
        """Extract structured content from one document."""
        ...
