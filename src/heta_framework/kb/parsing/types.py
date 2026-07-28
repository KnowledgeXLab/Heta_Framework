"""Data types for parsed knowledge base documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedSource:
    """Original object metadata for one parsed document."""

    key: str
    name: str
    file_type: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.key.strip() == "":
            raise ValueError("key must not be empty")
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.file_type.strip() == "":
            raise ValueError("file_type must not be empty")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a full SHA-256 hex digest")
        try:
            int(self.content_sha256, 16)
        except ValueError as exc:
            raise ValueError("content_sha256 must be hex encoded") from exc


@dataclass(frozen=True)
class ParsedPage:
    """Parsed text for one page-like unit."""

    page_index: int
    text: str

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must not be negative")


@dataclass(frozen=True)
class ParsedTextContent:
    """Optional document-level textual content preserved by a parser."""

    text: str
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        if self.media_type.strip() == "":
            raise ValueError("media_type must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable dictionary."""
        return {"text": self.text, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParsedTextContent":
        """Create parsed text content from a dictionary."""
        return cls(text=data["text"], media_type=data["media_type"])


@dataclass(frozen=True)
class ParsedDocument:
    """Unified parser output consumed by downstream KB steps."""

    document_id: str
    source: ParsedSource
    pages: list[ParsedPage]
    original_content: ParsedTextContent | None = None

    def __post_init__(self) -> None:
        if self.document_id.strip() == "":
            raise ValueError("document_id must not be empty")
        if not self.pages:
            raise ValueError("pages must not be empty")
        indexes = [page.page_index for page in self.pages]
        if len(indexes) != len(set(indexes)):
            raise ValueError("page_index values must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the document to compact JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def to_json_bytes(self) -> bytes:
        """Serialize the document to UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedDocument":
        """Create a parsed document from a dictionary."""
        raw_original_content = data.get("original_content")
        return cls(
            document_id=data["document_id"],
            source=ParsedSource(**data["source"]),
            pages=[ParsedPage(**page) for page in data["pages"]],
            original_content=(
                ParsedTextContent.from_dict(raw_original_content)
                if raw_original_content is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "ParsedDocument":
        """Create a parsed document from JSON text or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))


def compute_content_sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest for raw document bytes."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def make_document_id(content_sha256: str) -> str:
    """Create a stable document id from a SHA-256 hex digest."""
    source = ParsedSource(
        key="_",
        name="_",
        file_type="_",
        content_sha256=content_sha256,
    )
    return f"doc_{source.content_sha256[:16]}"


def make_parsed_source(*, key: str, name: str, file_type: str, data: bytes) -> ParsedSource:
    """Create source metadata from raw document bytes."""
    return ParsedSource(
        key=key,
        name=name,
        file_type=file_type,
        content_sha256=compute_content_sha256(data),
    )
