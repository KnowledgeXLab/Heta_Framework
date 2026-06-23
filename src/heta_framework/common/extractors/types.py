"""Shared data types for document extraction providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class DocumentInput:
    """Raw document bytes passed to an extraction provider."""

    data: bytes
    filename: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("data must be bytes")
        if self.filename.strip() == "":
            raise ValueError("filename must not be empty")


@dataclass(frozen=True)
class ExtractionOptions:
    """Provider-neutral options for document extraction."""

    language: str = "ch"
    enable_table: bool = True
    enable_formula: bool = True
    include_images: bool = True

    def __post_init__(self) -> None:
        if self.language.strip() == "":
            raise ValueError("language must not be empty")


@dataclass(frozen=True)
class BoundingBox:
    """Page-space bounding box for extracted content."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left:
            raise ValueError("right must be greater than or equal to left")
        if self.bottom < self.top:
            raise ValueError("bottom must be greater than or equal to top")


@dataclass(frozen=True)
class ExtractedAsset:
    """Binary or stored asset produced by an extractor."""

    name: str
    media_type: str | None = None
    data: bytes | None = None
    key: str | None = None
    content_sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.data is not None and not isinstance(self.data, bytes):
            raise TypeError("data must be bytes when provided")
        if self.key is not None and self.key.strip() == "":
            raise ValueError("key must not be empty when provided")
        if self.content_sha256 is not None and len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a full SHA-256 hex digest")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")


@dataclass(frozen=True)
class ExtractedBlock:
    """One ordered content block from an extracted document."""

    kind: str
    text: str = ""
    page_index: int | None = None
    bbox: BoundingBox | None = None
    asset: ExtractedAsset | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind.strip() == "":
            raise ValueError("kind must not be empty")
        if self.page_index is not None and self.page_index < 0:
            raise ValueError("page_index must not be negative")


@dataclass(frozen=True)
class ExtractedDocument:
    """Provider-neutral document extraction result."""

    markdown: str = ""
    text: str = ""
    blocks: tuple[ExtractedBlock, ...] = ()
    assets: tuple[ExtractedAsset, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_text(self) -> str:
        """Render blocks to retrieval-friendly plain text."""
        if self.blocks:
            return render_extracted_blocks(self.blocks)
        return self.text or self.markdown


def render_extracted_blocks(blocks: tuple[ExtractedBlock, ...] | list[ExtractedBlock]) -> str:
    """Render ordered extraction blocks to text without changing their order."""
    chunks: list[str] = []
    for block in blocks:
        text = block.text.strip()
        if block.kind in {"image", "picture"}:
            label = "Image"
            if block.asset and block.asset.key:
                chunks.append(f"{label}: {block.asset.key}")
            elif block.asset:
                chunks.append(f"{label}: {block.asset.name}")
            if text:
                chunks.append(f"{label} description: {text}")
            continue
        if block.kind == "table":
            chunks.append(f"Table:\n{text}" if text else "Table:")
            continue
        if text:
            chunks.append(text)
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())
