"""Data types for chunked knowledge base text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from heta_framework.kb.parsing import ParsedSource


@dataclass(frozen=True)
class ParsedChunk:
    """One retrieval-ready text chunk derived from a ParsedDocument page."""

    chunk_id: str
    document_id: str
    source: ParsedSource
    page_index: int
    chunk_index: int
    text: str
    token_start: int
    token_end: int
    parent_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.chunk_id.strip() == "":
            raise ValueError("chunk_id must not be empty")
        if self.document_id.strip() == "":
            raise ValueError("document_id must not be empty")
        if self.page_index < 0:
            raise ValueError("page_index must not be negative")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        if self.token_start < 0:
            raise ValueError("token_start must not be negative")
        if self.token_end < self.token_start:
            raise ValueError("token_end must be greater than or equal to token_start")
        normalized_parent_ids = tuple(parent_id for parent_id in self.parent_chunk_ids if parent_id.strip())
        if len(normalized_parent_ids) != len(self.parent_chunk_ids):
            raise ValueError("parent_chunk_ids must not contain empty values")
        object.__setattr__(self, "parent_chunk_ids", normalized_parent_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the chunk to compact JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def to_json_bytes(self) -> bytes:
        """Serialize the chunk to UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedChunk":
        """Create a parsed chunk from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source=ParsedSource(**data["source"]),
            page_index=data["page_index"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            token_start=data["token_start"],
            token_end=data["token_end"],
            parent_chunk_ids=tuple(data.get("parent_chunk_ids", ())),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "ParsedChunk":
        """Create a parsed chunk from JSON text or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))


def make_chunk_id(*, document_id: str, page_index: int, chunk_index: int, text: str) -> str:
    """Create a stable chunk id from document position and text."""
    payload = f"{document_id}\n{page_index}\n{chunk_index}\n{text}".encode("utf-8")
    return f"chunk_{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class ChunkEmbedding:
    """Embedding vector generated for one ParsedChunk."""

    chunk_id: str
    document_id: str
    model_name: str
    vector: list[float]
    dimension: int

    def __post_init__(self) -> None:
        if self.chunk_id.strip() == "":
            raise ValueError("chunk_id must not be empty")
        if self.document_id.strip() == "":
            raise ValueError("document_id must not be empty")
        if self.model_name.strip() == "":
            raise ValueError("model_name must not be empty")
        if not self.vector:
            raise ValueError("vector must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if len(self.vector) != self.dimension:
            raise ValueError("dimension must match vector length")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the embedding to compact JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def to_json_bytes(self) -> bytes:
        """Serialize the embedding to UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkEmbedding":
        """Create a chunk embedding from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            model_name=data["model_name"],
            vector=[float(value) for value in data["vector"]],
            dimension=data["dimension"],
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "ChunkEmbedding":
        """Create a chunk embedding from JSON text or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))
