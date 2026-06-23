"""Typed graph-building artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExtractedEntity:
    """One entity extracted from a parsed chunk."""

    entity_id: str
    chunk_id: str
    document_id: str
    name: str
    type: str
    subtype: str | None
    description: str
    attributes: Mapping[str, str]
    source_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.entity_id.strip() == "":
            raise ValueError("entity_id must not be empty")
        if self.chunk_id.strip() == "":
            raise ValueError("chunk_id must not be empty")
        if self.document_id.strip() == "":
            raise ValueError("document_id must not be empty")
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.type.strip() == "":
            raise ValueError("type must not be empty")
        if self.subtype is not None and self.subtype.strip() == "":
            raise ValueError("subtype must not be empty")
        if self.description.strip() == "":
            raise ValueError("description must not be empty")
        if not self.source_chunk_ids:
            raise ValueError("source_chunk_ids must not be empty")
        normalized_source_ids = tuple(
            chunk_id for chunk_id in self.source_chunk_ids if chunk_id.strip()
        )
        if len(normalized_source_ids) != len(self.source_chunk_ids):
            raise ValueError("source_chunk_ids must not contain empty values")
        normalized_attributes = {
            str(key).strip(): str(value).strip()
            for key, value in self.attributes.items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "attributes", normalized_attributes)
        object.__setattr__(self, "source_chunk_ids", normalized_source_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the entity to compact JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def to_json_bytes(self) -> bytes:
        """Serialize the entity to UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedEntity":
        """Create an extracted entity from a dictionary."""
        return cls(
            entity_id=data["entity_id"],
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            name=data["name"],
            type=data["type"],
            subtype=data.get("subtype"),
            description=data["description"],
            attributes=data.get("attributes", {}),
            source_chunk_ids=tuple(data["source_chunk_ids"]),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "ExtractedEntity":
        """Create an extracted entity from JSON text or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))


def make_entity_id(
    *,
    document_id: str,
    chunk_id: str,
    name: str,
    type: str,
    subtype: str | None,
    description: str,
) -> str:
    """Create a stable entity id from source and semantic fields."""
    payload = "\n".join(
        [
            document_id,
            chunk_id,
            name.strip(),
            type.strip(),
            (subtype or "").strip(),
            description.strip(),
        ]
    ).encode("utf-8")
    return f"entity_{hashlib.sha256(payload).hexdigest()[:16]}"


def make_deduplicated_entity_id(*, member_entity_ids: tuple[str, ...], name: str) -> str:
    """Create a stable entity id for a deduplicated entity group."""
    payload = "\n".join([name.strip(), *sorted(member_entity_ids)]).encode("utf-8")
    return f"entity_{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class ExtractedRelation:
    """One relation extracted from a parsed chunk and its entities."""

    relation_id: str
    chunk_id: str
    document_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    type: str
    name: str
    description: str
    attributes: Mapping[str, str]
    source_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.relation_id.strip() == "":
            raise ValueError("relation_id must not be empty")
        if self.chunk_id.strip() == "":
            raise ValueError("chunk_id must not be empty")
        if self.document_id.strip() == "":
            raise ValueError("document_id must not be empty")
        if self.source_entity_id.strip() == "":
            raise ValueError("source_entity_id must not be empty")
        if self.target_entity_id.strip() == "":
            raise ValueError("target_entity_id must not be empty")
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("source_entity_id and target_entity_id must be different")
        if self.source_entity_name.strip() == "":
            raise ValueError("source_entity_name must not be empty")
        if self.target_entity_name.strip() == "":
            raise ValueError("target_entity_name must not be empty")
        if self.type.strip() == "":
            raise ValueError("type must not be empty")
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.description.strip() == "":
            raise ValueError("description must not be empty")
        if not self.source_chunk_ids:
            raise ValueError("source_chunk_ids must not be empty")
        normalized_source_ids = tuple(
            chunk_id for chunk_id in self.source_chunk_ids if chunk_id.strip()
        )
        if len(normalized_source_ids) != len(self.source_chunk_ids):
            raise ValueError("source_chunk_ids must not contain empty values")
        normalized_attributes = {
            str(key).strip(): str(value).strip()
            for key, value in self.attributes.items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "attributes", normalized_attributes)
        object.__setattr__(self, "source_chunk_ids", normalized_source_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the relation to compact JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def to_json_bytes(self) -> bytes:
        """Serialize the relation to UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedRelation":
        """Create an extracted relation from a dictionary."""
        return cls(
            relation_id=data["relation_id"],
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            source_entity_name=data["source_entity_name"],
            target_entity_name=data["target_entity_name"],
            type=data["type"],
            name=data["name"],
            description=data["description"],
            attributes=data.get("attributes", {}),
            source_chunk_ids=tuple(data["source_chunk_ids"]),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "ExtractedRelation":
        """Create an extracted relation from JSON text or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))


def make_relation_id(
    *,
    document_id: str,
    chunk_id: str,
    source_entity_id: str,
    target_entity_id: str,
    type: str,
    name: str,
    description: str,
) -> str:
    """Create a stable relation id from source, endpoints, and semantic fields."""
    payload = "\n".join(
        [
            document_id,
            chunk_id,
            source_entity_id,
            target_entity_id,
            type.strip(),
            name.strip(),
            description.strip(),
        ]
    ).encode("utf-8")
    return f"relation_{hashlib.sha256(payload).hexdigest()[:16]}"


def make_deduplicated_relation_id(*, member_relation_ids: tuple[str, ...], name: str) -> str:
    """Create a stable relation id for a deduplicated relation group."""
    payload = "\n".join([name.strip(), *sorted(member_relation_ids)]).encode("utf-8")
    return f"relation_{hashlib.sha256(payload).hexdigest()[:16]}"
