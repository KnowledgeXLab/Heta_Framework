"""Data types for graph stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GraphNode:
    """One graph node to be stored."""

    id: str
    labels: tuple[str, ...] = ("Entity",)
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        labels = tuple(label.strip() for label in self.labels if label.strip())
        if not labels:
            raise ValueError("labels must not be empty")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "properties", dict(self.properties))


@dataclass(frozen=True)
class GraphEdge:
    """One directed graph edge to be stored."""

    id: str
    source_id: str
    target_id: str
    type: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.source_id.strip() == "":
            raise ValueError("source_id must not be empty")
        if self.target_id.strip() == "":
            raise ValueError("target_id must not be empty")
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must be different")
        if self.type.strip() == "":
            raise ValueError("type must not be empty")
        object.__setattr__(self, "type", self.type.strip())
        object.__setattr__(self, "properties", dict(self.properties))
