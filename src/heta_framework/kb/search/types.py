"""Shared data types for knowledge base queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryRequest:
    """One query request against a built knowledge base."""

    text: str
    mode: str | None = None
    top_k: int = 10
    filters: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    trace: bool = False

    def __post_init__(self) -> None:
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        if self.mode is not None and self.mode.strip() == "":
            raise ValueError("mode must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "mode", self.mode.strip() if self.mode else None)
        object.__setattr__(self, "filters", dict(self.filters))
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class QueryResult:
    """One result returned by a query engine."""

    id: str
    text: str
    score: float | None = None
    kind: str = "chunk"
    source: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.text.strip() == "":
            raise ValueError("text must not be empty")
        if self.kind.strip() == "":
            raise ValueError("kind must not be empty")
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class QueryCitation:
    """Citation or provenance item attached to a query response."""

    id: str
    result_id: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("id must not be empty")
        if self.result_id is not None and self.result_id.strip() == "":
            raise ValueError("result_id must not be empty")
        if self.text is not None and self.text.strip() == "":
            raise ValueError("text must not be empty")
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "result_id", self.result_id.strip() if self.result_id else None)
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "text", self.text.strip() if self.text else None)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class QueryTraceEvent:
    """Structured trace event produced while serving a query."""

    stage: str
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage.strip() == "":
            raise ValueError("stage must not be empty")
        if self.message.strip() == "":
            raise ValueError("message must not be empty")
        object.__setattr__(self, "stage", self.stage.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class QueryResponse:
    """Response returned by a query engine."""

    mode: str
    results: tuple[QueryResult, ...] = ()
    answer: str | None = None
    citations: tuple[QueryCitation, ...] = ()
    trace: tuple[QueryTraceEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode.strip() == "":
            raise ValueError("mode must not be empty")
        if self.answer is not None and self.answer.strip() == "":
            raise ValueError("answer must not be empty")
        object.__setattr__(self, "mode", self.mode.strip())
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "answer", self.answer.strip() if self.answer else None)
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "trace", tuple(self.trace))
        object.__setattr__(self, "metadata", dict(self.metadata))
