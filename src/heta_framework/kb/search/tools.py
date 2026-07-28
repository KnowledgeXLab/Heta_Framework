"""Tool protocol for agentic knowledge base query engines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from heta_framework.common.models.language.types import ToolDefinition
from heta_framework.kb.search.assets import SearchAsset, SearchAssetCollection, SearchAssetRef
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryCitation, QueryRequest, QueryResponse, QueryResult
from heta_framework.kb.steps.types import ComponentRef

if TYPE_CHECKING:
    from heta_framework.kb.recipe import KnowledgeRecipe


@dataclass(frozen=True)
class QueryToolBudget:
    """Execution bounds shared by an agentic query engine and its tools."""

    max_tool_calls: int = 8
    max_tool_result_chars: int = 8_000
    max_evidence_chars: int = 24_000

    def __post_init__(self) -> None:
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than zero")
        if self.max_tool_result_chars <= 0:
            raise ValueError("max_tool_result_chars must be greater than zero")
        if self.max_evidence_chars <= 0:
            raise ValueError("max_evidence_chars must be greater than zero")


@dataclass(frozen=True)
class QueryToolResult:
    """Result returned by one query tool invocation."""

    content: str
    results: tuple[QueryResult, ...] = ()
    citations: tuple[QueryCitation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def __post_init__(self) -> None:
        if self.content.strip() == "":
            raise ValueError("content must not be empty")
        object.__setattr__(self, "content", self.content.strip())
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def error(
        cls,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> QueryToolResult:
        """Create a model-visible error result without raising from the tool loop."""
        return cls(content=content, metadata=metadata or {}, is_error=True)


@dataclass(frozen=True)
class QueryEvidenceRecord:
    """One structured evidence item captured from a query tool result."""

    evidence_id: str
    tool_name: str
    content: str
    tool_call_id: str | None = None
    results: tuple[QueryResult, ...] = ()
    citations: tuple[QueryCitation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def __post_init__(self) -> None:
        if self.evidence_id.strip() == "":
            raise ValueError("evidence_id must not be empty")
        if self.tool_name.strip() == "":
            raise ValueError("tool_name must not be empty")
        if self.content.strip() == "":
            raise ValueError("content must not be empty")
        if self.tool_call_id is not None and self.tool_call_id.strip() == "":
            raise ValueError("tool_call_id must not be empty")
        object.__setattr__(self, "evidence_id", self.evidence_id.strip())
        object.__setattr__(self, "tool_name", self.tool_name.strip())
        object.__setattr__(self, "content", self.content.strip())
        object.__setattr__(
            self,
            "tool_call_id",
            self.tool_call_id.strip() if self.tool_call_id else None,
        )
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class QueryEvidenceLedger:
    """Mutable ledger of structured evidence gathered during one agentic query."""

    records: list[QueryEvidenceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.records = list(self.records)
        evidence_ids = [record.evidence_id for record in self.records]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("records must not contain duplicate evidence_id values")

    @property
    def evidence_records(self) -> tuple[QueryEvidenceRecord, ...]:
        """Return evidence records in insertion order."""
        return tuple(self.records)

    @property
    def results(self) -> tuple[QueryResult, ...]:
        """Return all query results captured in the ledger."""
        return tuple(result for record in self.records for result in record.results)

    @property
    def citations(self) -> tuple[QueryCitation, ...]:
        """Return all citations captured in the ledger."""
        return tuple(citation for record in self.records for citation in record.citations)

    @property
    def total_content_chars(self) -> int:
        """Return the total model-visible evidence size."""
        return sum(len(record.content) for record in self.records)

    def add(self, record: QueryEvidenceRecord) -> QueryEvidenceRecord:
        """Append one prebuilt evidence record."""
        if any(existing.evidence_id == record.evidence_id for existing in self.records):
            raise ValueError(f"duplicate evidence_id: {record.evidence_id}")
        self.records.append(record)
        return record

    def add_result(
        self,
        *,
        tool_name: str,
        result: QueryToolResult,
        tool_call_id: str | None = None,
    ) -> QueryEvidenceRecord:
        """Append evidence derived from one tool result."""
        record = QueryEvidenceRecord(
            evidence_id=self._next_evidence_id(tool_name),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content=result.content,
            results=result.results,
            citations=result.citations,
            metadata=result.metadata,
            is_error=result.is_error,
        )
        return self.add(record)

    def to_context_text(self, *, max_chars: int | None = None) -> str:
        """Return a compact text view suitable for final answer synthesis."""
        chunks = [
            f"[{record.evidence_id}] {record.tool_name}\n{record.content}"
            for record in self.records
        ]
        text = "\n\n".join(chunks)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n..."
        return text

    def _next_evidence_id(self, tool_name: str) -> str:
        slug = _slug_tool_name(tool_name)
        existing = {record.evidence_id for record in self.records}
        base = f"evidence_{len(self.records) + 1:03d}_{slug}"
        if base not in existing:
            return base
        suffix = 2
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"


@dataclass(frozen=True)
class QueryToolContext:
    """Runtime context passed to a query tool."""

    query_context: QueryContext
    request: QueryRequest
    ledger: QueryEvidenceLedger = field(default_factory=QueryEvidenceLedger)
    budget: QueryToolBudget = field(default_factory=QueryToolBudget)

    async def query(
        self,
        mode: str,
        request: QueryRequest | None = None,
    ) -> QueryResponse:
        """Run a registered query engine through the shared query context."""
        return await self.query_context.query(mode, request or self.request)

    def require_asset(self, ref: SearchAssetRef) -> SearchAsset:
        """Return a search asset required by a tool."""
        return self.query_context.assets.require(ref)

    def missing_assets(self, refs: Iterable[SearchAssetRef]) -> tuple[SearchAssetRef, ...]:
        """Return search assets that are unavailable to a tool."""
        return self.query_context.assets.missing(refs)

    def get_component(self, ref: ComponentRef) -> object:
        """Return a recipe component required by a tool."""
        return self.query_context.recipe.get_component(ref)


@runtime_checkable
class QueryToolProtocol(Protocol):
    """Protocol implemented by tools used inside agentic query engines."""

    @property
    def name(self) -> str:
        """Return the unique model-visible tool name."""
        ...

    @property
    def description(self) -> str:
        """Return a concise model-visible tool description."""
        ...

    @property
    def parameters_schema(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        ...

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return search assets required before this tool can run."""
        ...

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        """Return recipe components required before this tool can run."""
        ...

    async def run(
        self,
        arguments: Mapping[str, Any],
        context: QueryToolContext,
    ) -> QueryToolResult:
        """Run one tool invocation."""
        ...


class QueryToolRegistry:
    """Lookup, uniqueness, and model-schema helpers for query tools."""

    def __init__(self, tools: Iterable[QueryToolProtocol] = ()) -> None:
        self._tools: dict[str, QueryToolProtocol] = {}
        for tool in tools:
            self.register(tool)

    def __iter__(self) -> Iterator[QueryToolProtocol]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def tools(self) -> tuple[QueryToolProtocol, ...]:
        """Return registered tools in insertion order."""
        return tuple(self._tools.values())

    @property
    def names(self) -> frozenset[str]:
        """Return registered tool names."""
        return frozenset(self._tools)

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return the union of all registered tool asset requirements."""
        return frozenset(ref for tool in self._tools.values() for ref in tool.required_assets)

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        """Return the union of all registered tool component requirements."""
        return frozenset(ref for tool in self._tools.values() for ref in tool.required_components)

    def register(
        self,
        tool: QueryToolProtocol,
        *,
        replace: bool = False,
    ) -> QueryToolRegistry:
        """Register one query tool."""
        if not isinstance(tool, QueryToolProtocol):
            raise TypeError("tool must satisfy QueryToolProtocol")
        name = _normalize_tool_name(tool.name)
        if name in self._tools and not replace:
            raise ValueError(f"query tool already registered: {name}")
        self._tools[name] = tool
        return self

    def find(self, name: str) -> QueryToolProtocol | None:
        """Return a tool by name, or None when unregistered."""
        return self._tools.get(_normalize_tool_name(name))

    def get(self, name: str) -> QueryToolProtocol:
        """Return a tool by name, raising when it is unavailable."""
        normalized = _normalize_tool_name(name)
        try:
            return self._tools[normalized]
        except KeyError as exc:
            raise LookupError(f"query tool is not registered: {normalized}") from exc

    def model_tool_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return model-facing tool definitions for registered query tools."""
        return tuple(make_query_tool_definition(tool) for tool in self.tools)


def make_query_tool_definition(tool: QueryToolProtocol) -> ToolDefinition:
    """Convert a query tool into a provider-neutral model tool definition."""
    parameters_schema = dict(tool.parameters_schema) or {"type": "object", "properties": {}}
    return ToolDefinition(
        name=_normalize_tool_name(tool.name),
        description=tool.description,
        parameters_schema=parameters_schema,
    )


def missing_query_tool_assets(
    tool: QueryToolProtocol,
    assets: SearchAssetCollection,
) -> tuple[SearchAssetRef, ...]:
    """Return search assets that are required by a tool but missing."""
    return assets.missing(tool.required_assets)


def missing_query_tool_components(
    tool: QueryToolProtocol,
    recipe: KnowledgeRecipe,
) -> tuple[ComponentRef, ...]:
    """Return recipe components that are required by a tool but missing."""
    return tuple(ref for ref in tool.required_components if not recipe.has_component(ref))


def _normalize_tool_name(name: str) -> str:
    normalized = name.strip()
    if normalized == "":
        raise ValueError("tool name must not be empty")
    return normalized


def _slug_tool_name(name: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in _normalize_tool_name(name).lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "tool"
