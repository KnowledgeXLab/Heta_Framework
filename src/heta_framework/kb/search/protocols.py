"""Protocols for knowledge base query engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from heta_framework.kb.search.assets import SearchAssetCollection, SearchAssetRef
from heta_framework.kb.search.types import QueryRequest, QueryResponse

if TYPE_CHECKING:
    from heta_framework.kb.recipe import KnowledgeRecipe
    from heta_framework.kb.search.registry import QueryEngineRegistry
    from heta_framework.kb.state import RecipeRunRecord


@dataclass(frozen=True)
class QueryContext:
    """Runtime context passed to a query engine."""

    recipe: KnowledgeRecipe
    run_record: RecipeRunRecord
    assets: SearchAssetCollection
    engines: QueryEngineRegistry
    call_stack: tuple[str, ...] = field(default_factory=tuple)

    async def query(self, mode: str, request: QueryRequest) -> QueryResponse:
        """Run another query engine through the shared validation path."""
        query_mode = _normalize_mode(mode)
        if query_mode in self.call_stack:
            chain = " -> ".join((*self.call_stack, query_mode))
            raise RuntimeError(f"recursive query engine call detected: {chain}")

        engine = self.engines.get(query_mode)
        missing_assets = self.assets.missing(engine.required_assets)
        if missing_assets:
            missing = ", ".join(ref.key for ref in missing_assets)
            raise LookupError(f"{query_mode} requires missing search asset(s): {missing}")
        missing_components = [
            ref for ref in _required_components(engine) if not self.recipe.has_component(ref)
        ]
        if missing_components:
            missing = ", ".join(ref.key for ref in missing_components)
            raise LookupError(f"{query_mode} requires missing component(s): {missing}")

        next_context = QueryContext(
            recipe=self.recipe,
            run_record=self.run_record,
            assets=self.assets,
            engines=self.engines,
            call_stack=(*self.call_stack, query_mode),
        )
        return await engine.query(_request_with_mode(request, query_mode), next_context)


@runtime_checkable
class QueryEngineProtocol(Protocol):
    """Minimal protocol implemented by query engines."""

    @property
    def mode(self) -> str:
        """Return the query mode implemented by this engine."""
        ...

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return search assets required before this engine can run."""
        ...

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run one query."""
        ...


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip()
    if normalized == "":
        raise ValueError("mode must not be empty")
    return normalized


def _request_with_mode(request: QueryRequest, mode: str) -> QueryRequest:
    if request.mode == mode:
        return request
    return QueryRequest(
        text=request.text,
        mode=mode,
        top_k=request.top_k,
        filters=request.filters,
        options=request.options,
        trace=request.trace,
    )


def _required_components(engine: QueryEngineProtocol) -> frozenset[object]:
    refs = getattr(engine, "required_components", frozenset())
    return refs if isinstance(refs, frozenset) else frozenset(refs)
