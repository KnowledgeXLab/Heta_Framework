"""Registry for knowledge base query engines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from heta_framework.kb.search.assets import SearchAssetCollection
from heta_framework.kb.search.protocols import QueryEngineProtocol

if TYPE_CHECKING:
    from heta_framework.kb.recipe import KnowledgeRecipe


class QueryEngineRegistry:
    """Lookup and availability checks for query engines."""

    def __init__(self, engines: Iterable[QueryEngineProtocol] = ()) -> None:
        self._engines: dict[str, QueryEngineProtocol] = {}
        for engine in engines:
            self.register(engine)

    @classmethod
    def defaults(cls) -> "QueryEngineRegistry":
        """Return the built-in query engine registry."""
        from heta_framework.kb.search.engines import (
            FullTextSearchEngine,
            GraphRAGGlobalQueryEngine,
            GraphRAGLocalQueryEngine,
            HetaGraphSearchEngine,
            HybridSearchEngine,
            LightRAGGlobalQueryEngine,
            LightRAGHybridQueryEngine,
            LightRAGLocalQueryEngine,
            LightRAGMixQueryEngine,
            MultiHopSearchEngine,
            RerankSearchEngine,
            RewriteSearchEngine,
            SqlTextSearchEngine,
            VectorSearchEngine,
        )

        return cls(
            [
                VectorSearchEngine(),
                SqlTextSearchEngine(),
                FullTextSearchEngine(),
                HetaGraphSearchEngine(),
                GraphRAGLocalQueryEngine(),
                GraphRAGGlobalQueryEngine(),
                LightRAGLocalQueryEngine(),
                LightRAGGlobalQueryEngine(),
                LightRAGHybridQueryEngine(),
                LightRAGMixQueryEngine(),
                HybridSearchEngine(),
                RerankSearchEngine(),
                RewriteSearchEngine(),
                MultiHopSearchEngine(),
            ]
        )

    @property
    def modes(self) -> frozenset[str]:
        """Return all registered query modes."""
        return frozenset(self._engines)

    def register(
        self,
        engine: QueryEngineProtocol,
        *,
        replace: bool = False,
    ) -> "QueryEngineRegistry":
        """Register one query engine."""
        if not isinstance(engine, QueryEngineProtocol):
            raise TypeError("engine must satisfy QueryEngineProtocol")
        mode = engine.mode.strip()
        if mode == "":
            raise ValueError("engine.mode must not be empty")
        if mode in self._engines and not replace:
            raise ValueError(f"query engine already registered for mode: {mode}")
        self._engines[mode] = engine
        return self

    def find(self, mode: str) -> QueryEngineProtocol | None:
        """Return an engine for a mode, or None when unregistered."""
        return self._engines.get(_normalize_mode(mode))

    def get(self, mode: str) -> QueryEngineProtocol:
        """Return an engine for a mode, raising when unavailable."""
        normalized = _normalize_mode(mode)
        try:
            return self._engines[normalized]
        except KeyError as exc:
            raise LookupError(f"query engine is not registered: {normalized}") from exc

    def available_modes(self, assets: SearchAssetCollection) -> frozenset[str]:
        """Return registered modes whose required assets are satisfied."""
        return frozenset(
            mode
            for mode, engine in self._engines.items()
            if _is_discoverable(engine)
            if not assets.missing(engine.required_assets)
        )

    def available_modes_for(
        self,
        recipe: "KnowledgeRecipe",
        assets: SearchAssetCollection,
    ) -> frozenset[str]:
        """Return registered modes whose asset and component requirements are satisfied."""
        return frozenset(
            mode
            for mode, engine in self._engines.items()
            if _is_discoverable(engine)
            if not assets.missing(engine.required_assets)
            and not _missing_components(recipe, engine)
        )


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip()
    if normalized == "":
        raise ValueError("mode must not be empty")
    return normalized


def _missing_components(
    recipe: "KnowledgeRecipe",
    engine: QueryEngineProtocol,
) -> tuple[object, ...]:
    refs = getattr(engine, "required_components", frozenset())
    return tuple(ref for ref in refs if not recipe.has_component(ref))


def _is_discoverable(engine: QueryEngineProtocol) -> bool:
    return bool(getattr(engine, "discoverable", True))
