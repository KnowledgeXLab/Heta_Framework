"""Knowledge base user-facing object."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from heta_framework.kb.builder import KnowledgeBaseBuilder, KnowledgeBaseBuilderConfig
from heta_framework.kb.manifests import MANIFEST_SCHEMA_VERSION, KnowledgeBaseManifest
from heta_framework.kb.recipe import KnowledgeRecipe
from heta_framework.kb.search import (
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    SearchAssetCollection,
)
from heta_framework.kb.state import RecipeRunRecord


@dataclass(frozen=True)
class KnowledgeBase:
    """A built knowledge base and its latest build record."""

    name: str
    description: str | None
    recipe: KnowledgeRecipe
    run_record: RecipeRunRecord
    created_at: str
    updated_at: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    query_engines: QueryEngineRegistry = field(default_factory=QueryEngineRegistry.defaults)

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.created_at.strip() == "":
            raise ValueError("created_at must not be empty")
        if self.updated_at.strip() == "":
            raise ValueError("updated_at must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def search_assets(self) -> SearchAssetCollection:
        """Return queryable assets produced by the latest build."""
        return SearchAssetCollection(self.run_record.capabilities.search_assets)

    @property
    def available_queries(self) -> frozenset[str]:
        """Return query modes supported by this knowledge base."""
        return self.query_engines.available_modes_for(self.recipe, self.search_assets)

    @classmethod
    async def create(
        cls,
        *,
        recipe: KnowledgeRecipe,
        name: str,
        description: str | None = None,
        initial_artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
        builder: KnowledgeBaseBuilder | None = None,
    ) -> "KnowledgeBase":
        """Build a knowledge base from a recipe."""
        active_builder = builder or KnowledgeBaseBuilder()
        result = await active_builder.build(recipe, initial_artifacts=initial_artifacts)
        now = _utc_now()
        return cls(
            name=name,
            description=description,
            recipe=recipe,
            run_record=result.record,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    @classmethod
    def restore(
        cls,
        *,
        manifest: KnowledgeBaseManifest,
        recipe: KnowledgeRecipe,
    ) -> "KnowledgeBase":
        """Restore knowledge base metadata with a runtime recipe."""
        return cls(
            name=manifest.name,
            description=manifest.description,
            recipe=recipe,
            run_record=manifest.run_record,
            created_at=manifest.created_at,
            updated_at=manifest.updated_at,
            metadata=manifest.metadata,
        )

    async def resume(
        self,
        *,
        builder: KnowledgeBaseBuilder | None = None,
        initial_artifacts: Mapping[str, Any] | None = None,
    ) -> "KnowledgeBase":
        """Resume a knowledge base build from the current run record."""
        active_builder = builder or KnowledgeBaseBuilder(
            KnowledgeBaseBuilderConfig(skip_succeeded_steps=True)
        )
        result = await active_builder.build(
            self.recipe,
            initial_artifacts=initial_artifacts,
            previous_record=self.run_record,
        )
        return KnowledgeBase(
            name=self.name,
            description=self.description,
            recipe=self.recipe,
            run_record=result.record,
            created_at=self.created_at,
            updated_at=_utc_now(),
            metadata=self.metadata,
            query_engines=self.query_engines,
        )

    async def query(
        self,
        text: str | QueryRequest,
        *,
        mode: str | None = None,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        trace: bool = False,
    ) -> QueryResponse:
        """Run a query against this knowledge base."""
        request = (
            text
            if isinstance(text, QueryRequest)
            else QueryRequest(
                text=text,
                mode=mode,
                top_k=top_k,
                filters=filters or {},
                options=options or {},
                trace=trace,
            )
        )
        query_mode = request.mode
        if query_mode is None:
            available = sorted(self.available_queries)
            if len(available) != 1:
                raise ValueError(
                    "query mode is required when zero or multiple query modes are available"
                )
            query_mode = available[0]

        context = QueryContext(
            recipe=self.recipe,
            run_record=self.run_record,
            assets=self.search_assets,
            engines=self.query_engines,
        )
        return await context.query(query_mode, request)

    def manifest(self) -> KnowledgeBaseManifest:
        """Return a persistable knowledge base manifest."""
        return KnowledgeBaseManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name=self.name,
            description=self.description,
            created_at=self.created_at,
            updated_at=self.updated_at,
            recipe=self.recipe.manifest(),
            run_record=self.run_record,
            metadata=self.metadata,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["KnowledgeBase"]
