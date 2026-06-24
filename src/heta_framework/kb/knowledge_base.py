"""Knowledge base user-facing object."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorStoreProtocol
from heta_framework.kb.builder import KnowledgeBaseBuilder, KnowledgeBaseBuilderConfig
from heta_framework.kb.cleanup import (
    CleanupIssue,
    CleanupTarget,
    KnowledgeBaseDeletePlan,
    KnowledgeBaseDeleteResult,
)
from heta_framework.kb.manifests import (
    MANIFEST_SCHEMA_VERSION,
    KnowledgeBaseManifest,
    run_record_to_dict,
)
from heta_framework.kb.recipe import KnowledgeRecipe
from heta_framework.kb.runtime import KnowledgeBaseAlreadyExistsError, KnowledgeBaseRuntime
from heta_framework.kb.search import (
    QueryContext,
    QueryEngineRegistry,
    QueryRequest,
    QueryResponse,
    SearchAssetCollection,
)
from heta_framework.kb.state import RecipeRunRecord, RecipeRunState
from heta_framework.kb.steps.types import ComponentRef


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
        runtime = KnowledgeBaseRuntime(name)
        runtime_store = _runtime_object_store(recipe)
        run_state: RecipeRunState | None = None
        previous_record: RecipeRunRecord | None = None
        resume_existing = False

        if runtime_store is not None and await runtime_store.exists(runtime.latest_run_key):
            latest = _loads_json(await runtime_store.get(runtime.latest_run_key))
            run_id = str(latest["run_id"])
            run_state = await RecipeRunState.load(
                object_store=runtime_store,
                state_key=runtime.state_key(run_id),
            )
            if run_state.status == "succeeded":
                raise KnowledgeBaseAlreadyExistsError(
                    f"knowledge base already exists and succeeded: {name}"
                )
            previous_record = run_state.to_record()
            resume_existing = True
        elif runtime_store is not None:
            run_id = f"run_{_run_token()}"
            run_state = RecipeRunState.start(
                run_id=run_id,
                started_at=_utc_now(),
                object_store=runtime_store,
                state_key=runtime.state_key(run_id),
            )
            await run_state.save()
            await _put_json(
                runtime_store,
                runtime.latest_run_key,
                {
                    "run_id": run_id,
                    "state_key": runtime.state_key(run_id),
                    "record_key": runtime.record_key(run_id),
                    "status": "running",
                },
            )

        active_builder = builder or KnowledgeBaseBuilder(
            KnowledgeBaseBuilderConfig(skip_succeeded_steps=resume_existing)
        )
        result = await active_builder.build(
            recipe,
            initial_artifacts=initial_artifacts,
            previous_record=previous_record,
            run_state=run_state,
        )
        now = _utc_now()
        kb = cls(
            name=name,
            description=description,
            recipe=recipe,
            run_record=result.record,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        if runtime_store is not None:
            await _write_runtime_metadata(runtime_store, runtime, kb)
        return kb

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

    def delete_plan(self) -> KnowledgeBaseDeletePlan:
        """Return the persistent resources that would be deleted with this KB."""
        targets: list[CleanupTarget] = []
        artifacts = self.run_record.artifacts
        for step in self.recipe.expanded_steps():
            targets.extend(step.cleanup_plan(artifacts).targets)

        runtime_store = _runtime_object_store(self.recipe)
        if runtime_store is not None:
            targets.append(
                CleanupTarget(
                    kind="runtime_prefix",
                    value=KnowledgeBaseRuntime(self.name).prefix,
                    component="stores.objects",
                )
            )
        return KnowledgeBaseDeletePlan(tuple(targets))

    async def delete(self, *, dry_run: bool = False) -> KnowledgeBaseDeleteResult:
        """Delete derived KB resources without deleting raw input objects."""
        plan = self.delete_plan()
        if dry_run:
            return KnowledgeBaseDeleteResult(dry_run=True, targets=plan.targets)

        deleted_object_keys: list[str] = []
        deleted_runtime_prefixes: list[str] = []
        dropped_sql_tables: list[str] = []
        dropped_vector_collections: list[str] = []
        issues: list[CleanupIssue] = []

        for target in _ordered_cleanup_targets(plan.targets):
            try:
                if target.kind == "vector_collection":
                    vector_store = _require_vector_store(
                        self.recipe.get_component(_component_ref(target, default="stores.vector"))
                    )
                    await vector_store.drop_collection(target.value)
                    dropped_vector_collections.append(target.value)
                elif target.kind == "sql_table":
                    sql_store = _require_sql_store(
                        self.recipe.get_component(_component_ref(target, default="stores.sql"))
                    )
                    await sql_store.execute(f"DROP TABLE IF EXISTS {target.value}")
                    dropped_sql_tables.append(target.value)
                elif target.kind == "object_key":
                    object_store = _require_object_store(
                        self.recipe.get_component(_component_ref(target, default="stores.objects"))
                    )
                    await object_store.delete(target.value)
                    deleted_object_keys.append(target.value)
                elif target.kind == "runtime_prefix":
                    object_store = _require_object_store(
                        self.recipe.get_component(_component_ref(target, default="stores.objects"))
                    )
                    for item in await object_store.list(target.value):
                        await object_store.delete(item.key)
                    deleted_runtime_prefixes.append(target.value)
                else:
                    raise ValueError(f"unsupported cleanup target kind: {target.kind}")
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    CleanupIssue(
                        target=target,
                        message=str(exc) or exc.__class__.__name__,
                        error_type=exc.__class__.__name__,
                    )
                )

        return KnowledgeBaseDeleteResult(
            dry_run=False,
            targets=plan.targets,
            deleted_object_keys=tuple(deleted_object_keys),
            deleted_runtime_prefixes=tuple(deleted_runtime_prefixes),
            dropped_sql_tables=tuple(dropped_sql_tables),
            dropped_vector_collections=tuple(dropped_vector_collections),
            issues=tuple(issues),
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_token() -> str:
    from uuid import uuid4

    return uuid4().hex


def _runtime_object_store(recipe: KnowledgeRecipe) -> ObjectStoreProtocol | None:
    store = recipe.stores.objects
    if isinstance(store, ObjectStoreProtocol):
        return store
    return None


def _ordered_cleanup_targets(targets: tuple[CleanupTarget, ...]) -> tuple[CleanupTarget, ...]:
    order = {
        "vector_collection": 0,
        "sql_table": 1,
        "object_key": 2,
        "runtime_prefix": 3,
    }
    return tuple(sorted(targets, key=lambda target: order[target.kind]))


def _component_ref(target: CleanupTarget, *, default: str) -> ComponentRef:
    return _component_ref_from_key(target.component or default)


def _component_ref_from_key(key: str) -> ComponentRef:
    parts = key.split(".")
    if len(parts) == 2:
        namespace, kind = parts
        return ComponentRef(namespace=namespace, kind=kind)  # type: ignore[arg-type]
    if len(parts) == 3:
        namespace, kind, name = parts
        return ComponentRef(namespace=namespace, kind=kind, name=name)  # type: ignore[arg-type]
    raise ValueError(f"invalid component key: {key}")


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component


async def _write_runtime_metadata(
    object_store: ObjectStoreProtocol,
    runtime: KnowledgeBaseRuntime,
    kb: KnowledgeBase,
) -> None:
    run_id = kb.run_record.run_id
    await _put_json(object_store, runtime.record_key(run_id), run_record_to_dict(kb.run_record))
    await _put_json(
        object_store,
        runtime.latest_run_key,
        {
            "run_id": run_id,
            "state_key": runtime.state_key(run_id),
            "record_key": runtime.record_key(run_id),
            "status": kb.run_record.status,
        },
    )
    await _put_json(object_store, runtime.manifest_key, kb.manifest().to_dict())


async def _put_json(
    object_store: ObjectStoreProtocol,
    key: str,
    value: Mapping[str, Any],
) -> None:
    await object_store.put(
        key,
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def _loads_json(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime metadata must be a JSON object")
    return value


__all__ = ["KnowledgeBase", "KnowledgeBaseAlreadyExistsError"]
