"""Static knowledge recipe definitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Mapping

from heta_framework.kb.components import (
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeStores,
    MissingComponentError,
)
from heta_framework.kb.manifests import (
    MANIFEST_SCHEMA_VERSION,
    KnowledgeRecipeManifest,
    StepManifest,
)
from heta_framework.kb.procedures import KnowledgeProcedureProtocol
from heta_framework.kb.steps import (
    ComponentRef,
    KnowledgeStepProtocol,
    StepCapabilities,
)
from heta_framework.kb.validation import (
    RecipeValidationError,
    RecipeValidationIssue,
    RecipeValidationResult,
)

RecipeItem = KnowledgeStepProtocol | KnowledgeProcedureProtocol


@dataclass(frozen=True)
class KnowledgeRecipe:
    """Static construction plan for a knowledge base."""

    models: KnowledgeModels = field(default_factory=KnowledgeModels)
    stores: KnowledgeStores = field(default_factory=KnowledgeStores)
    parsers: KnowledgeParsers = field(default_factory=KnowledgeParsers)
    steps: tuple[RecipeItem, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def expanded_steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Return recipe steps with procedures expanded in place."""
        expanded: list[KnowledgeStepProtocol] = []
        for item in self.steps:
            if isinstance(item, KnowledgeProcedureProtocol):
                expanded.extend(item.steps())
            else:
                expanded.append(item)
        return tuple(expanded)

    def get_component(self, ref: ComponentRef) -> object:
        """Return a recipe component by reference."""
        if ref.namespace == "models":
            return self.models.get(ref)
        if ref.namespace == "stores":
            return self.stores.get(ref)
        if ref.namespace == "parsers":
            return self.parsers.get(ref)
        raise MissingComponentError(f"unsupported component namespace: {ref.key}")

    def has_component(self, ref: ComponentRef) -> bool:
        """Return whether this recipe can resolve a component reference."""
        try:
            self.get_component(ref)
        except MissingComponentError:
            return False
        return True

    def validate(
        self,
        *,
        initial_artifacts: Iterable[str] = (),
        initial_queries: Iterable[str] = (),
    ) -> RecipeValidationResult:
        """Validate static recipe dataflow without touching runtime resources."""
        issues: list[RecipeValidationIssue] = []
        available_artifacts = set(initial_artifacts)
        available_queries = set(initial_queries)
        produced_artifacts: dict[str, int] = {}

        for index, step in enumerate(self.expanded_steps()):
            requirements = step.requirements
            capabilities = step.capabilities

            for ref in sorted(requirements.components):
                if not self.has_component(ref):
                    issues.append(
                        RecipeValidationIssue(
                            severity="error",
                            code="missing_component",
                            message=f"Step requires missing component: {ref.key}",
                            step_index=index,
                            step_name=step.name,
                            details={"component": ref.key},
                        )
                    )

            missing_artifacts = requirements.artifacts - available_artifacts
            for artifact in sorted(missing_artifacts):
                issues.append(
                    RecipeValidationIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"Step requires unavailable artifact: {artifact}",
                        step_index=index,
                        step_name=step.name,
                        details={"artifact": artifact},
                    )
                )

            missing_queries = requirements.queries - available_queries
            for query in sorted(missing_queries):
                issues.append(
                    RecipeValidationIssue(
                        severity="error",
                        code="missing_query",
                        message=f"Step requires unavailable query capability: {query}",
                        step_index=index,
                        step_name=step.name,
                        details={"query": query},
                    )
                )

            for artifact in sorted(capabilities.artifacts):
                previous_index = produced_artifacts.get(artifact)
                if previous_index is not None:
                    issues.append(
                        RecipeValidationIssue(
                            severity="warning",
                            code="duplicate_artifact_output",
                            message=f"Artifact is produced by multiple steps: {artifact}",
                            step_index=index,
                            step_name=step.name,
                            details={
                                "artifact": artifact,
                                "previous_step_index": str(previous_index),
                            },
                        )
                    )
                produced_artifacts[artifact] = index

            available_artifacts.update(capabilities.artifacts)
            available_queries.update(capabilities.queries)

        return RecipeValidationResult(tuple(issues))

    def require_valid(
        self,
        *,
        initial_artifacts: Iterable[str] = (),
        initial_queries: Iterable[str] = (),
    ) -> None:
        """Raise if static recipe validation fails."""
        result = self.validate(
            initial_artifacts=initial_artifacts,
            initial_queries=initial_queries,
        )
        if not result.valid:
            raise RecipeValidationError(result)

    def manifest(self) -> KnowledgeRecipeManifest:
        """Return a persistable recipe manifest."""
        steps = self.expanded_steps()
        component_refs = {
            ref.key for step in steps for ref in step.requirements.components
        }
        artifacts_required = {
            artifact for step in steps for artifact in step.requirements.artifacts
        }
        capabilities = _merge_capabilities(step.capabilities for step in steps)
        return KnowledgeRecipeManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            steps=tuple(
                StepManifest(
                    index=index,
                    name=step.name,
                    type=type(step).__name__,
                    requirements=step.requirements,
                    capabilities=step.capabilities,
                )
                for index, step in enumerate(steps)
            ),
            component_refs=tuple(sorted(component_refs)),
            artifacts_required=tuple(sorted(artifacts_required)),
            capabilities_provided=tuple(sorted(capabilities.queries)),
            metadata=self.metadata,
        )


def _merge_capabilities(capabilities: Iterable[StepCapabilities]) -> StepCapabilities:
    artifacts: set[str] = set()
    queries: set[str] = set()
    search_assets = []
    for capability in capabilities:
        artifacts.update(capability.artifacts)
        queries.update(capability.queries)
        search_assets.extend(capability.search_assets)
    return StepCapabilities(
        artifacts=frozenset(artifacts),
        queries=frozenset(queries),
        search_assets=tuple(search_assets),
    )
