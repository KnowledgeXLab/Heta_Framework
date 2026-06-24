"""Knowledge base builder and step execution context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import uuid4

from heta_framework.kb.recipe import KnowledgeRecipe
from heta_framework.kb.state import RecipeRunRecord, RecipeRunResult, RecipeRunState, StepRunRecord
from heta_framework.kb.steps import (
    ComponentRef,
    StepCapabilities,
    StepContextProtocol,
    StepIssue,
)


@dataclass(frozen=True)
class KnowledgeBaseBuilderConfig:
    """Configuration for KnowledgeBaseBuilder."""

    stop_on_error: bool = True
    skip_succeeded_steps: bool = False


@dataclass
class StepExecutionContext:
    """Runtime context passed to one recipe step."""

    recipe: KnowledgeRecipe
    artifacts: dict[str, Any]

    def get_component(self, key: str) -> Any:
        """Return a recipe component by stable key."""
        return self.recipe.get_component(_component_ref_from_key(key))

    def get_artifact(self, key: str) -> Any:
        """Return an artifact by key."""
        try:
            return self.artifacts[key]
        except KeyError as exc:
            raise KeyError(f"missing artifact: {key}") from exc

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact by key."""
        if key.strip() == "":
            raise ValueError("artifact key must not be empty")
        self.artifacts[key] = value


class KnowledgeBaseBuilder:
    """Build knowledge bases from KnowledgeRecipe objects."""

    def __init__(self, config: KnowledgeBaseBuilderConfig | None = None) -> None:
        self.config = config or KnowledgeBaseBuilderConfig()

    async def build(
        self,
        recipe: KnowledgeRecipe,
        *,
        initial_artifacts: Mapping[str, Any] | None = None,
        previous_record: RecipeRunRecord | None = None,
        run_state: RecipeRunState | None = None,
    ) -> RecipeRunResult:
        """Run a knowledge recipe and return a build result."""
        artifacts: dict[str, Any] = {}
        if previous_record is not None:
            artifacts.update(previous_record.artifacts)
        if initial_artifacts is not None:
            artifacts.update(initial_artifacts)

        recipe.require_valid(initial_artifacts=artifacts.keys())

        started_at = _utc_now()
        state = run_state or RecipeRunState.start(
            run_id=f"run_{uuid4().hex}",
            started_at=started_at,
        )
        context = StepExecutionContext(recipe=recipe, artifacts=artifacts)
        steps = recipe.expanded_steps()
        previous_success = _previous_success_records(previous_record)

        all_issues: list[StepIssue] = []
        run_status = "succeeded"

        for index, step in enumerate(steps):
            previous_step = previous_success.get((index, step.name, type(step).__name__))
            if self.config.skip_succeeded_steps and previous_step is not None:
                await state.skip_step(
                    StepRunRecord(
                        index=index,
                        step_name=step.name,
                        step_type=type(step).__name__,
                        status="skipped",
                        started_at=None,
                        finished_at=None,
                        requirements=step.requirements,
                        capabilities=step.capabilities,
                        input_artifacts=tuple(sorted(step.requirements.artifacts)),
                        output_artifacts=previous_step.output_artifacts,
                        issues=previous_step.issues,
                    ),
                    artifacts=context.artifacts,
                )
                all_issues.extend(previous_step.issues)
                continue

            before_artifacts = set(context.artifacts)
            step_started_at = _utc_now()
            await state.start_step(
                StepRunRecord(
                    index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    status="running",
                    started_at=step_started_at,
                    finished_at=None,
                    requirements=step.requirements,
                    capabilities=step.capabilities,
                    input_artifacts=tuple(sorted(step.requirements.artifacts)),
                    output_artifacts=(),
                ),
                artifacts=context.artifacts,
            )
            try:
                await step.run(context)
            except Exception as exc:  # noqa: BLE001
                finished_at = _utc_now()
                step_record = StepRunRecord(
                    index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    status="failed",
                    started_at=step_started_at,
                    finished_at=finished_at,
                    requirements=step.requirements,
                    capabilities=step.capabilities,
                    input_artifacts=tuple(sorted(step.requirements.artifacts)),
                    output_artifacts=tuple(sorted(set(context.artifacts) - before_artifacts)),
                    issues=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                await state.finish_step(step_record, artifacts=context.artifacts)
                run_status = "failed"
                if self.config.stop_on_error:
                    break
                continue

            finished_at = _utc_now()
            output_artifacts = tuple(sorted(set(context.artifacts) - before_artifacts))
            step_issues = _issues_from_artifacts(
                context.artifacts,
                output_artifacts,
            )
            all_issues.extend(step_issues)
            await state.finish_step(
                StepRunRecord(
                    index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    status="succeeded",
                    started_at=step_started_at,
                    finished_at=finished_at,
                    requirements=step.requirements,
                    capabilities=step.capabilities,
                    input_artifacts=tuple(sorted(step.requirements.artifacts)),
                    output_artifacts=output_artifacts,
                    issues=step_issues,
                ),
                artifacts=context.artifacts,
                issues=step_issues,
            )

        capabilities = _capabilities_from_step_records(state.step_records)
        finished_at = _utc_now()
        await state.finish_run(
            status=run_status,  # type: ignore[arg-type]
            finished_at=finished_at,
            artifacts=context.artifacts,
            issues=tuple(all_issues),
        )
        record = state.to_record(
            artifacts=dict(context.artifacts),
            capabilities=capabilities,
            issues=tuple(all_issues),
        )
        return RecipeRunResult(
            record=record,
            artifacts=record.artifacts,
            capabilities=record.capabilities,
            issues=record.issues,
        )


def _component_ref_from_key(key: str) -> ComponentRef:
    parts = key.split(".")
    if len(parts) == 2:
        namespace, kind = parts
        return ComponentRef(namespace=namespace, kind=kind)  # type: ignore[arg-type]
    if len(parts) == 3:
        namespace, kind, name = parts
        return ComponentRef(namespace=namespace, kind=kind, name=name)  # type: ignore[arg-type]
    raise ValueError(f"invalid component key: {key}")


def _previous_success_records(
    previous_record: RecipeRunRecord | None,
) -> dict[tuple[int, str, str], StepRunRecord]:
    if previous_record is None:
        return {}
    return {
        (record.index, record.step_name, record.step_type): record
        for record in previous_record.step_records
        if record.status == "succeeded"
    }


def _issues_from_artifacts(
    artifacts: Mapping[str, Any],
    output_artifacts: tuple[str, ...],
) -> tuple[StepIssue, ...]:
    issues: list[StepIssue] = []
    for key in output_artifacts:
        value = artifacts.get(key)
        value_issues = getattr(value, "issues", None)
        if isinstance(value_issues, tuple):
            issues.extend(issue for issue in value_issues if isinstance(issue, StepIssue))
    return tuple(issues)


def _capabilities_from_step_records(records: list[StepRunRecord]) -> StepCapabilities:
    artifacts: set[str] = set()
    queries: set[str] = set()
    search_assets = []
    for record in records:
        if record.status not in {"succeeded", "skipped"}:
            continue
        artifacts.update(record.capabilities.artifacts)
        queries.update(record.capabilities.queries)
        search_assets.extend(record.capabilities.search_assets)
    return StepCapabilities(
        artifacts=frozenset(artifacts),
        queries=frozenset(queries),
        search_assets=tuple(search_assets),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "KnowledgeBaseBuilder",
    "KnowledgeBaseBuilderConfig",
    "StepExecutionContext",
]
