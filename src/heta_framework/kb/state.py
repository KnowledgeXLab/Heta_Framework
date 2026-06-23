"""Run records for knowledge base builds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from heta_framework.kb.steps import StepCapabilities, StepIssue, StepRequirements

StepRunStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
RecipeRunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class StepRunRecord:
    """Execution record for one recipe step."""

    index: int
    step_name: str
    step_type: str
    status: StepRunStatus
    started_at: str | None
    finished_at: str | None
    requirements: StepRequirements
    capabilities: StepCapabilities
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    issues: tuple[StepIssue, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must not be negative")
        if self.step_name.strip() == "":
            raise ValueError("step_name must not be empty")
        if self.step_type.strip() == "":
            raise ValueError("step_type must not be empty")
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        object.__setattr__(self, "output_artifacts", tuple(self.output_artifacts))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True)
class RecipeRunRecord:
    """Execution record for one knowledge recipe run."""

    run_id: str
    status: RecipeRunStatus
    started_at: str
    finished_at: str | None
    step_records: tuple[StepRunRecord, ...]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    capabilities: StepCapabilities = field(default_factory=StepCapabilities)
    issues: tuple[StepIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.run_id.strip() == "":
            raise ValueError("run_id must not be empty")
        if self.started_at.strip() == "":
            raise ValueError("started_at must not be empty")
        object.__setattr__(self, "step_records", tuple(self.step_records))
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True)
class RecipeRunResult:
    """Return value from KnowledgeBaseBuilder.build."""

    record: RecipeRunRecord
    artifacts: Mapping[str, Any]
    capabilities: StepCapabilities
    issues: tuple[StepIssue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "issues", tuple(self.issues))
