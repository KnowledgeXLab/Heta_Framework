"""Small, typed handoffs; task contents remain ordinary Harbor files."""

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskRequest(Record):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    package_uri: str
    variant: int = Field(ge=0)


class BuildResult(Record):
    schema_version: str = "1.0"
    task_id: str
    candidate_revision: str
    status: Literal["self_test_passed", "validation_pending", "build_failed", "validation_failed"]
    build_report_path: str = "build-report.md"
    validation_path: str = "validation.json"
    issues: list[dict] = Field(default_factory=list)
    execution_mode: Literal["mock", "harbor"]
    simulated_status: Literal["self_test_passed", "validation_failed"] | None = None


class ReviewResult(Record):
    schema_version: str = "1.0"
    task_id: str
    candidate_revision: str
    decision: Literal["pass", "revise", "needs_materials", "validation_pending"]
    issues: list[dict] = Field(default_factory=list)
    checked_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    difficulty_calibrated: bool = False
    execution_mode: Literal["mock", "harbor"]
    simulated_decision: Literal["pass", "revise"] | None = None


class TaskResult(Record):
    task_id: str
    status: Literal[
        "mock_completed",
        "accepted",
        "revision_limit",
        "validation_pending",
        "needs_materials",
        "failed",
    ]
    execution_mode: Literal["mock", "harbor"]
    validated: bool = False
    difficulty_calibrated: bool = False
    task_package_uri: str | None = None
    candidate_revision: str | None = None
    revision: int = 0
    report_uri: str
    error: str | None = None
    skill_status: Literal[
        "disabled", "pending", "accepted", "mock_completed", "not_applicable", "rejected", "failed"
    ] = "disabled"
    skill_package_uri: str | None = None
    skill_report_uri: str | None = None
    skill_error: str | None = None


class RoleBackend(Protocol):
    """A backend writes role outputs; orchestration owns revisions and acceptance."""

    mode: Literal["mock", "harbor"]

    def author(
        self,
        request: TaskRequest,
        source: Path,
        output: Path,
        revision: int,
        previous: Path | None,
        feedback: dict | None,
    ) -> None: ...

    def build(
        self, request: TaskRequest, candidate: Path, output: Path, candidate_revision: str
    ) -> BuildResult: ...

    def review(
        self,
        request: TaskRequest,
        source: Path,
        candidate: Path,
        build_output: Path,
        output: Path,
        candidate_revision: str,
        revision: int,
    ) -> ReviewResult: ...

    def extract_skill(
        self, request: TaskRequest, inputs: Path, output: Path, revision: str
    ) -> None: ...

    def review_skill(
        self,
        request: TaskRequest,
        inputs: Path,
        draft: Path,
        output: Path,
        revision: str,
        skill_revision: str,
    ) -> None: ...
