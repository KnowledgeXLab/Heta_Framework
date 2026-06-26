"""Run state and records for knowledge base builds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps import StepCapabilities, StepIssue, StepRequirements
from heta_framework.kb.steps.types import ComponentRef, IssueResolution, IssueSubject

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


class RecipeRunState:
    """Mutable state for one recipe run, optionally persisted to an ObjectStore."""

    def __init__(
        self,
        *,
        run_id: str,
        started_at: str,
        status: RecipeRunStatus = "running",
        finished_at: str | None = None,
        step_records: tuple[StepRunRecord, ...] = (),
        current_step: StepRunRecord | None = None,
        artifacts: Mapping[str, Any] | None = None,
        issues: tuple[StepIssue, ...] = (),
        object_store: ObjectStoreProtocol | None = None,
        state_key: str | None = None,
    ) -> None:
        if run_id.strip() == "":
            raise ValueError("run_id must not be empty")
        if started_at.strip() == "":
            raise ValueError("started_at must not be empty")
        self.run_id = run_id
        self.status = status
        self.started_at = started_at
        self.finished_at = finished_at
        self.step_records = list(step_records)
        self.current_step = current_step
        self.artifacts = dict(artifacts or {})
        self.issues = list(issues)
        self.object_store = object_store
        self.state_key = state_key

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        started_at: str,
        object_store: ObjectStoreProtocol | None = None,
        state_key: str | None = None,
    ) -> "RecipeRunState":
        """Create a new running recipe state."""
        return cls(
            run_id=run_id,
            started_at=started_at,
            status="running",
            object_store=object_store,
            state_key=state_key,
        )

    @classmethod
    async def load(
        cls,
        *,
        object_store: ObjectStoreProtocol,
        state_key: str,
    ) -> "RecipeRunState":
        """Load a persisted recipe run state from an ObjectStore."""
        data = json.loads((await object_store.get(state_key)).decode("utf-8"))
        return cls.from_dict(data, object_store=object_store, state_key=state_key)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        object_store: ObjectStoreProtocol | None = None,
        state_key: str | None = None,
    ) -> "RecipeRunState":
        """Create a recipe state from a JSON dictionary."""
        current = data.get("current_step")
        return cls(
            run_id=str(data["run_id"]),
            status=data["status"],
            started_at=str(data["started_at"]),
            finished_at=data.get("finished_at"),
            step_records=tuple(
                _step_record_from_dict(item) for item in data.get("step_records", [])
            ),
            current_step=_step_record_from_dict(current) if isinstance(current, Mapping) else None,
            artifacts=dict(data.get("artifacts", {})),
            issues=tuple(_issue_from_dict(item) for item in data.get("issues", [])),
            object_store=object_store,
            state_key=state_key,
        )

    async def save(self) -> None:
        """Persist the current state if persistence is configured."""
        if self.object_store is None or self.state_key is None:
            return
        await self.object_store.put(
            self.state_key,
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

    async def start_step(
        self,
        record: StepRunRecord,
        *,
        artifacts: Mapping[str, Any],
    ) -> None:
        """Record that a step has started."""
        self.status = "running"
        self.current_step = record
        self.artifacts = _json_safe_artifacts(artifacts)
        await self.save()

    async def skip_step(
        self,
        record: StepRunRecord,
        *,
        artifacts: Mapping[str, Any],
    ) -> None:
        """Record a skipped step unless a previous success already exists."""
        if not self._has_final_record(record):
            self._upsert_step_record(record)
        self.current_step = None
        self.artifacts = _json_safe_artifacts(artifacts)
        await self.save()

    async def finish_step(
        self,
        record: StepRunRecord,
        *,
        artifacts: Mapping[str, Any],
        issues: tuple[StepIssue, ...] = (),
    ) -> None:
        """Record a successful or failed step."""
        self._upsert_step_record(record)
        self.current_step = None
        self.artifacts = _json_safe_artifacts(artifacts)
        self.issues.extend(issues)
        if record.status == "failed":
            self.status = "failed"
        await self.save()

    async def finish_run(
        self,
        *,
        status: RecipeRunStatus,
        finished_at: str,
        artifacts: Mapping[str, Any],
        issues: tuple[StepIssue, ...],
    ) -> None:
        """Finalize the run state."""
        self.status = status
        self.finished_at = finished_at
        self.current_step = None
        self.artifacts = _json_safe_artifacts(artifacts)
        self.issues = list(issues)
        await self.save()

    def to_record(
        self,
        *,
        artifacts: Mapping[str, Any] | None = None,
        capabilities: StepCapabilities | None = None,
        issues: tuple[StepIssue, ...] | None = None,
    ) -> RecipeRunRecord:
        """Return the immutable run record snapshot."""
        return RecipeRunRecord(
            run_id=self.run_id,
            status=self.status,
            started_at=self.started_at,
            finished_at=self.finished_at,
            step_records=tuple(self.step_records),
            artifacts=dict(artifacts if artifacts is not None else self.artifacts),
            capabilities=capabilities or StepCapabilities(),
            issues=tuple(issues if issues is not None else self.issues),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary for persistence."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_step": _step_record_to_dict(self.current_step) if self.current_step else None,
            "step_records": [_step_record_to_dict(record) for record in self.step_records],
            "artifacts": _json_safe_artifacts(self.artifacts),
            "issues": [_issue_to_dict(issue) for issue in self.issues],
        }

    def _has_final_record(self, record: StepRunRecord) -> bool:
        return any(
            existing.index == record.index
            and existing.step_name == record.step_name
            and existing.step_type == record.step_type
            and existing.status in {"succeeded", "failed", "skipped"}
            for existing in self.step_records
        )

    def _upsert_step_record(self, record: StepRunRecord) -> None:
        for index, existing in enumerate(self.step_records):
            if (
                existing.index == record.index
                and existing.step_name == record.step_name
                and existing.step_type == record.step_type
            ):
                self.step_records[index] = record
                return
        self.step_records.append(record)


def recipe_run_record_from_dict(data: Mapping[str, Any]) -> RecipeRunRecord:
    """Create a recipe run record from a JSON dictionary."""
    return RecipeRunRecord(
        run_id=str(data["run_id"]),
        status=data["status"],
        started_at=str(data["started_at"]),
        finished_at=data.get("finished_at"),
        step_records=tuple(
            _step_record_from_dict(item) for item in data.get("step_records", ())
        ),
        artifacts=dict(data.get("artifacts", {})),
        capabilities=_capabilities_from_dict(data.get("capabilities", {})),
        issues=tuple(_issue_from_dict(item) for item in data.get("issues", ())),
    )


def _step_record_to_dict(record: StepRunRecord) -> dict[str, Any]:
    return {
        "index": record.index,
        "step_name": record.step_name,
        "step_type": record.step_type,
        "status": record.status,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "requirements": _requirements_to_dict(record.requirements),
        "capabilities": _capabilities_to_dict(record.capabilities),
        "input_artifacts": list(record.input_artifacts),
        "output_artifacts": list(record.output_artifacts),
        "issues": [_issue_to_dict(issue) for issue in record.issues],
        "error": record.error,
    }


def _step_record_from_dict(data: Mapping[str, Any]) -> StepRunRecord:
    return StepRunRecord(
        index=int(data["index"]),
        step_name=str(data["step_name"]),
        step_type=str(data["step_type"]),
        status=data["status"],
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        requirements=_requirements_from_dict(data.get("requirements", {})),
        capabilities=_capabilities_from_dict(data.get("capabilities", {})),
        input_artifacts=tuple(str(item) for item in data.get("input_artifacts", ())),
        output_artifacts=tuple(str(item) for item in data.get("output_artifacts", ())),
        issues=tuple(_issue_from_dict(item) for item in data.get("issues", ())),
        error=data.get("error"),
    )


def _requirements_to_dict(requirements: StepRequirements) -> dict[str, Any]:
    return {
        "components": [ref.key for ref in sorted(requirements.components)],
        "artifacts": sorted(requirements.artifacts),
        "queries": sorted(requirements.queries),
    }


def _requirements_from_dict(data: Mapping[str, Any]) -> StepRequirements:
    return StepRequirements(
        components=frozenset(
            _component_ref_from_key(str(item)) for item in data.get("components", ())
        ),
        artifacts=frozenset(str(item) for item in data.get("artifacts", ())),
        queries=frozenset(str(item) for item in data.get("queries", ())),
    )


def _capabilities_to_dict(capabilities: StepCapabilities) -> dict[str, Any]:
    return {
        "artifacts": sorted(capabilities.artifacts),
        "queries": sorted(capabilities.queries),
        "search_assets": [asset.to_dict() for asset in capabilities.search_assets],
    }


def _capabilities_from_dict(data: Mapping[str, Any]) -> StepCapabilities:
    return StepCapabilities(
        artifacts=frozenset(str(item) for item in data.get("artifacts", ())),
        queries=frozenset(str(item) for item in data.get("queries", ())),
        search_assets=tuple(SearchAsset(**item) for item in data.get("search_assets", ())),
    )


def _issue_to_dict(issue: StepIssue) -> dict[str, Any]:
    return asdict(issue)


def _issue_from_dict(data: Mapping[str, Any]) -> StepIssue:
    resolution = data.get("resolution")
    return StepIssue(
        step=str(data["step"]),
        subject=IssueSubject(**data["subject"]),
        message=str(data["message"]),
        code=str(data["code"]),
        severity=data.get("severity", "warning"),
        resolution=IssueResolution(**resolution) if isinstance(resolution, Mapping) else None,
        details=dict(data.get("details", {})),
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


def _json_safe_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifacts.items() if _is_json_safe(value)}


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list | tuple):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False
