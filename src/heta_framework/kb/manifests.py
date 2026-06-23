"""Persistable manifests for knowledge recipes and knowledge bases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from heta_framework.kb.state import RecipeRunRecord
from heta_framework.kb.steps import StepCapabilities, StepRequirements

MANIFEST_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class StepManifest:
    """Persistable summary of one recipe step."""

    index: int
    name: str
    type: str
    requirements: StepRequirements
    capabilities: StepCapabilities

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "type": self.type,
            "requirements": _requirements_to_dict(self.requirements),
            "capabilities": _capabilities_to_dict(self.capabilities),
        }


@dataclass(frozen=True)
class KnowledgeRecipeManifest:
    """Persistable summary of a knowledge recipe."""

    schema_version: str
    steps: tuple[StepManifest, ...]
    component_refs: tuple[str, ...]
    artifacts_required: tuple[str, ...]
    capabilities_provided: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "component_refs", tuple(self.component_refs))
        object.__setattr__(self, "artifacts_required", tuple(self.artifacts_required))
        object.__setattr__(self, "capabilities_provided", tuple(self.capabilities_provided))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "steps": [step.to_dict() for step in self.steps],
            "component_refs": list(self.component_refs),
            "artifacts_required": list(self.artifacts_required),
            "capabilities_provided": list(self.capabilities_provided),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KnowledgeBaseManifest:
    """Persistable summary of a knowledge base."""

    schema_version: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    recipe: KnowledgeRecipeManifest
    run_record: RecipeRunRecord
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("name must not be empty")
        if self.created_at.strip() == "":
            raise ValueError("created_at must not be empty")
        if self.updated_at.strip() == "":
            raise ValueError("updated_at must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "recipe": self.recipe.to_dict(),
            "run_record": run_record_to_dict(self.run_record),
            "metadata": dict(self.metadata),
        }


def run_record_to_dict(record: RecipeRunRecord) -> dict[str, Any]:
    """Return a JSON-safe dictionary for a recipe run record."""
    return {
        "run_id": record.run_id,
        "status": record.status,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "step_records": [
            {
                "index": step.index,
                "step_name": step.step_name,
                "step_type": step.step_type,
                "status": step.status,
                "started_at": step.started_at,
                "finished_at": step.finished_at,
                "requirements": _requirements_to_dict(step.requirements),
                "capabilities": _capabilities_to_dict(step.capabilities),
                "input_artifacts": list(step.input_artifacts),
                "output_artifacts": list(step.output_artifacts),
                "issues": [asdict(issue) for issue in step.issues],
                "error": step.error,
            }
            for step in record.step_records
        ],
        "artifacts": _manifest_artifacts(record.artifacts),
        "capabilities": _capabilities_to_dict(record.capabilities),
        "issues": [asdict(issue) for issue in record.issues],
    }


def _requirements_to_dict(requirements: StepRequirements) -> dict[str, Any]:
    return {
        "components": [ref.key for ref in sorted(requirements.components)],
        "artifacts": sorted(requirements.artifacts),
        "queries": sorted(requirements.queries),
    }


def _capabilities_to_dict(capabilities: StepCapabilities) -> dict[str, Any]:
    return {
        "artifacts": sorted(capabilities.artifacts),
        "queries": sorted(capabilities.queries),
        "search_assets": [asset.to_dict() for asset in capabilities.search_assets],
    }


def _manifest_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in artifacts.items():
        if _is_json_safe(value):
            result[key] = value
        else:
            result[key] = {
                "type": type(value).__name__,
                "manifest_note": "runtime artifact omitted",
            }
    return result


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list | tuple):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False
