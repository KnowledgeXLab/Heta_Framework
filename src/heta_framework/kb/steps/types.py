"""Shared data types for knowledge base build steps."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from heta_framework.kb.search.assets import SearchAsset

ComponentNamespace = Literal["models", "stores", "parsers"]
IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True, order=True)
class ComponentRef:
    """Reference to a recipe component required by a build step."""

    namespace: ComponentNamespace
    kind: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind.strip() == "":
            raise ValueError("kind must not be empty")
        if self.name is not None and self.name.strip() == "":
            raise ValueError("name must not be empty")

    @property
    def key(self) -> str:
        """Return a stable printable key for diagnostics and recipe summaries."""
        if self.name is None:
            return f"{self.namespace}.{self.kind}"
        return f"{self.namespace}.{self.kind}.{self.name}"


@dataclass(frozen=True)
class StepRequirements:
    """Capabilities and artifacts required before a step can run."""

    components: frozenset[ComponentRef] = field(default_factory=frozenset)
    artifacts: frozenset[str] = field(default_factory=frozenset)
    queries: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", frozenset(self.components))
        object.__setattr__(
            self,
            "artifacts",
            _normalize_names(self.artifacts, field_name="artifacts"),
        )
        object.__setattr__(self, "queries", _normalize_names(self.queries, field_name="queries"))


@dataclass(frozen=True)
class StepCapabilities:
    """Artifacts and query modes provided after a step completes."""

    artifacts: frozenset[str] = field(default_factory=frozenset)
    queries: frozenset[str] = field(default_factory=frozenset)
    search_assets: tuple[SearchAsset, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifacts",
            _normalize_names(self.artifacts, field_name="artifacts"),
        )
        object.__setattr__(self, "queries", _normalize_names(self.queries, field_name="queries"))
        object.__setattr__(self, "search_assets", tuple(self.search_assets))


@dataclass(frozen=True)
class IssueSubject:
    """Object affected by a non-fatal step issue."""

    type: str
    id: str

    def __post_init__(self) -> None:
        if self.type.strip() == "":
            raise ValueError("type must not be empty")
        if self.id.strip() == "":
            raise ValueError("id must not be empty")


@dataclass(frozen=True)
class IssueResolution:
    """Action taken by the framework after a step issue occurs."""

    action: str
    outcome: str

    def __post_init__(self) -> None:
        if self.action.strip() == "":
            raise ValueError("action must not be empty")
        if self.outcome.strip() == "":
            raise ValueError("outcome must not be empty")


@dataclass(frozen=True)
class StepIssue:
    """Non-fatal diagnostic issue reported by a knowledge build step."""

    step: str
    subject: IssueSubject
    message: str
    code: str
    severity: IssueSeverity = "warning"
    resolution: IssueResolution | None = None
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step.strip() == "":
            raise ValueError("step must not be empty")
        if self.message.strip() == "":
            raise ValueError("message must not be empty")
        if self.code.strip() == "":
            raise ValueError("code must not be empty")
        normalized_details = {
            str(key).strip(): str(value).strip()
            for key, value in self.details.items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "details", normalized_details)


def model_ref(kind: str, name: str | None = None) -> ComponentRef:
    """Reference a model component from KnowledgeModels."""
    return ComponentRef(namespace="models", kind=kind, name=name)


def store_ref(kind: str, name: str | None = None) -> ComponentRef:
    """Reference a store component from KnowledgeStores."""
    return ComponentRef(namespace="stores", kind=kind, name=name)


def parser_ref(name: str | None = None) -> ComponentRef:
    """Reference the parser registry or a named parser registry."""
    return ComponentRef(namespace="parsers", kind="documents", name=name)


def _normalize_names(names: Iterable[str], *, field_name: str) -> frozenset[str]:
    normalized = frozenset(name.strip() for name in names)
    if any(name == "" for name in normalized):
        raise ValueError(f"{field_name} must not contain empty values")
    return normalized
