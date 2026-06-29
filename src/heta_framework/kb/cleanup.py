"""Cleanup plans for knowledge base lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

CleanupTargetKind = Literal[
    "object_key",
    "runtime_prefix",
    "sql_table",
    "text_index",
    "vector_collection",
]


@dataclass(frozen=True)
class CleanupTarget:
    """One persistent resource that can be removed by KnowledgeBase.delete."""

    kind: CleanupTargetKind
    value: str
    component: str | None = None

    def __post_init__(self) -> None:
        if self.value.strip() == "":
            raise ValueError("value must not be empty")
        if self.component is not None and self.component.strip() == "":
            raise ValueError("component must not be empty")


@dataclass(frozen=True)
class StepCleanupPlan:
    """Resources produced by one step that should be removed with the KB."""

    targets: tuple[CleanupTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class KnowledgeBaseDeletePlan:
    """Complete cleanup plan for one knowledge base."""

    targets: tuple[CleanupTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _dedupe_targets(self.targets))

    @property
    def object_keys(self) -> tuple[str, ...]:
        """Return object keys in this plan."""
        return tuple(target.value for target in self.targets if target.kind == "object_key")

    @property
    def runtime_prefixes(self) -> tuple[str, ...]:
        """Return runtime metadata prefixes in this plan."""
        return tuple(target.value for target in self.targets if target.kind == "runtime_prefix")

    @property
    def sql_tables(self) -> tuple[str, ...]:
        """Return SQL tables in this plan."""
        return tuple(target.value for target in self.targets if target.kind == "sql_table")

    @property
    def vector_collections(self) -> tuple[str, ...]:
        """Return vector collections in this plan."""
        return tuple(
            target.value for target in self.targets if target.kind == "vector_collection"
        )

    @property
    def text_indexes(self) -> tuple[str, ...]:
        """Return full-text indexes in this plan."""
        return tuple(target.value for target in self.targets if target.kind == "text_index")


@dataclass(frozen=True)
class CleanupIssue:
    """Non-fatal issue encountered while deleting one cleanup target."""

    target: CleanupTarget
    message: str
    error_type: str

    def __post_init__(self) -> None:
        if self.message.strip() == "":
            raise ValueError("message must not be empty")
        if self.error_type.strip() == "":
            raise ValueError("error_type must not be empty")


@dataclass(frozen=True)
class KnowledgeBaseDeleteResult:
    """Result returned by KnowledgeBase.delete."""

    dry_run: bool
    targets: tuple[CleanupTarget, ...]
    deleted_object_keys: tuple[str, ...] = ()
    deleted_runtime_prefixes: tuple[str, ...] = ()
    dropped_sql_tables: tuple[str, ...] = ()
    dropped_text_indexes: tuple[str, ...] = ()
    dropped_vector_collections: tuple[str, ...] = ()
    issues: tuple[CleanupIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "deleted_object_keys", tuple(self.deleted_object_keys))
        object.__setattr__(
            self,
            "deleted_runtime_prefixes",
            tuple(self.deleted_runtime_prefixes),
        )
        object.__setattr__(self, "dropped_sql_tables", tuple(self.dropped_sql_tables))
        object.__setattr__(self, "dropped_text_indexes", tuple(self.dropped_text_indexes))
        object.__setattr__(
            self,
            "dropped_vector_collections",
            tuple(self.dropped_vector_collections),
        )
        object.__setattr__(self, "issues", tuple(self.issues))


def object_key_targets(
    artifacts: Mapping[str, Any],
    artifact_name: str,
    *,
    component: str,
) -> tuple[CleanupTarget, ...]:
    """Return object-key cleanup targets from an artifact containing object keys."""
    return tuple(
        CleanupTarget(kind="object_key", value=key, component=component)
        for key in _artifact_strings(artifacts, artifact_name)
    )


def _artifact_strings(artifacts: Mapping[str, Any], artifact_name: str) -> tuple[str, ...]:
    value = artifacts.get(artifact_name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, frozenset, set)):
        return tuple(item for item in value if isinstance(item, str) and item.strip())
    return ()


def _dedupe_targets(targets: tuple[CleanupTarget, ...]) -> tuple[CleanupTarget, ...]:
    seen: set[CleanupTarget] = set()
    deduped: list[CleanupTarget] = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        deduped.append(target)
    return tuple(deduped)


__all__ = [
    "CleanupIssue",
    "CleanupTarget",
    "CleanupTargetKind",
    "KnowledgeBaseDeletePlan",
    "KnowledgeBaseDeleteResult",
    "StepCleanupPlan",
    "object_key_targets",
]
