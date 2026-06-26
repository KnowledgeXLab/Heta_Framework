"""Runtime metadata keys for knowledge bases."""

from __future__ import annotations

import re
from dataclasses import dataclass

KB_METADATA_PREFIX = "_heta/knowledge_bases"
MAX_KNOWLEDGE_BASE_NAME_LENGTH = 80


class KnowledgeBaseAlreadyExistsError(RuntimeError):
    """Raised when a completed knowledge base already exists."""


class KnowledgeBaseNotFoundError(RuntimeError):
    """Raised when persisted knowledge base metadata cannot be found."""


class KnowledgeBaseNotReadyError(RuntimeError):
    """Raised when a persisted knowledge base is not ready for loading."""


@dataclass(frozen=True)
class KnowledgeBaseRuntime:
    """ObjectStore key layout for one knowledge base runtime."""

    name: str

    def __post_init__(self) -> None:
        _validate_knowledge_base_name(self.name)

    @property
    def safe_name(self) -> str:
        """Return the ObjectStore-safe name segment."""
        return safe_knowledge_base_name(self.name)

    @property
    def prefix(self) -> str:
        """Return the runtime metadata prefix for this knowledge base."""
        return f"{KB_METADATA_PREFIX}/{self.safe_name}"

    @property
    def manifest_key(self) -> str:
        """Return the manifest object key."""
        return f"{self.prefix}/manifest.json"

    @property
    def latest_run_key(self) -> str:
        """Return the latest run pointer object key."""
        return f"{self.prefix}/latest_run.json"

    def run_prefix(self, run_id: str) -> str:
        """Return the runtime prefix for one run."""
        if run_id.strip() == "":
            raise ValueError("run_id must not be empty")
        return f"{self.prefix}/runs/{run_id}"

    def state_key(self, run_id: str) -> str:
        """Return the run state object key."""
        return f"{self.run_prefix(run_id)}/state.json"

    def record_key(self, run_id: str) -> str:
        """Return the final run record object key."""
        return f"{self.run_prefix(run_id)}/record.json"


def safe_knowledge_base_name(name: str) -> str:
    """Return a stable, ObjectStore-safe path segment for a KB name."""
    _validate_knowledge_base_name(name)
    normalized = re.sub(r"[^\w]+", "_", name.strip().lower(), flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if normalized == "":
        raise ValueError("knowledge base name must contain letters or numbers")
    return normalized


def _validate_knowledge_base_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("knowledge base name must be a string")
    value = name.strip()
    if value == "":
        raise ValueError("knowledge base name must not be empty")
    if len(value) > MAX_KNOWLEDGE_BASE_NAME_LENGTH:
        raise ValueError(
            f"knowledge base name must be at most {MAX_KNOWLEDGE_BASE_NAME_LENGTH} characters"
        )
    allowed = {" ", "_", "-"}
    if any(not (char.isalnum() or char in allowed) for char in value):
        raise ValueError(
            "knowledge base name may contain only letters, numbers, spaces, underscores, or hyphens"
        )


__all__ = [
    "KB_METADATA_PREFIX",
    "KnowledgeBaseAlreadyExistsError",
    "KnowledgeBaseNotFoundError",
    "KnowledgeBaseNotReadyError",
    "KnowledgeBaseRuntime",
    "safe_knowledge_base_name",
]
