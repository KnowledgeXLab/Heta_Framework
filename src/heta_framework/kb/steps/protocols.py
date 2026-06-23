"""Step protocols for knowledge base recipes."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from heta_framework.kb.steps.types import StepCapabilities, StepRequirements


class StepContextProtocol(Protocol):
    """Execution context provided to a knowledge build step."""

    def get_component(self, key: str) -> Any:
        """Return a recipe component by its stable key."""
        ...

    def get_artifact(self, key: str) -> Any:
        """Return an artifact produced by an earlier step."""
        ...

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact produced by this step."""
        ...


@runtime_checkable
class KnowledgeStepProtocol(Protocol):
    """Capability protocol for one recipe build step."""

    @property
    def name(self) -> str:
        """Stable step name used in traces and recipe summaries."""
        ...

    @property
    def requirements(self) -> StepRequirements:
        """Components, artifacts, and query modes required by this step."""
        ...

    @property
    def capabilities(self) -> StepCapabilities:
        """Artifacts and query modes provided after this step completes."""
        ...

    async def run(self, context: StepContextProtocol) -> None:
        """Execute this step against a recipe context."""
        ...
