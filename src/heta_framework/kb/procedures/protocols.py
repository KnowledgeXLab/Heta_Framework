"""Procedure protocols for knowledge recipes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heta_framework.kb.steps import KnowledgeStepProtocol


@runtime_checkable
class KnowledgeProcedureProtocol(Protocol):
    """Reusable static composition of knowledge build steps."""

    @property
    def name(self) -> str:
        """Stable procedure name used in recipe summaries."""
        ...

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Expand this procedure into executable recipe steps."""
        ...
