"""Runtime component containers for knowledge recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from heta_framework.kb.steps import ComponentRef


class MissingComponentError(LookupError):
    """Raised when a recipe component reference cannot be resolved."""


@dataclass(frozen=True)
class KnowledgeModels:
    """Model components available to recipe steps."""

    language: object | None = None
    embedding: object | None = None
    reranker: object | None = None
    named: Mapping[str, object] = field(default_factory=dict)

    def get(self, ref: ComponentRef) -> object:
        """Return a model component by reference."""
        if ref.namespace != "models":
            raise MissingComponentError(f"component is not a model reference: {ref.key}")
        return _lookup_component(
            namespace="models",
            ref=ref,
            defaults={
                "language": self.language,
                "embedding": self.embedding,
                "reranker": self.reranker,
            },
            named=self.named,
        )

    def has(self, ref: ComponentRef) -> bool:
        """Return whether this container can resolve the model reference."""
        try:
            self.get(ref)
        except MissingComponentError:
            return False
        return True


@dataclass(frozen=True)
class KnowledgeStores:
    """Store components available to recipe steps."""

    objects: object | None = None
    vector: object | None = None
    sql: object | None = None
    graph: object | None = None
    named: Mapping[str, object] = field(default_factory=dict)

    def get(self, ref: ComponentRef) -> object:
        """Return a store component by reference."""
        if ref.namespace != "stores":
            raise MissingComponentError(f"component is not a store reference: {ref.key}")
        return _lookup_component(
            namespace="stores",
            ref=ref,
            defaults={
                "objects": self.objects,
                "vector": self.vector,
                "sql": self.sql,
                "graph": self.graph,
            },
            named=self.named,
        )

    def has(self, ref: ComponentRef) -> bool:
        """Return whether this container can resolve the store reference."""
        try:
            self.get(ref)
        except MissingComponentError:
            return False
        return True


@dataclass(frozen=True)
class KnowledgeParsers:
    """Parser components available to recipe steps."""

    documents: object | None = None
    named: Mapping[str, object] = field(default_factory=dict)

    def get(self, ref: ComponentRef) -> object:
        """Return a parser component by reference."""
        if ref.namespace != "parsers":
            raise MissingComponentError(f"component is not a parser reference: {ref.key}")
        return _lookup_component(
            namespace="parsers",
            ref=ref,
            defaults={"documents": self.documents},
            named=self.named,
        )

    def has(self, ref: ComponentRef) -> bool:
        """Return whether this container can resolve the parser reference."""
        try:
            self.get(ref)
        except MissingComponentError:
            return False
        return True


def _lookup_component(
    *,
    namespace: str,
    ref: ComponentRef,
    defaults: Mapping[str, object | None],
    named: Mapping[str, object],
) -> object:
    if ref.name is None:
        component = defaults.get(ref.kind)
        if component is None:
            raise MissingComponentError(f"missing component: {ref.key}")
        return component

    named_key = f"{ref.kind}.{ref.name}"
    if named_key in named:
        return named[named_key]
    if ref.name in named:
        return named[ref.name]
    raise MissingComponentError(f"missing named {namespace} component: {ref.key}")
