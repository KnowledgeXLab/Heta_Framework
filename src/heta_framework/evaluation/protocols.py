"""Protocols for benchmark-driven recipe evaluation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from heta_framework.evaluation.types import (
    BenchmarkCase,
    BenchmarkDocument,
    BenchmarkManifest,
    BenchmarkResource,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
    EvaluationScore,
    PreparedBenchmark,
)
from heta_framework.kb.search import QueryResponse


@runtime_checkable
class BenchmarkEvaluatorProtocol(Protocol):
    """Evaluation method declared by a benchmark."""

    @property
    def name(self) -> str:
        """Return the stable evaluator name used in reports."""
        ...

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore:
        """Score one query response for one benchmark case."""
        ...


@runtime_checkable
class BenchmarkProtocol(Protocol):
    """Protocol implemented by benchmark adapters.

    A benchmark adapter owns data preparation and default scoring policy. It does not
    build a knowledge base or call query engines; BenchmarkRunner will do that later.
    """

    @property
    def manifest(self) -> BenchmarkManifest:
        """Return stable benchmark identity metadata."""
        ...

    def resources(self) -> tuple[BenchmarkResource, ...]:
        """Return external resources needed to prepare this benchmark."""
        ...

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark:
        """Prepare benchmark resources in a local workspace."""
        ...

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]:
        """Yield source documents that should be written to ObjectStore raw keys."""
        ...

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]:
        """Yield query cases used to evaluate the recipe-built knowledge base."""
        ...

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]:
        """Yield independent KB build-and-query units for this benchmark run."""
        ...

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]:
        """Return this benchmark's default evaluation methods."""
        ...


__all__ = [
    "BenchmarkEvaluatorProtocol",
    "BenchmarkProtocol",
]
