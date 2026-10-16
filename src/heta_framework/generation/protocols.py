"""Contracts for complete task-production implementations."""

from typing import Protocol, TypeVar, runtime_checkable

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.generation.types import TaskRequest, TaskRequirements, TaskResult

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


@runtime_checkable
class TaskProducerProtocol(Protocol[ConfigT_co]):
    """Produce Harbor tasks from declared inputs using an object store.

    Implementations own configuration, input preparation, and native execution.
    Requirements describe inputs; TaskResult identifies published outputs.
    """

    @property
    def name(self) -> str:
        """Stable producer name used in reports."""
        ...

    @property
    def config(self) -> ConfigT_co:
        """Typed configuration owned by this producer, exposed read-only."""
        ...

    @property
    def requirements(self) -> TaskRequirements:
        """Inputs required by this configured producer."""
        ...

    async def produce(
        self,
        request: TaskRequest,
        *,
        object_store: ObjectStoreProtocol,
    ) -> TaskResult:
        """Read inputs and publish a Harbor task manifest and production report.

        Validate the request against requirements before native execution.
        Return readable object keys, including an empty manifest for zero tasks.
        Infrastructure errors that prevent publication propagate to the caller.
        """
        ...
