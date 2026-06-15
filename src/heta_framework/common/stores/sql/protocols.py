"""SQL store capability protocols."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from heta_framework.common.stores.sql.types import SQLParameters, SQLResult, SQLRow


@runtime_checkable
class SQLExecutorProtocol(Protocol):
    """Capability protocol for executing SQL statements."""

    async def execute(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLResult:
        """Execute one SQL statement."""
        ...

    async def fetch_one(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLRow | None:
        """Fetch one row."""
        ...

    async def fetch_all(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> list[SQLRow]:
        """Fetch all rows."""
        ...


@runtime_checkable
class SQLTransactionProtocol(SQLExecutorProtocol, Protocol):
    """Capability protocol for SQL transactions."""


@runtime_checkable
class SQLStoreProtocol(SQLExecutorProtocol, Protocol):
    """Capability protocol for SQL stores."""

    def transaction(self) -> AbstractAsyncContextManager[SQLTransactionProtocol]:
        """Open a transaction context."""
        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""
        ...
