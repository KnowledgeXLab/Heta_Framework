"""SQLAlchemy-backed SQL store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from heta_framework.common.stores.sql.types import SQLParameters, SQLResult, SQLRow


@dataclass(frozen=True)
class SQLStoreConfig:
    """Configuration for SQLAlchemy-backed SQL stores."""

    url: str
    echo: bool = False
    pool_pre_ping: bool = True

    def __post_init__(self) -> None:
        if self.url.strip() == "":
            raise ValueError("url must not be empty")


class SQLStore:
    """SQLAlchemy-backed SQL store for arbitrary parameterized SQL."""

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_pre_ping: bool = True,
        engine: Any | None = None,
    ) -> None:
        self.config = SQLStoreConfig(url=url, echo=echo, pool_pre_ping=pool_pre_ping)
        self._engine = engine if engine is not None else _create_engine(self.config)

    async def execute(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLResult:
        """Execute one SQL statement and commit it."""
        _validate_statement(statement)
        with self._engine.begin() as connection:
            result = connection.execute(_sql_text(statement), parameters or {})
        return SQLResult(rowcount=int(result.rowcount or 0))

    async def fetch_one(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLRow | None:
        """Fetch one row."""
        _validate_statement(statement)
        with self._engine.connect() as connection:
            result = connection.execute(_sql_text(statement), parameters or {})
            row = result.mappings().first()
        return dict(row) if row is not None else None

    async def fetch_all(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> list[SQLRow]:
        """Fetch all rows."""
        _validate_statement(statement)
        with self._engine.connect() as connection:
            result = connection.execute(_sql_text(statement), parameters or {})
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["SQLTransaction"]:
        """Open a transaction context."""
        with self._engine.begin() as connection:
            yield SQLTransaction(connection)

    async def aclose(self) -> None:
        """Release resources held by the store."""
        self._engine.dispose()


class SQLTransaction:
    """SQL executor bound to one active transaction."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def execute(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLResult:
        """Execute one SQL statement inside this transaction."""
        _validate_statement(statement)
        result = self._connection.execute(_sql_text(statement), parameters or {})
        return SQLResult(rowcount=int(result.rowcount or 0))

    async def fetch_one(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> SQLRow | None:
        """Fetch one row inside this transaction."""
        _validate_statement(statement)
        result = self._connection.execute(_sql_text(statement), parameters or {})
        row = result.mappings().first()
        return dict(row) if row is not None else None

    async def fetch_all(
        self,
        statement: str,
        parameters: SQLParameters | None = None,
    ) -> list[SQLRow]:
        """Fetch all rows inside this transaction."""
        _validate_statement(statement)
        result = self._connection.execute(_sql_text(statement), parameters or {})
        return [dict(row) for row in result.mappings().all()]


def _create_engine(config: SQLStoreConfig) -> Any:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError("SQLAlchemy is not installed; install the `heta[sql]` extra") from exc

    return create_engine(
        config.url,
        echo=config.echo,
        pool_pre_ping=config.pool_pre_ping,
    )


def _sql_text(statement: str) -> Any:
    try:
        from sqlalchemy import text
    except ImportError as exc:
        raise ImportError("SQLAlchemy is not installed; install the `heta[sql]` extra") from exc
    return text(statement)


def _validate_statement(statement: str) -> None:
    if statement.strip() == "":
        raise ValueError("statement must not be empty")
