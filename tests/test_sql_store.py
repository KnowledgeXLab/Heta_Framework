import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import SQLStore, SQLStoreProtocol


def test_sql_store_satisfies_protocol():
    store = SQLStore("sqlite:///:memory:")

    async def close():
        await store.aclose()

    try:
        assert isinstance(store, SQLStoreProtocol)
    finally:
        asyncio.run(close())


def test_sql_store_executes_parameterized_sql():
    async def run():
        store = SQLStore("sqlite:///:memory:")
        try:
            await store.execute(
                "CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            result = await store.execute(
                "INSERT INTO documents (id, text) VALUES (:id, :text)",
                {"id": "doc-001", "text": "hello heta"},
            )
            row = await store.fetch_one(
                "SELECT id, text FROM documents WHERE id = :id",
                {"id": "doc-001"},
            )
            rows = await store.fetch_all(
                "SELECT id, text FROM documents WHERE text LIKE :keyword",
                {"keyword": "%heta%"},
            )
            return result.rowcount, row, rows
        finally:
            await store.aclose()

    rowcount, row, rows = asyncio.run(run())

    assert rowcount == 1
    assert row == {"id": "doc-001", "text": "hello heta"}
    assert rows == [{"id": "doc-001", "text": "hello heta"}]


def test_sql_store_transaction_commits():
    async def run():
        store = SQLStore("sqlite:///:memory:")
        try:
            await store.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT)")
            async with store.transaction() as tx:
                await tx.execute(
                    "INSERT INTO chunks (id, text) VALUES (:id, :text)",
                    {"id": "chunk-001", "text": "first"},
                )
                await tx.execute(
                    "INSERT INTO chunks (id, text) VALUES (:id, :text)",
                    {"id": "chunk-002", "text": "second"},
                )
            return await store.fetch_all("SELECT id, text FROM chunks ORDER BY id")
        finally:
            await store.aclose()

    rows = asyncio.run(run())

    assert rows == [
        {"id": "chunk-001", "text": "first"},
        {"id": "chunk-002", "text": "second"},
    ]


def test_sql_store_transaction_rolls_back():
    async def run():
        store = SQLStore("sqlite:///:memory:")
        try:
            await store.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT)")
            with pytest.raises(RuntimeError):
                async with store.transaction() as tx:
                    await tx.execute(
                        "INSERT INTO chunks (id, text) VALUES (:id, :text)",
                        {"id": "chunk-001", "text": "first"},
                    )
                    raise RuntimeError("rollback")
            return await store.fetch_all("SELECT id, text FROM chunks")
        finally:
            await store.aclose()

    rows = asyncio.run(run())

    assert rows == []


def test_sql_store_rejects_empty_statement():
    async def run():
        store = SQLStore("sqlite:///:memory:")
        try:
            await store.execute("   ")
        finally:
            await store.aclose()

    with pytest.raises(ValueError, match="statement must not be empty"):
        asyncio.run(run())
