# SQL Stores

SQL Stores are Heta's general interface to relational databases. They handle connection management, parameterized SQL execution, query results, and transactions.

`SQLStore` is not tied to a Heta business schema. Tables for documents, chunks, graph facts, memory, or application metadata should be defined by the corresponding step, procedure, or application layer.

## Quick Start

```python
from heta_framework.common.stores import SQLStore

store = SQLStore("sqlite:///heta.db")

await store.execute(
    "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
)

await store.execute(
    "INSERT INTO documents (id, text) VALUES (:id, :text)",
    {"id": "doc-001", "text": "hello heta"},
)

row = await store.fetch_one(
    "SELECT id, text FROM documents WHERE id = :id",
    {"id": "doc-001"},
)
```

Always use parameter binding:

```python
await store.fetch_all(
    "SELECT id, text FROM documents WHERE text LIKE :keyword",
    {"keyword": "%heta%"},
)
```

Do not concatenate user input into SQL strings.

## Installation

SQLite:

```bash
pip install "heta[sql]"
```

PostgreSQL:

```bash
pip install "heta[postgres]"
```

MySQL:

```bash
pip install "heta[mysql]"
```

Connection URL examples:

```python
SQLStore("sqlite:///heta.db")
SQLStore("postgresql+psycopg://user:password@host:5432/db")
SQLStore("mysql+pymysql://user:password@host:3306/db")
```

## Core Objects

| Object | Meaning |
| --- | --- |
| `SQLStoreProtocol` | SQL capability protocol used by recipes, memory, custom stores, and steps. |
| `SQLStore` | SQLAlchemy-based implementation. |
| `SQLTransaction` | SQL executor within a transaction. |
| `SQLResult` | Execution result, currently with `rowcount`. |
| `SQLRow` | Query row, typed as `dict[str, Any]`. |

## Methods

```python
await store.execute(statement, parameters)
await store.fetch_one(statement, parameters)
await store.fetch_all(statement, parameters)
```

| Method | Meaning |
| --- | --- |
| `execute` | Execute a statement and auto-commit. |
| `fetch_one` | Return one row, or `None` when not found. |
| `fetch_all` | Return `list[dict[str, Any]]`. |
| `transaction` | Create a transaction context. |
| `aclose` | Close the SQLAlchemy engine. |

## Transactions

```python
async with store.transaction() as tx:
    await tx.execute(
        "INSERT INTO documents (id, text) VALUES (:id, :text)",
        {"id": "doc-001", "text": "hello"},
    )
    await tx.execute(
        "INSERT INTO chunks (id, document_id, text) VALUES (:id, :document_id, :text)",
        {"id": "chunk-001", "document_id": "doc-001", "text": "chunk text"},
    )
```

The transaction commits on normal exit and rolls back on exception.

## Scope

SQL Stores handle SQLAlchemy engines, parameterized SQL, row queries, transactions, and SQLAlchemy-supported databases such as SQLite, PostgreSQL, and MySQL.

They do not handle business schemas, migrations, ORM mappings, authorization, or the `KnowledgeBase` lifecycle.
