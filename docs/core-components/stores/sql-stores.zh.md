# SQL Stores

SQL Stores 是 Heta 与关系型数据库交互的通用入口。它负责连接管理、参数化 SQL 执行、查询返回和事务。

SQLStore 不绑定某一套 Heta 业务表。Document、chunk、graph facts、memory 或业务 metadata 的表结构，都应该由对应 step、procedure 或上层应用决定。

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

SQLStore 支持自由 SQL，但应该始终使用参数绑定：

```python
await store.fetch_all(
    "SELECT id, text FROM documents WHERE text LIKE :keyword",
    {"keyword": "%heta%"},
)
```

不要把用户输入拼接进 SQL 字符串。

## Installation

SQLite：

```bash
pip install "heta-framework[sql]"
```

PostgreSQL：

```bash
pip install "heta-framework[postgres]"
```

MySQL：

```bash
pip install "heta-framework[mysql]"
```

连接 URL 示例：

```python
SQLStore("sqlite:///heta.db")
SQLStore("postgresql+psycopg://user:password@host:5432/db")
SQLStore("mysql+pymysql://user:password@host:3306/db")
```

## Core Objects

| 对象 | 说明 |
| --- | --- |
| `SQLStoreProtocol` | SQL 存储能力协议，用于 Recipe、memory、自定义 store 和 steps。 |
| `SQLStore` | 基于 SQLAlchemy engine 的实现。 |
| `SQLTransaction` | 一个事务中的 SQL 执行器。 |
| `SQLResult` | SQL 执行结果，当前包含 `rowcount`。 |
| `SQLRow` | 查询返回行，类型为 `dict[str, Any]`。 |

## Methods

```python
await store.execute(statement, parameters)
await store.fetch_one(statement, parameters)
await store.fetch_all(statement, parameters)
```

| 方法 | 说明 |
| --- | --- |
| `execute` | 执行一条 SQL，并自动提交。 |
| `fetch_one` | 查询一行，查不到返回 `None`。 |
| `fetch_all` | 查询多行，返回 `list[dict[str, Any]]`。 |
| `transaction` | 创建事务上下文。 |
| `aclose` | 关闭 SQLAlchemy engine。 |

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

事务上下文正常退出会提交；如果抛出异常，SQLAlchemy 会回滚。

## Scope

SQL Stores 负责：

- SQLAlchemy engine 管理。
- 参数化 SQL 执行。
- 查询一行或多行。
- 事务上下文。
- SQLite、PostgreSQL、MySQL 等 SQLAlchemy 支持的数据库接入。

SQL Stores 不负责业务 schema、表迁移、ORM 映射、权限系统或 `KnowledgeBase` 生命周期管理。业务表结构应由 Recipe、memory、step 或更高层模块决定。
