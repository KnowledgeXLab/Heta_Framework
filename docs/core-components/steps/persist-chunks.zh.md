# Persist Chunks

`PersistChunks` 将 `ParsedChunk` JSON 写入 `SQLStore`。

```text
ParsedChunk JSON -> SQL chunk table
```

它主要服务三件事：

- SQL 文本检索
- 证据查询和 chunk 回溯
- Heta-style graph 构建中的证据表关联

默认情况下，它消费 `rechunked_chunk_keys`，对齐 HetaDB 中“rechunk 后写 PostgreSQL chunk table”的阶段。也可以配置为直接消费原始 `chunk_keys`。

## Contract

`PersistChunks` 使用两个 recipe components：

```text
stores.objects
stores.sql
```

默认读取：

```text
rechunked_chunk_keys
```

执行语义：

```text
read chunk keys
  -> load ParsedChunk JSON from ObjectStore
  -> create SQL chunk table if needed
  -> delete existing row by chunk_id
  -> insert current row
  -> expose persist_chunks_result artifact
  -> enable sql_text_search query mode
```

按 `chunk_id` 先删后插，使同输入重跑时不会重复写入旧行。

## Configuration

```python
PersistChunksConfig(
    table_names=ChunkTableNames(chunks="chunks"),
    dialect="generic",
    object_store=None,
    sql_store=None,
    chunk_keys_artifact="rechunked_chunk_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `table_names.chunks` | SQL chunk 表名。必须是简单 SQL identifier。 |
| `dialect` | `generic` 或 `postgresql`。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `sql_store` | 命名 SQLStore。默认引用 `stores.sql`。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |

`generic` 使用保守 SQL 表结构，适合 SQLite、MySQL 和基础 SQL smoke test。内置 `sql_text_search` 会使用 `LIKE` 作为兜底策略。

`postgresql` 会额外创建：

```text
content_tsv tsvector
GIN index
```

内置 `sql_text_search` 会使用 `plainto_tsquery('simple', query)` 和 `ts_rank` 排序。

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        store_ref("sql"),
    }),
    artifacts=frozenset({
        "rechunked_chunk_keys",
    }),
)
```

如果改为持久化原始 chunks：

```python
PersistChunksConfig(
    chunk_keys_artifact="chunk_keys",
)
```

requirements 也会对应变成需要 `chunk_keys`。

## Capabilities

`PersistChunks` 提供：

```python
StepCapabilities(
    artifacts=frozenset({
        "persist_chunks_result",
    }),
    queries=frozenset({
        "sql_text_search",
    }),
)
```

同时声明一个 search asset：

```text
SearchAsset(kind="chunk_text_index")
```

这表示当前 KB 已经具备 SQL chunk text table，可用于 `sql_text_search` 和后续证据查询。

## Table Shape

通用字段：

```text
chunk_id
document_id
content_text
source_id
source_chunk
metadata_json
created_at
```

其中：

```text
source_chunk = parent_chunk_ids 或自身 chunk_id
```

这保留了 HetaDB 的核心溯源语义：即使当前写入的是 rechunked chunk，也能追溯到原始 chunk。

## Usage

```python
from heta_framework.common.stores import SQLStore

sql_store = SQLStore(
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
)

steps=[
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    MergeChunks(),
    RechunkDocuments(),
    PersistChunks(
        PersistChunksConfig(
            table_names=ChunkTableNames(chunks="papers_chunks"),
            dialect="postgresql",
        )
    ),
]
```

如果只想持久化原始 chunks：

```python
PersistChunks(
    PersistChunksConfig(
        table_names=ChunkTableNames(chunks="raw_chunks"),
        chunk_keys_artifact="chunk_keys",
    )
)
```
