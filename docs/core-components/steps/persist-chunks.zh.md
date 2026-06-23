# Persist Chunks

`PersistChunks` 将 `ParsedChunk` JSON 写入 SQLStore。它主要服务关键词检索、证据查询、图谱溯源和后续结构化分析。

```text
ParsedChunk JSON -> SQL table
```

它默认消费 `rechunked_chunk_keys`，对齐 HetaDB 中“rechunk 后写 PostgreSQL chunk table”的阶段。

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

默认输出：

```text
persist_chunks_result
```

完成后会声明：

```text
SearchAsset(kind="chunk_text_index")
query mode: keyword_search
```

因此知识库可以通过 `KnowledgeBase.query(..., mode="keyword_search")` 进行关键词检索。

## Configuration

```python
PersistChunksConfig(
    table_names=ChunkTableNames(chunks="chunks"),
    dialect="generic",
    chunk_keys_artifact="rechunked_chunk_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `table_names.chunks` | SQL chunk 表名。必须是简单 SQL identifier。 |
| `dialect` | `generic` 或 `postgresql`。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |

`generic` 使用保守的 SQL 表结构，适合 SQLite、MySQL 和基础 SQL smoke test。内置 `keyword_search` 会使用 `LIKE` 作为兜底策略。

`postgresql` 会额外创建：

```text
content_tsv tsvector
GIN index
```

用于 PostgreSQL 全文检索召回。内置 `keyword_search` 会使用 `plainto_tsquery('simple', query)` 和 `ts_rank` 排序。

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

这保留了 HetaDB 的核心溯源语义。

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
