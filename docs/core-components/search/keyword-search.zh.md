# SQL Text Search

`sql_text_search` 检索 `PersistChunks` 写入 SQLStore 的 chunk 文本。

它是 SQL 文本检索能力，不等同于专业全文检索索引。需要 BM25、phrase query、analyzer 或 Elasticsearch/OpenSearch 这类能力时，应使用 `IndexFullText` 解锁 `full_text_search`。

## Required Asset

`PersistChunks` 会声明：

```python
SearchAsset(
    kind="chunk_text_index",
    name="chunks",
    store="stores.sql",
    metadata={
        "table": "chunks",
        "dialect": "generic",
        "text_field": "content_text",
    },
)
```

只要 KB 的 latest run record 中存在这个资产，默认 query registry 就会启用：

```text
sql_text_search
```

## Usage

```python
response = await kb.query(
    "flight control fault",
    mode="sql_text_search",
    top_k=5,
)

for result in response.results:
    print(result.score, result.text)
```

返回结果仍然是统一的 `QueryResponse`。每条 `QueryResult` 表示一个 chunk，并包含：

```text
id
text
score
source
metadata
```

`source` 中会包含：

```text
document_id
source_key
source_chunk
page_index
chunk_index
token_start
token_end
```

## SQL Strategy

`generic` 使用：

```text
LOWER(content_text) LIKE LOWER(:pattern)
```

它适合 SQLite、MySQL 和最小部署环境，召回能力保守但通用。

`postgresql` 使用：

```text
content_tsv @@ plainto_tsquery('simple', :query)
ts_rank(content_tsv, plainto_tsquery('simple', :query))
```

`PersistChunks(dialect="postgresql")` 会创建 `content_tsv` 和 GIN index。

## Boundary

`sql_text_search` 只负责 SQL 文本召回，不做 embedding、rerank、query rewrite 或 hybrid fusion。

`sql_text_search` 和 `full_text_search` 是并列能力。前者来自 SQL chunk table，后者来自 full-text index。它们不互相依赖，也不自动 fallback。
