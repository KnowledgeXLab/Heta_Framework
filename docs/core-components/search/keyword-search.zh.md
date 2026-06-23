# Keyword Search

`keyword_search` 检索 `PersistChunks` 写入 SQLStore 的 chunk 文本。

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
keyword_search
```

## Usage

```python
response = await kb.query(
    "flight control fault",
    mode="keyword_search",
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

`keyword_search` 只负责 SQL 文本召回，不做 embedding、rerank、query rewrite 或 hybrid fusion。
如果需要向量加关键词融合，应通过显式的 hybrid query engine 或 procedure 表达。
