# SQL Text Search

`sql_text_search` retrieves chunk text written by `PersistChunks` into `SQLStore`.

It is SQL text retrieval, not a dedicated full-text search index. For BM25, phrase query, analyzers, or Elasticsearch/OpenSearch-style indexing, use `IndexFullText` and `full_text_search`.

## Required Asset

`PersistChunks` declares:

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

When this asset exists in the latest run record, the default query registry enables:

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

Each `QueryResult` represents one chunk and includes:

```text
id
text
score
source
metadata
```

`source` includes:

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

`generic` uses:

```text
LOWER(content_text) LIKE LOWER(:pattern)
```

It works across SQLite, MySQL, and minimal SQL environments. Recall is conservative but portable.

`postgresql` uses:

```text
content_tsv @@ plainto_tsquery('simple', :query)
ts_rank(content_tsv, plainto_tsquery('simple', :query))
```

`PersistChunks(dialect="postgresql")` creates `content_tsv` and a GIN index.

## Scope

`sql_text_search` only performs SQL text recall. It does not embed, rerank, rewrite queries, fuse hybrid results, or generate answers.

`sql_text_search` and `full_text_search` are parallel capabilities. They do not depend on each other and do not automatically fall back to each other.
