# Text Index Stores

Text Index Stores are Heta's full-text search index interface. They handle:

```text
chunk id -> searchable text + metadata
```

`IndexFullText` writes to `TextIndexStoreProtocol`; `full_text_search` queries it. Text index stores do not save raw files, chunk JSON, vectors, or graph facts.

## Implementations

| Store | Use |
| --- | --- |
| `InMemoryTextIndexStore` | Local development, unit tests, and lightweight demos. |
| `ElasticsearchTextIndexStore` | Production full-text retrieval with Elasticsearch BM25 ranking. |

## Elasticsearch

Install:

```bash
pip install "heta[elasticsearch]"
```

Configure:

```python
from heta_framework.common.stores import (
    ElasticsearchTextIndexStore,
    ElasticsearchTextIndexStoreConfig,
)

text_index = ElasticsearchTextIndexStore(
    ElasticsearchTextIndexStoreConfig(
        hosts="http://localhost:9200",
        request_timeout=30,
    )
)
```

Use it in a recipe:

```python
recipe = KnowledgeRecipe(
    stores=KnowledgeStores(
        objects=objects,
        text_index=text_index,
    ),
    steps=(
        ParseDocuments(),
        SplitDocuments(),
        IndexFullText(),
    ),
)
```

Query:

```python
response = await kb.query(
    "flight control fault",
    mode="full_text_search",
    top_k=5,
)
```

## Relationship To SQL Text Search

`IndexFullText` and `PersistChunks` are separate steps:

| Step | Store | Query mode | Use |
| --- | --- | --- | --- |
| `IndexFullText` | `TextIndexStoreProtocol` | `full_text_search` | BM25-style full-text retrieval. |
| `PersistChunks` | `SQLStoreProtocol` | `sql_text_search` | SQL persistence and lightweight LIKE retrieval. |

If you only need Elasticsearch full-text search, use `IndexFullText`. If you also need SQL evidence tables, chunk tables, or lightweight LIKE retrieval, add `PersistChunks`.

## Scope

Text Index Stores write searchable text, preserve retrieval metadata, return relevance-ranked results, and connect to systems such as Elasticsearch.

They do not store raw files, persist chunk JSON, compute embeddings, run vector search, build graphs, or manage the `KnowledgeBase` lifecycle.
