# Full-Text Search

`full_text_search` retrieves chunk text written by `IndexFullText` into a `TextIndexStore`.

It is intended for BM25, phrase query, analyzers, field boosts, and full-text systems such as Elasticsearch, OpenSearch, or Tantivy. Heta currently includes:

- `InMemoryTextIndexStore` for local development and tests.
- `ElasticsearchTextIndexStore` for production full-text retrieval.

Install Elasticsearch support:

```bash
pip install "heta-framework[elasticsearch]"
```

## Required Asset

`IndexFullText` declares:

```python
SearchAsset(
    kind="chunk_full_text_index",
    name="chunk_full_text",
    store="stores.text_index",
    metadata={
        "index": "chunk_full_text",
        "ranking": "bm25",
    },
)
```

When this asset exists in the latest run record, the default query registry enables:

```text
full_text_search
```

## Usage

```python
response = await kb.query(
    "flight control fault",
    mode="full_text_search",
    top_k=5,
)
```

Production configuration usually looks like:

```python
from heta_framework.common.stores import (
    ElasticsearchTextIndexStore,
    ElasticsearchTextIndexStoreConfig,
)

text_index = ElasticsearchTextIndexStore(
    ElasticsearchTextIndexStoreConfig(
        hosts="http://localhost:9200",
    )
)
```

The response is the same `QueryResponse` shape as other modes. Each `QueryResult` represents a chunk:

```text
id
text
score
source
metadata
```

`source` is aligned with `vector_search` and `sql_text_search` and includes document, object key, chunk id, page, chunk index, and token offsets.

## Scope

`full_text_search` only performs full-text recall and ranking. It does not persist SQL chunk tables. If you need SQL evidence tables, add `PersistChunks` explicitly.

`full_text_search` and `sql_text_search` are parallel capabilities. The former comes from a full-text index; the latter comes from a SQL chunk table. They do not depend on each other and do not automatically fall back.
