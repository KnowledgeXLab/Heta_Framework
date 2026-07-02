# Full-Text Search

`full_text_search` 检索 `IndexFullText` 写入 TextIndexStore 的 chunk 文本。

它面向 BM25、phrase query、analyzer、field boost 以及 Elasticsearch/OpenSearch/Tantivy 这类全文检索索引。当前框架内置：

- `InMemoryTextIndexStore`：本地开发和测试使用。
- `ElasticsearchTextIndexStore`：生产全文检索使用，基于官方 Elasticsearch async client。

安装 Elasticsearch 支持：

```bash
pip install "heta[elasticsearch]"
```

## Required Asset

`IndexFullText` 会声明：

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

只要 KB 的 latest run record 中存在这个资产，默认 query registry 就会启用：

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

生产环境通常这样配置：

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

返回结果仍然是统一的 `QueryResponse`。每条 `QueryResult` 表示一个 chunk：

```text
id
text
score
source
metadata
```

`source` 与 `vector_search`、`sql_text_search` 保持一致，包含 document、object key、chunk id、page、chunk index 和 token offset 等 provenance 字段。

## Scope

`full_text_search` 只负责全文检索召回和排序，不负责 SQL chunk 持久化。需要 SQL 证据表时，继续显式加入 `PersistChunks`。

`full_text_search` 和 `sql_text_search` 是并列能力。前者来自 full-text index，后者来自 SQL chunk table。它们不互相依赖，也不自动 fallback。
