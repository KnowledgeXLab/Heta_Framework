# Text Index Stores

Text Index Stores 是 Heta 的全文检索索引入口。它处理的是：

```text
chunk id -> searchable text + metadata
```

`IndexFullText` 写入 `TextIndexStoreProtocol`，`full_text_search` 查询它。它不负责保存原始文件、chunk JSON、向量或图谱事实。

## Implementations

Heta 当前提供两个实现：

| Store | 用途 |
| --- | --- |
| `InMemoryTextIndexStore` | 本地开发、单元测试和轻量 demo。 |
| `ElasticsearchTextIndexStore` | 生产全文检索，使用 Elasticsearch BM25 排序。 |

## Elasticsearch

安装：

```bash
pip install "heta-framework[elasticsearch]"
```

配置：

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

放进 recipe：

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

查询：

```python
response = await kb.query(
    "flight control fault",
    mode="full_text_search",
    top_k=5,
)
```

## Relationship To SQL Text Search

`IndexFullText` 和 `PersistChunks` 是两个独立步骤：

| Step | Store | Query mode | 用途 |
| --- | --- | --- | --- |
| `IndexFullText` | `TextIndexStoreProtocol` | `full_text_search` | BM25-style 全文检索。 |
| `PersistChunks` | `SQLStoreProtocol` | `sql_text_search` | SQL 表持久化和轻量 LIKE 检索。 |

如果业务只需要 Elasticsearch 全文检索，可以只加入 `IndexFullText`。如果还需要 SQL 证据表、chunk 表或轻量 LIKE 检索，再加入 `PersistChunks`。

## Scope

Text Index Stores 负责：

- 写入可搜索文本。
- 保存检索所需 metadata。
- 根据 query 返回按文本相关性排序的结果。
- 对接 Elasticsearch 等全文检索系统。

Text Index Stores 不负责原始文件存储、chunk JSON 持久化、embedding、向量检索、图谱构建或 `KnowledgeBase` 生命周期管理。
