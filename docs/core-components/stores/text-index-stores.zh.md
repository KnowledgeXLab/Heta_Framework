# Text Index Stores

`TextIndexStoreProtocol` 负责全文检索索引。

它只处理一类数据：

```text
chunk id -> searchable text + metadata
```

`IndexFullText` 写入它，`full_text_search` 查询它。它不负责保存原始文件、chunk JSON、向量或图谱。

## Implementations

Heta 当前提供两个实现：

| Store | 用途 |
| --- | --- |
| `InMemoryTextIndexStore` | 本地开发、单元测试、轻量 demo |
| `ElasticsearchTextIndexStore` | 生产全文检索，使用 Elasticsearch BM25 排序 |

## Elasticsearch

安装：

```bash
pip install "heta[elasticsearch]"
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

## Boundaries

`TextIndexStoreProtocol` 不要求 SQL。`PersistChunks` 和 `IndexFullText` 是两个独立步骤：

- `PersistChunks` 产出 `sql_text_search`
- `IndexFullText` 产出 `full_text_search`

如果业务只需要 Elasticsearch 全文检索，可以只加入 `IndexFullText`。如果还需要 SQL 证据表或轻量 LIKE 检索，再加入 `PersistChunks`。
