# Vector Search

`vector_search` 是 Heta 的基础向量检索 query mode。它消费 `IndexVectors` 产出的 `chunk_vector_index`，通过 `KnowledgeBase.query()` 返回语义相似的 chunk。

## Required Asset

`IndexVectors` 会声明：

```python
SearchAsset(
    kind="chunk_vector_index",
    name="chunks",
    store="stores.vector",
    metadata={
        "collection": "chunks",
        "id_field": "id",
        "text_field": "text",
        "metadata_field": "metadata",
    },
)
```

只要 KB 的 latest run record 中存在这个资产，默认 query registry 就会启用：

```text
vector_search
```

## Usage

```python
response = await kb.query(
    "How does Heta build a knowledge base?",
    mode="vector_search",
    top_k=5,
)

for result in response.results:
    print(result.score, result.text)
```

返回结果是统一的 `QueryResponse`：

```text
mode
results
metadata
```

每条 `QueryResult` 表示一个 chunk：

```text
id
text
score
kind = "chunk"
source
metadata
```

`source` 中包含 chunk 的来源信息：

```text
document_id
source_key
source_name
page_index
chunk_index
token_start
token_end
```

## Execution Flow

```text
query text
  -> models.embedding.embed()
  -> stores.vector.search(collection="chunks")
  -> QueryResponse
```

`VectorSearchEngine` 不直接读取 ObjectStore。它只依赖 vector record 中的 `text` 和 `metadata`。

## Filters

`filters` 会原样传给底层 `VectorStore.search()`：

```python
response = await kb.query(
    "Heta graph",
    mode="vector_search",
    filters={"document_id": "doc_123"},
)
```

具体 filter 能力取决于底层 vector store。内存实现使用 metadata 精确匹配；Milvus 实现会转换为 Milvus filter expression。

## Scope

`vector_search` 只负责向量召回。它不做 BM25、SQL 文本检索、图谱补全、rerank、query rewrite 或答案生成。
