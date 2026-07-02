# Index Vectors

`IndexVectors` 将 `ParsedChunk` 和 `ChunkEmbedding` 合并为 `VectorRecord`，写入 `VectorStore`。这是最小向量知识库链路中第一个提供查询能力的 step。

```text
ParsedChunk JSON + ChunkEmbedding JSON -> VectorStore
```

完成后提供：

```text
vector_search
```

## Contract

`IndexVectors` 使用两个 recipe components：

```text
stores.objects
stores.vector
```

默认读取上游 artifacts：

```text
chunk_keys
chunk_embedding_keys
```

执行流程：

```text
read chunk_keys and chunk_embedding_keys
  -> load ParsedChunk JSON
  -> load ChunkEmbedding JSON
  -> validate chunk_id and document_id alignment
  -> create vector collection if needed
  -> upsert VectorRecord batches
  -> expose index_vectors_result artifact
  -> enable vector_search query mode
```

`IndexVectors` 不调用 embedding model。它只负责把已经生成的向量写入向量存储。

## Configuration

```python
IndexVectorsConfig(
    collection_names=ChunkVectorCollections(chunks="chunks"),
    metric="cosine",
    batch_size=128,
    object_store=None,
    vector_store=None,
    chunk_keys_artifact="chunk_keys",
    chunk_embedding_keys_artifact="chunk_embedding_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `collection_names.chunks` | chunk 向量 collection 名称。 |
| `metric` | 向量距离度量，支持 `cosine`、`dot`、`l2`。 |
| `batch_size` | 单次 upsert 的 record 数量。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `vector_store` | 命名 VectorStore。默认引用 `stores.vector`。 |
| `chunk_keys_artifact` | 上游 chunk key artifact 名称。 |
| `chunk_embedding_keys_artifact` | 上游 embedding key artifact 名称。 |

命名组件示例：

```python
def chunk_vectors(prefix: str) -> ChunkVectorCollections:
    return ChunkVectorCollections(chunks=f"{prefix}_chunks")

IndexVectorsConfig(
    vector_store="milvus",
    collection_names=chunk_vectors("papers"),
)
```

对应引用：

```text
stores.vector.milvus
```

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        store_ref("vector"),
    }),
    artifacts=frozenset({
        "chunk_keys",
        "chunk_embedding_keys",
    }),
)
```

含义：

| Requirement | 说明 |
| --- | --- |
| `stores.objects` | 满足 `ObjectStoreProtocol` 的对象存储。 |
| `stores.vector` | 满足 `VectorStoreProtocol` 的向量存储。 |
| `chunk_keys` | `SplitDocuments` 产生的 chunk JSON key 列表。 |
| `chunk_embedding_keys` | `EmbedChunks` 产生的 embedding JSON key 列表。 |

## Capabilities

`IndexVectors` 提供一个 artifact 和一个 query mode：

```python
StepCapabilities(
    artifacts=frozenset({
        "index_vectors_result",
    }),
    queries=frozenset({
        "vector_search",
    }),
)
```

`vector_search` 表示当前知识库已经具备向量检索能力。`KnowledgeBase.query(..., mode="vector_search")` 会使用这个能力。

## Artifacts

`index_vectors_result` 是 `IndexVectorsResult`：

```python
IndexVectorsResult(
    collection="chunks",
    indexed_count=12,
    dimension=1536,
)
```

| 字段 | 说明 |
| --- | --- |
| `collection` | 写入的 VectorStore collection。 |
| `indexed_count` | 本次写入的 vector record 数量。 |
| `dimension` | collection 使用的向量维度。 |

## Vector Record

每个 chunk 会写成一个 `VectorRecord`：

```python
VectorRecord(
    id=chunk.chunk_id,
    vector=embedding.vector,
    text=chunk.text,
    metadata={
        "document_id": chunk.document_id,
        "source_key": chunk.source.key,
        "source_name": chunk.source.name,
        "source_file_type": chunk.source.file_type,
        "page_index": chunk.page_index,
        "chunk_index": chunk.chunk_index,
        "token_start": chunk.token_start,
        "token_end": chunk.token_end,
        "embedding_model": embedding.model_name,
    },
)
```

metadata 保留 citation 和定位所需的信息。检索结果可以直接知道命中的 chunk 来自哪个文档、哪个 page-like 单元，以及在该 page token 序列中的范围。

## Validation

`IndexVectors` 会校验：

- 每个 chunk 都有对应 embedding
- embedding keys 不能包含重复 chunk id
- `ChunkEmbedding.document_id` 必须与 `ParsedChunk.document_id` 一致
- VectorStore collection 的维度必须与 embedding 维度一致

这些校验可以避免 chunk 与 embedding 错配后被写入索引。
