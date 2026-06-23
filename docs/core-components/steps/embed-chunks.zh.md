# Embed Chunks

`EmbedChunks` 为 `ParsedChunk` 生成 embedding，并将结果写成可缓存的 `ChunkEmbedding` JSON。

```text
ParsedChunk JSON -> ChunkEmbedding JSON
```

它只负责调用 embedding model。向量库写入和查询能力由后续 `IndexVectors` step 提供。

## Contract

`EmbedChunks` 使用两个 recipe components：

```text
stores.objects
models.embedding
```

默认读取上游 artifact：

```text
chunk_keys
```

默认写入路径：

```text
embeddings/
```

执行语义：

```text
read chunk_keys
  -> load ParsedChunk JSON from ObjectStore
  -> batch texts
  -> call EmbeddingModelProtocol.embed
  -> write embeddings/{chunk_id}.json
  -> expose embedding keys as artifacts
```

`EmbedChunks` 不写入 VectorStore。这个分离让 embedding 结果可以被缓存、检查、复用，或者写入不同的向量后端。

## Configuration

```python
EmbedChunksConfig(
    embeddings_prefix="embeddings",
    batch_size=64,
    object_store=None,
    embedding_model=None,
    chunk_keys_artifact="chunk_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `embeddings_prefix` | `ChunkEmbedding` JSON 写入 prefix。 |
| `batch_size` | 单次 embedding 请求包含的 chunk 数量。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `embedding_model` | 命名 embedding model。默认引用 `models.embedding`。 |
| `chunk_keys_artifact` | 上游 chunk key artifact 名称。 |

命名组件示例：

```python
EmbedChunksConfig(
    object_store="local",
    embedding_model="large",
)
```

对应引用：

```text
stores.objects.local
models.embedding.large
```

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        model_ref("embedding"),
    }),
    artifacts=frozenset({
        "chunk_keys",
    }),
)
```

含义：

| Requirement | 说明 |
| --- | --- |
| `stores.objects` | 满足 `ObjectStoreProtocol` 的对象存储。 |
| `models.embedding` | 满足 `EmbeddingModelProtocol` 的 embedding model。 |
| `chunk_keys` | `SplitDocuments` 产生的 chunk JSON key 列表。 |

## Capabilities

`EmbedChunks` 提供两个 artifacts：

```python
StepCapabilities(
    artifacts=frozenset({
        "embed_chunks_result",
        "chunk_embedding_keys",
    })
)
```

它不直接提供 query mode。查询能力由 `IndexVectors` 在写入 VectorStore 后提供。

## Artifacts

`embed_chunks_result` 是 `EmbedChunksResult`：

```python
EmbedChunksResult(
    embedding_keys=(
        "embeddings/chunk_abc123.json",
    ),
    chunk_count=12,
    model_name="text-embedding-v4",
    dimension=1536,
)
```

| 字段 | 说明 |
| --- | --- |
| `embedding_keys` | 已写入 ObjectStore 的 `ChunkEmbedding` JSON keys。 |
| `chunk_count` | 本次处理的 chunk 数量。 |
| `model_name` | 使用的 embedding model 名称。 |
| `dimension` | embedding 向量维度。 |

`chunk_embedding_keys` 是 `embedding_keys` 的快捷 tuple，方便 `IndexVectors` 读取。

## Embedding Output

每个写入的 JSON 都是 `ChunkEmbedding`：

```python
ChunkEmbedding(
    chunk_id="chunk_...",
    document_id="doc_...",
    model_name="text-embedding-v4",
    vector=[...],
    dimension=1536,
)
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | 被 embedding 的 chunk ID。 |
| `document_id` | chunk 所属文档 ID，用于一致性校验和后续聚合。 |
| `model_name` | 生成该向量的 embedding model。 |
| `vector` | embedding 向量。 |
| `dimension` | 向量维度，必须与 `vector` 长度一致。 |

ObjectStore 中的默认位置：

```text
embeddings/{chunk_id}.json
```

`ChunkEmbedding` 不包含 chunk 文本。文本保留在 `ParsedChunk` 中，避免 embedding 产物重复存储正文。
