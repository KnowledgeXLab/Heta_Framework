# Embed Chunks

`EmbedChunks` generates embeddings for `ParsedChunk` records and writes them as reusable `ChunkEmbedding` JSON.

```text
ParsedChunk JSON -> ChunkEmbedding JSON
```

It only calls the embedding model. Vector store writes and query capability are handled by `IndexVectors`.

## Contract

`EmbedChunks` uses:

```text
stores.objects
models.embedding
```

Default input artifact:

```text
chunk_keys
```

Default output prefix:

```text
embeddings/
```

Execution flow:

```text
read chunk_keys
  -> load ParsedChunk JSON
  -> batch chunk text
  -> call EmbeddingModelProtocol.embed
  -> write embeddings/{chunk_id}.json
  -> expose chunk_embedding_keys
```

Keeping embeddings in ObjectStore makes them cacheable, inspectable, and reusable with different vector stores.

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

| Parameter | Meaning |
| --- | --- |
| `embeddings_prefix` | Prefix for `ChunkEmbedding` JSON. |
| `batch_size` | Number of chunks per embedding request. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `embedding_model` | Named embedding model. Defaults to `models.embedding`. |
| `chunk_keys_artifact` | Upstream chunk key artifact name. |

## Requirements

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        model_ref("embedding"),
    }),
    artifacts=frozenset({"chunk_keys"}),
)
```

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({
        "embed_chunks_result",
        "chunk_embedding_keys",
    })
)
```

This step does not unlock a query mode. `IndexVectors` unlocks `vector_search`.

## Artifacts

```python
EmbedChunksResult(
    embedding_keys=("embeddings/chunk_abc123.json",),
    chunk_count=12,
    model_name="text-embedding-3-small",
    dimension=1536,
)
```

`chunk_embedding_keys` is the tuple consumed by `IndexVectors`.

## Embedding Output

```python
ChunkEmbedding(
    chunk_id="chunk_...",
    document_id="doc_...",
    model_name="text-embedding-3-small",
    vector=[...],
    dimension=1536,
)
```

Default object key:

```text
embeddings/{chunk_id}.json
```

`ChunkEmbedding` does not duplicate chunk text. Text remains in `ParsedChunk`.
