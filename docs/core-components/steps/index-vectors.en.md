# Index Vectors

`IndexVectors` writes chunk embeddings into a `VectorStore` and unlocks semantic retrieval.

```text
ParsedChunk JSON + ChunkEmbedding JSON -> VectorStore
```

After this step succeeds, the KB can use:

```text
vector_search
```

## Contract

`IndexVectors` uses:

```text
stores.objects
stores.vector
```

Default input artifacts:

```text
chunk_keys
chunk_embedding_keys
```

Execution flow:

```text
read chunk_keys and chunk_embedding_keys
  -> validate chunk_id and document_id match
  -> build VectorRecord with text and metadata
  -> upsert into VectorStore
  -> declare chunk_vector_index
  -> unlock vector_search
```

## Configuration

```python
IndexVectorsConfig(
    collection_name="chunks",
    object_store=None,
    vector_store=None,
    chunk_keys_artifact="chunk_keys",
    embedding_keys_artifact="chunk_embedding_keys",
)
```

| Parameter | Meaning |
| --- | --- |
| `collection_name` | Vector collection to write. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `vector_store` | Named VectorStore. Defaults to `stores.vector`. |
| `chunk_keys_artifact` | Artifact containing chunk JSON keys. |
| `embedding_keys_artifact` | Artifact containing embedding JSON keys. |

## Requirements

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

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({"index_vectors_result"}),
    queries=frozenset({"vector_search"}),
    search_assets=(
        SearchAsset(kind="chunk_vector_index", ...),
    ),
)
```

`chunk_vector_index` tells the query layer which vector collection to use.

## Vector Records

Each vector record stores:

```text
id = chunk_id
vector = embedding vector
text = chunk text
metadata = citation and source fields
```

Important metadata:

```text
document_id
source_key
source_name
file_type
page_index
chunk_index
token_start
token_end
```

This lets `vector_search` return source-aware `QueryResult` objects without loading chunk JSON from ObjectStore.

## Idempotency

`IndexVectors` uses vector upsert semantics. Re-running the same build with the same chunk IDs updates the same records instead of creating duplicates.
