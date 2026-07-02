# Index Full Text

`IndexFullText` writes chunk text into a `TextIndexStore` and unlocks BM25-style full-text retrieval.

```text
ParsedChunk JSON -> TextIndexStore
```

After this step succeeds, the KB can use:

```text
full_text_search
```

## Contract

`IndexFullText` uses:

```text
stores.objects
stores.text_index
```

Default input artifact:

```text
chunk_keys
```

Execution flow:

```text
read chunk_keys
  -> load ParsedChunk JSON
  -> write text and metadata into TextIndexStore
  -> declare chunk_full_text_index
  -> unlock full_text_search
```

## Configuration

```python
IndexFullTextConfig(
    index_name="chunks",
    object_store=None,
    text_index_store=None,
    chunk_keys_artifact="chunk_keys",
)
```

| Parameter | Meaning |
| --- | --- |
| `index_name` | Full-text index name. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `text_index_store` | Named TextIndexStore. Defaults to `stores.text_index`. |
| `chunk_keys_artifact` | Artifact containing chunk JSON keys. |

## Requirements

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        store_ref("text_index"),
    }),
    artifacts=frozenset({"chunk_keys"}),
)
```

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({"index_full_text_result"}),
    queries=frozenset({"full_text_search"}),
    search_assets=(
        SearchAsset(kind="chunk_full_text_index", ...),
    ),
)
```

## Relationship To PersistChunks

`IndexFullText` and `PersistChunks` are separate:

| Step | Store | Query mode | Use |
| --- | --- | --- | --- |
| `IndexFullText` | `TextIndexStoreProtocol` | `full_text_search` | BM25-style retrieval, usually Elasticsearch in production. |
| `PersistChunks` | `SQLStoreProtocol` | `sql_text_search` | SQL chunk persistence and lightweight LIKE retrieval. |

Use `IndexFullText` when you need a dedicated full-text index. Use `PersistChunks` when SQL chunk tables are part of your graph, evidence, or operational workflow.
