# Persist Chunks

`PersistChunks` writes chunk text and source metadata into SQL. It provides a lightweight SQL text retrieval path and gives graph workflows a durable evidence table to reference.

```text
ParsedChunk JSON -> SQL chunk table
```

After this step succeeds, the KB can use:

```text
sql_text_search
```

## Contract

`PersistChunks` uses:

```text
stores.objects
stores.sql
```

Default input artifact:

```text
chunk_keys
```

In Heta graph procedures, it often consumes `rechunked_chunk_keys` instead.

Execution flow:

```text
read chunk keys
  -> load ParsedChunk JSON
  -> upsert chunk rows by chunk_id
  -> declare chunk_text_index
  -> unlock sql_text_search
```

## Configuration

```python
PersistChunksConfig(
    table_name="chunks",
    dialect="generic",
    object_store=None,
    sql_store=None,
    chunk_keys_artifact="chunk_keys",
)
```

| Parameter | Meaning |
| --- | --- |
| `table_name` | SQL table for chunk rows. |
| `dialect` | SQL dialect behavior. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `sql_store` | Named SQLStore. Defaults to `stores.sql`. |
| `chunk_keys_artifact` | Artifact containing chunk JSON keys. |

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({"persist_chunks_result"}),
    queries=frozenset({"sql_text_search"}),
    search_assets=(
        SearchAsset(kind="chunk_text_index", ...),
    ),
)
```

## Table Shape

The table stores chunk identity, text, source metadata, and trace fields:

```text
chunk_id
document_id
source_key
source_name
file_type
page_index
chunk_index
token_start
token_end
text
metadata
```

`sql_text_search` uses this table for simple SQL text matching. For production BM25 retrieval, use `IndexFullText` with a `TextIndexStore`.

## Idempotency

`PersistChunks` writes by `chunk_id`. Re-running the same build updates or replaces the same logical rows instead of appending duplicate chunks.
