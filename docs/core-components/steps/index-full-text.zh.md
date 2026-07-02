# Index Full Text

`IndexFullText` 将 `ParsedChunk` 写入 `TextIndexStore`，用于提供独立的全文检索能力。

```text
ParsedChunk JSON -> TextIndexStore
```

完成后提供：

```text
full_text_search
```

它是可选步骤，和 `PersistChunks`、`EmbedChunks`、`IndexVectors` 是并列的下游分支。它不依赖 SQL chunk 表，也不依赖向量索引。

## Contract

`IndexFullText` 使用两个 recipe components：

```text
stores.objects
stores.text_index
```

默认读取上游 artifact：

```text
chunk_keys
```

执行语义：

```text
read chunk_keys
  -> load ParsedChunk JSON from ObjectStore
  -> create full-text index if needed
  -> upsert TextIndexRecord batches
  -> expose index_full_text_result artifact
  -> enable full_text_search query mode
```

## Configuration

```python
IndexFullTextConfig(
    index_names=FullTextIndexNames(chunk_text="chunk_full_text"),
    batch_size=128,
    object_store=None,
    text_index_store=None,
    chunk_keys_artifact="chunk_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `index_names.chunk_text` | chunk 全文检索索引名。 |
| `batch_size` | 每批写入 `TextIndexStore` 的记录数量。 |
| `object_store` | 命名 ObjectStore。默认引用 `stores.objects`。 |
| `text_index_store` | 命名 TextIndexStore。默认引用 `stores.text_index`。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |

## Requirements

默认 requirements：

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        store_ref("text_index"),
    }),
    artifacts=frozenset({
        "chunk_keys",
    }),
)
```

## Capabilities

`IndexFullText` 提供一个 artifact 和一个 query mode：

```python
StepCapabilities(
    artifacts=frozenset({
        "index_full_text_result",
    }),
    queries=frozenset({
        "full_text_search",
    }),
)
```

同时声明一个 search asset：

```text
SearchAsset(kind="chunk_full_text_index")
```

这表示当前 KB 已经具备全文检索索引。`KnowledgeBase.query(..., mode="full_text_search")` 会使用该 asset。

## Artifacts

`index_full_text_result` 是 `IndexFullTextResult`：

```python
IndexFullTextResult(
    index_name="chunk_full_text",
    indexed_count=12,
)
```

| 字段 | 说明 |
| --- | --- |
| `index_name` | 写入的 TextIndexStore 索引名。 |
| `indexed_count` | 本次写入的 chunk text record 数量。 |

## TextIndexRecord

每个 chunk 会写成一个 `TextIndexRecord`：

```python
TextIndexRecord(
    id=chunk.chunk_id,
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
        "parent_chunk_ids": list(chunk.parent_chunk_ids),
    },
)
```

metadata 保留 citation、过滤和回溯所需的信息。

## Boundary

`IndexFullText` 不依赖 `PersistChunks`：

```text
SplitDocuments
  ├─ PersistChunks        -> sql_text_search
  ├─ IndexFullText        -> full_text_search
  └─ EmbedChunks
      └─ IndexVectors     -> vector_search
```

如果需要 BM25-style 检索，使用 `IndexFullText`。如果需要 SQL 证据表、`LIKE` 兜底检索或图谱溯源表，使用 `PersistChunks`。
