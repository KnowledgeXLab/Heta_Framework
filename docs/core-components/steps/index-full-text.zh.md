# Index Full Text

`IndexFullText` 将 `ParsedChunk` 写入 TextIndexStore，用于解锁 `full_text_search`。

```text
ParsedChunk JSON -> full-text index
```

它是可选步骤，和 `PersistChunks`、`EmbedChunks`、`IndexVectors` 是并列下游分支。

## Contract

`IndexFullText` 使用两个 recipe components：

```text
stores.objects
stores.text_index
```

默认读取：

```text
chunk_keys
```

默认输出：

```text
index_full_text_result
```

完成后会声明：

```text
SearchAsset(kind="chunk_full_text_index")
query mode: full_text_search
```

因此知识库可以通过 `KnowledgeBase.query(..., mode="full_text_search")` 进行全文检索。

## Configuration

```python
IndexFullTextConfig(
    index_names=FullTextIndexNames(chunk_text="chunk_full_text"),
    chunk_keys_artifact="chunk_keys",
    batch_size=128,
)
```

| 参数 | 说明 |
| --- | --- |
| `index_names.chunk_text` | chunk 全文检索索引名。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |
| `batch_size` | 每批写入 TextIndexStore 的记录数量。 |

## Boundary

`IndexFullText` 不依赖 `PersistChunks`。它直接从 ObjectStore 读取 chunk JSON，再写入全文检索索引。

```text
SplitDocuments
  ├─ PersistChunks
  ├─ IndexFullText
  └─ EmbedChunks -> IndexVectors
```

如果只需要全文检索，可以只配置 ObjectStore 和 TextIndexStore；如果还需要 SQL 证据表，可以额外加入 `PersistChunks`。
