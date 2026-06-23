# Rechunk Documents

`RechunkDocuments` 是可选的图谱输入增强 step。它按 document/source 聚合 chunks，将文本重新拼接后再切分，输出更稳定的 `ParsedChunk`。

```text
merged_chunk_keys
  -> group by document/source
  -> concatenate text
  -> split again
  -> rechunked_chunk_keys
```

它对应 HetaDB 的 `rechunk_by_source` 思路：merge 后的 chunk 不直接进入图谱抽取，而是先按原始文档重新组织边界。

## Contract

`RechunkDocuments` 使用一个 recipe component：

```text
stores.objects
```

默认读取：

```text
merged_chunk_keys
```

默认输出：

```text
rechunk_documents_result
rechunked_chunk_keys
```

如果不启用 `MergeChunks`，也可以显式让它读取原始 chunks：

```python
RechunkDocuments(
    RechunkDocumentsConfig(chunk_keys_artifact="chunk_keys")
)
```

## Configuration

```python
RechunkDocumentsConfig(
    rechunked_chunks_prefix="rechunked_chunks",
    chunk_size=1024,
    overlap=50,
    encoding_name="cl100k_base",
    chunk_keys_artifact="merged_chunk_keys",
)
```

| 参数 | 说明 |
| --- | --- |
| `rechunked_chunks_prefix` | rechunked chunk JSON 写入 ObjectStore 的 prefix。 |
| `chunk_size` | 重新切分的目标 token 数。 |
| `overlap` | 相邻 chunk 的 token overlap。 |
| `encoding_name` | tokenizer 名称。 |
| `chunk_keys_artifact` | 输入 chunk key artifact 名称。 |

## Output

rechunk 后仍然输出 `ParsedChunk`：

```python
ParsedChunk(
    chunk_id="chunk_...",
    document_id="doc_...",
    text="重新切分后的文本",
    parent_chunk_ids=("chunk_a", "chunk_b"),
    ...
)
```

`parent_chunk_ids` 记录当前 rechunked chunk 覆盖到的原始 chunk ids。下游图谱、证据查询和 citation 可以通过它回溯来源。

## Usage

```python
steps=[
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    MergeChunks(),
    RechunkDocuments(),
]
```

图谱抽取可以选择读取：

```text
chunk_keys              # 快速图谱
rechunked_chunk_keys    # 高质量图谱
```
