# Merge Chunks

`MergeChunks` 是可选的 chunk 质量增强 step。它使用向量相似度召回候选 chunk，再由 LLM 判断哪些候选与主 chunk 语义重复，并将合并结果写回一个临时 merge collection。

```text
ParsedChunk JSON + ChunkEmbedding JSON
  -> merge collection
  -> vector candidates
  -> LLM merge decision
  -> merged ParsedChunk JSON
```

它对齐 HetaDB 的 chunk merge 思路，但保留 Heta Framework 的组件边界：ObjectStore 管 chunk JSON，VectorStore 管 merge 工作区，LanguageModel 做合并判断，EmbeddingModel 生成 merged chunk 向量。

## Contract

`MergeChunks` 使用四个 recipe components：

```text
stores.objects
stores.vector
models.language
models.embedding
```

默认读取 artifacts：

```text
chunk_keys
chunk_embedding_keys
```

默认输出 artifacts：

```text
merge_chunks_result
merged_chunk_keys
```

`merged_chunk_keys` 表示 merge 后仍然活跃的 chunk keys。它可能包含未被合并的原始 chunk，也可能包含新生成的 merged chunk。

## Configuration

```python
MergeChunksConfig(
    merged_chunks_prefix="merged_chunks",
    merge_collection="merge_chunks",
    metric="cosine",
    top_k=8,
    num_topk_candidates=5,
    max_rounds=10,
    min_similarity=0.85,
    merge_threshold=0.05,
    recreate_collection=True,
)
```

| 参数 | 说明 |
| --- | --- |
| `merged_chunks_prefix` | merged chunk JSON 写入 ObjectStore 的 prefix。 |
| `merge_collection` | VectorStore 中的临时 merge collection。 |
| `top_k` | 向量召回候选数量。 |
| `num_topk_candidates` | 实际交给 LLM 判断的候选数量。 |
| `max_rounds` | 最多 merge 轮数。 |
| `min_similarity` | 进入 LLM 判断前的向量相似度下限。 |
| `merge_threshold` | 本轮合并比例低于该值时停止。 |
| `recreate_collection` | 是否在运行开始时重建 merge collection。 |

## Merge Collection

`merge_collection` 是工作区，不是最终查询 collection。

`MergeChunks.run()` 默认会重建这个 collection，然后写入当前输入 chunks。每次 LLM 决定合并后，step 会删除被合并的旧 chunk 向量，并写入新的 merged chunk 向量。

这让 merge 过程保持自洽：下一轮召回使用的是最新活跃 chunk 集合。

## Output

merged chunk 仍然是 `ParsedChunk`：

```python
ParsedChunk(
    chunk_id="chunk_...",
    document_id="doc_...",
    text="合并后的文本",
    parent_chunk_ids=("chunk_a", "chunk_b"),
    ...
)
```

`parent_chunk_ids` 是溯源字段。原始 chunk 的 `parent_chunk_ids` 为空；merged chunk 会记录它吸收的原始 chunk ids。

## Usage

```python
steps=[
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
    MergeChunks(),
]
```

如果后续要做高质量图谱抽取，通常继续接：

```python
RechunkDocuments()
```
