# Merge Chunks

`MergeChunks` is an optional Heta graph procedure step. It merges semantically overlapping chunks before graph extraction.

```text
ParsedChunk JSON + chunk vectors -> merged ParsedChunk JSON
```

The output is still `ParsedChunk`. This keeps downstream steps compatible whether the recipe uses original chunks, merged chunks, or rechunked chunks.

## When To Use It

Use `MergeChunks` when graph extraction benefits from wider, less fragmented context. It is normally placed after `IndexVectors` and before `RechunkDocuments`.

```text
SplitDocuments
  -> EmbedChunks
  -> IndexVectors
  -> MergeChunks
  -> RechunkDocuments
  -> ExtractEntities
```

It does not create a new user-facing query mode. Its purpose is to improve graph construction.

## Contract

`MergeChunks` uses:

```text
stores.objects
stores.vector
models.embedding
models.language
```

Default input artifacts:

```text
chunk_keys
chunk_embedding_keys
```

Execution flow:

```text
load chunks
  -> search vector neighbors
  -> send candidates to LLM
  -> merge accepted chunks
  -> write merged_chunks/{chunk_id}.json
  -> expose merged_chunk_keys
```

## Output

Merged chunks are written as `ParsedChunk` records:

```python
ParsedChunk(
    chunk_id="chunk_merged_...",
    document_id="doc_...",
    text="merged text",
    parent_chunk_ids=("chunk_a", "chunk_b"),
)
```

The shared `ParsedChunk` shape is important: later steps do not need separate code paths for original, merged, or rechunked chunks.

## Artifacts

```text
merge_chunks_result
merged_chunk_keys
```

`merged_chunk_keys` can be consumed by `RechunkDocuments`. If a recipe skips `MergeChunks`, `RechunkDocuments` can consume `chunk_keys` instead.

## HetaDB Alignment

The algorithm follows the HetaDB idea:

- use vector similarity to find merge candidates
- ask an LLM to judge whether candidates should merge
- keep source traceability through parent chunk IDs
- avoid exposing the merge collection as a final retrieval index

The merge collection is an internal workspace for graph construction, not the main vector search collection.
