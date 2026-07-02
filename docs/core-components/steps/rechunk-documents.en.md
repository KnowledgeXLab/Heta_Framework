# Rechunk Documents

`RechunkDocuments` is an optional Heta graph procedure step. It rebuilds graph extraction chunks from merged or original chunks.

```text
ParsedChunk JSON -> rechunked ParsedChunk JSON
```

The output is still `ParsedChunk`, so entity and relation extraction can consume either original chunks or rechunked chunks with the same protocol.

## Purpose

Vector retrieval usually wants compact chunks. Graph extraction often benefits from context that is grouped differently. `RechunkDocuments` lets a recipe prepare graph-friendly chunks without changing the base retrieval index.

It does not write a new vector collection and does not unlock a new query mode.

## Contract

`RechunkDocuments` uses:

```text
stores.objects
```

Default input artifact:

```text
merged_chunk_keys
```

It can also consume `chunk_keys` if the recipe skips `MergeChunks`.

Execution flow:

```text
read chunk keys
  -> group by document/source
  -> concatenate text in stable order
  -> split again with configured chunk size and overlap
  -> write rechunked_chunks/{chunk_id}.json
  -> expose rechunked_chunk_keys
```

## Configuration

The configuration mirrors the normal splitter: output prefix, chunk size, overlap, tokenizer, punctuation boundaries, input artifact name, and object store reference.

## Artifacts

```text
rechunk_documents_result
rechunked_chunk_keys
```

`rechunked_chunk_keys` is usually consumed by:

```text
PersistChunks
ExtractEntities
ExtractRelations
```

## Output

Rechunked records are `ParsedChunk` objects:

```python
ParsedChunk(
    chunk_id="chunk_rechunked_...",
    document_id="doc_...",
    text="graph extraction context",
    parent_chunk_ids=("chunk_a", "chunk_b"),
)
```

Using the same structure keeps graph steps independent of the chunking strategy.
