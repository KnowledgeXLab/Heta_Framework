# Split Documents

`SplitDocuments` turns `ParsedDocument` JSON into `ParsedChunk` JSON for retrieval, indexing, and graph extraction.

```text
ParsedDocument JSON -> ParsedChunk JSON
```

It only creates chunks. Embedding, vector indexing, full-text indexing, graph extraction, merge, and rechunking are handled by later steps.

## Contract

`SplitDocuments` uses:

```text
stores.objects
```

Default input artifact:

```text
parsed_document_keys
```

Default output prefix:

```text
chunks/
```

Execution flow:

```text
read parsed_document_keys
  -> load ParsedDocument JSON
  -> split page text into token windows
  -> write chunks/{chunk_id}.json
  -> expose chunk_keys
```

## Configuration

```python
SplitDocumentsConfig(
    chunks_prefix="chunks",
    chunk_size=1024,
    overlap=50,
    encoding_name="cl100k_base",
    split_punctuation=("。", ".", ",", "，", "!", "?", "！", "？", "\n"),
    object_store=None,
    parsed_document_keys_artifact="parsed_document_keys",
)
```

| Parameter | Meaning |
| --- | --- |
| `chunks_prefix` | Prefix for `ParsedChunk` JSON. |
| `chunk_size` | Target token count for each chunk. |
| `overlap` | Token overlap between neighboring chunks. Must be smaller than `chunk_size`. |
| `encoding_name` | Tokenizer name. `cl100k_base` is the production default. |
| `split_punctuation` | Preferred punctuation boundaries. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `parsed_document_keys_artifact` | Upstream parsed document artifact name. |

`encoding_name="unicode"` is an offline tokenizer for tests and local demos.

## Requirements

```python
StepRequirements(
    components=frozenset({store_ref("objects")}),
    artifacts=frozenset({"parsed_document_keys"}),
)
```

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({
        "split_documents_result",
        "chunk_keys",
    })
)
```

`SplitDocuments` does not unlock a query mode. It prepares chunks for later index steps.

## Artifacts

```python
SplitDocumentsResult(
    chunk_keys=("chunks/chunk_abc123.json",),
    document_count=1,
    chunk_count=12,
)
```

`chunk_keys` is the tuple consumed by embedding, indexing, and graph steps.

## Chunk Output

Each JSON file is a `ParsedChunk`:

```python
ParsedChunk(
    chunk_id="chunk_...",
    document_id="doc_...",
    source=ParsedSource(
        key="raw/paper.pdf",
        name="paper.pdf",
        file_type="pdf",
        content_sha256="...",
    ),
    page_index=0,
    chunk_index=0,
    text="...",
    token_start=0,
    token_end=512,
)
```

Default object key:

```text
chunks/{chunk_id}.json
```

| Field | Meaning |
| --- | --- |
| `chunk_id` | Stable chunk ID used by embeddings, indexes, graph facts, and citations. |
| `document_id` | Parent parsed document ID. |
| `source` | Original source metadata for filtering and citation. |
| `page_index` | Page-like unit where the chunk came from. |
| `chunk_index` | Chunk order inside the document. |
| `text` | Chunk text used by retrieval and LLM context. |
| `token_start` | Start position in the page token sequence. |
| `token_end` | End position in the page token sequence. |

`token_start` and `token_end` make chunks easier to debug, reassemble, and evaluate. They add little cost because the splitter already computes token boundaries.

## HetaDB Alignment

The default splitter follows HetaDB's base chunking behavior:

- token-based windows
- `cl100k_base`
- `chunk_size=1024`
- `overlap=50`
- punctuation-aware boundaries
- stable chunk IDs

The later HetaDB-style merge and rechunk stages are separate steps in Heta Framework.
